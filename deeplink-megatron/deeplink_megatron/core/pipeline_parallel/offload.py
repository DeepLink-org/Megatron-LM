"""Scheduler-scoped activation offload runtime for deeplink_megatron.

This module is adapted from the project-owned offload runtime. The deeplink_megatron version
keeps the scheduler-facing offload/reload machinery and removes dependencies
that belong outside the pipeline scheduler boundary: model activation analysis,
precision-context patching, custom comm streams, and external memcpy kernels.
"""

import contextlib
import math
from collections import defaultdict
from dataclasses import dataclass

import torch

from megatron.training import get_args


_GROUPS = {}
_OFFLOAD_TENSORS = None
_PINNED_BUFFER_POOL = defaultdict(list)
_MEMCPY_STREAMS = {}


def _get_arg(name, default=None):
    return getattr(get_args(), name, default)


def _offload_enabled():
    return bool(_get_arg("deeplink_megatron_enable_activation_offload", False))


def _offload_ratio():
    ratio = _get_arg("deeplink_megatron_activation_offload_ratio", 1.0)
    if not isinstance(ratio, float):
        from megatron.core import parallel_state

        ratio = ratio[parallel_state.get_pipeline_model_parallel_rank()]
    return max(0.0, min(1.0, ratio))


def _offload_group_count():
    return max(1, int(_get_arg("deeplink_megatron_activation_offload_stages", 1)))


def _make_key(virtual_microbatch_id, model_chunk_id):
    return model_chunk_id, virtual_microbatch_id


def _tensor_nbytes(tensor):
    return tensor.numel() * tensor.element_size()


def _is_tensor_eligible(tensor):
    if not isinstance(tensor, torch.Tensor):
        return False
    if tensor.device.type != "cuda":
        return False
    if isinstance(tensor, torch.nn.Parameter):
        return False
    if tensor.is_leaf and tensor.requires_grad:
        return False
    if not _get_arg("deeplink_megatron_offload_non_contiguous", False):
        if not tensor.is_contiguous():
            return False
        if tensor._base is not None:
            return False
        if tensor.storage_offset() != 0:
            return False
    if _tensor_nbytes(tensor) < _get_arg("deeplink_megatron_offload_min_bytes", 1024 * 1024):
        return False
    return not (tensor.dim() == 4 and tensor.shape[1] == 1 and tensor.shape[2] == 1)


def get_memcpy_stream(key):
    if key not in ("offload", "onload"):
        raise ValueError(f"unsupported memcpy stream key: {key}")
    if key not in _MEMCPY_STREAMS:
        _MEMCPY_STREAMS[key] = torch.cuda.Stream()
    return _MEMCPY_STREAMS[key]


def get_cpu_buffer(num_bytes):
    if num_bytes == 0:
        return None
    key = (num_bytes, torch.uint8)
    if _PINNED_BUFFER_POOL[key]:
        return _PINNED_BUFFER_POOL[key].pop()
    pin_memory = bool(_get_arg("deeplink_megatron_offload_pin_memory", True))
    return torch.empty(num_bytes, dtype=torch.uint8, device="cpu", pin_memory=pin_memory)


def recycle_cpu_buffer(buffer):
    if buffer is not None:
        _PINNED_BUFFER_POOL[(buffer.numel(), buffer.dtype)].append(buffer)


def fast_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    return tensor.contiguous()


def _byte_vector(tensor):
    return tensor.view(torch.uint8).reshape(-1)


class TensorWrap:
    def __init__(self, tensor):
        self.x = tensor
        self.shape = tensor.shape
        self.dtype = tensor.dtype
        self.device = tensor.device
        self.cpu_buffer = None
        self.base = None


class TensorPack:
    def __init__(self, tensor_wrap, op_name=None, **kwargs):
        self.tensor_wrap = tensor_wrap
        self.op_name = op_name
        self.extra_kwargs = kwargs

    def get(self):
        return self.tensor_wrap.x

    def __del__(self):
        if self.tensor_wrap.base is not None:
            self.tensor_wrap.base.ref_cnt -= 1


class CopyTaskGroup:
    def __init__(self, offload_group_num):
        self.offload_group_num = offload_group_num
        self.copy_task_groups = [[] for _ in range(offload_group_num)]

    def add_tensor(self, task_id, tensor_wrap):
        self.copy_task_groups[task_id % self.offload_group_num].append(tensor_wrap)

    def get_task_groups(self):
        return self.copy_task_groups


class ActivationGroup:
    def __init__(self, tensors, key, offload_groups):
        self.key = key
        self.tensors = sorted(tensors, key=lambda t: (not t.x.is_contiguous(), -t.x.numel()))
        self.offload_group_num = max(1, offload_groups)
        self.copy_task_groups = []
        self.offloaded_tensors = []
        self.cpu_buffers = []

    def _offload_budget(self):
        total_bytes = sum(_tensor_nbytes(tensor.x) for tensor in self.tensors if tensor.x is not None)
        per_batch_mib = _get_arg("deeplink_megatron_per_batch_offload_size", 0)
        if per_batch_mib and per_batch_mib > 0:
            return min(total_bytes, int(per_batch_mib * 2**20))
        return int(math.ceil(total_bytes * _offload_ratio()))

    def offload_prologue(self):
        if not self.tensors:
            self.copy_task_groups = []
            return

        budget = self._offload_budget()
        copied_bytes = 0
        groups = CopyTaskGroup(self.offload_group_num)

        for tensor_wrap in self.tensors:
            tensor = tensor_wrap.x
            if tensor is None:
                continue
            tensor_bytes = _tensor_nbytes(tensor)
            if copied_bytes + tensor_bytes > budget:
                continue
            if not tensor.is_contiguous():
                tensor_wrap.x = fast_contiguous(tensor)
                tensor = tensor_wrap.x

            cpu_buffer = get_cpu_buffer(tensor_bytes)
            tensor_wrap.cpu_buffer = cpu_buffer
            self.cpu_buffers.append(cpu_buffer)
            self.offloaded_tensors.append(tensor_wrap)
            groups.add_tensor(len(self.offloaded_tensors) - 1, tensor_wrap)
            copied_bytes += tensor_bytes

        self.copy_task_groups = groups.get_task_groups()

    def offload_issue(self, group_id):
        if not self.copy_task_groups:
            return
        stream = get_memcpy_stream("offload")
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for tensor_wrap in self.copy_task_groups[group_id]:
                source = _byte_vector(tensor_wrap.x)
                tensor_wrap.cpu_buffer.copy_(source, non_blocking=True)

    def offload_epilogue(self):
        if not self.copy_task_groups:
            return
        stream = get_memcpy_stream("offload")
        torch.cuda.current_stream().wait_stream(stream)
        for tensor_wrap in self.offloaded_tensors:
            # Drop only the saved-tensor hook reference. The same Tensor object can
            # also be a pipeline boundary tensor in scheduler-scoped hooks; resizing
            # its storage would corrupt that live alias before send/backward uses it.
            tensor_wrap.x = None

    def onload_prologue(self, *, overlap_d2h_h2d=True, ping_pong_onload=False):
        if not self.offloaded_tensors:
            return
        self.onload_stream = get_memcpy_stream("onload" if overlap_d2h_h2d else "offload")
        self.onload_stream.wait_stream(torch.cuda.current_stream())

    def onload_issue(self, group_id):
        if not self.copy_task_groups:
            return
        with torch.cuda.stream(self.onload_stream):
            for tensor_wrap in self.copy_task_groups[group_id]:
                tensor = torch.empty(tensor_wrap.shape, dtype=tensor_wrap.dtype, device=tensor_wrap.device)
                _byte_vector(tensor).copy_(tensor_wrap.cpu_buffer, non_blocking=True)
                tensor_wrap.x = tensor

    def onload_epilogue(self, stream=None, buffer=None, ping_pong_onload=False):
        if not self.offloaded_tensors:
            return
        torch.cuda.current_stream().wait_stream(self.onload_stream)
        for cpu_buffer in self.cpu_buffers:
            recycle_cpu_buffer(cpu_buffer)
        self.cpu_buffers = []
        self.offloaded_tensors = []


def pack_hook(x, op_name=None):
    global _OFFLOAD_TENSORS
    if x is None:
        return None

    tensor_wrap = TensorWrap(x)
    if _OFFLOAD_TENSORS is not None and _is_tensor_eligible(x):
        _OFFLOAD_TENSORS.append(tensor_wrap)
    return TensorPack(tensor_wrap, op_name=op_name)


def unpack_hook(tensor_pack):
    if tensor_pack is None:
        return None
    return tensor_pack.get()


def pack_hook_offload(x, op_name=None):
    return pack_hook(x, op_name=op_name)


def unpack_hook_offload(tensor_pack):
    return unpack_hook(tensor_pack)


@contextlib.contextmanager
def record(key, group_num=1):
    global _OFFLOAD_TENSORS
    if not _offload_enabled():
        yield
        return

    previous = _OFFLOAD_TENSORS
    _OFFLOAD_TENSORS = []
    try:
        with torch.autograd.graph.saved_tensors_hooks(pack_hook_offload, unpack_hook_offload):
            yield
    finally:
        _GROUPS[key] = ActivationGroup(_OFFLOAD_TENSORS, key, group_num)
        _OFFLOAD_TENSORS = previous


class OffloadAsync:
    def __init__(self, key, group_num=1):
        self.disable_offload = not _offload_enabled() or key not in _GROUPS
        if self.disable_offload:
            return
        self.issued_group = 0
        self.group_num = max(1, group_num)
        self.group = _GROUPS[key]

    def __enter__(self):
        if self.disable_offload:
            return self
        self.group.offload_prologue()
        return self

    def issue(self, group_id):
        if self.disable_offload:
            return
        while self.issued_group <= group_id and self.issued_group < self.group_num:
            self.group.offload_issue(self.issued_group)
            self.issued_group += 1

    def __exit__(self, *exc_info):
        if self.disable_offload:
            return False
        while self.issued_group < self.group_num:
            self.group.offload_issue(self.issued_group)
            self.issued_group += 1
        self.group.offload_epilogue()
        return False


class OnloadAsync:
    def __init__(self, key, group_num=1):
        self.disable_offload = not _offload_enabled() or key not in _GROUPS
        if self.disable_offload:
            return
        self.issued_group = 0
        self.group_num = max(1, group_num)
        self.key = key
        self.group = _GROUPS[key]

    def __enter__(self):
        if self.disable_offload:
            return self
        self.group.onload_prologue(overlap_d2h_h2d=True, ping_pong_onload=False)
        return self

    def issue(self, group_id):
        if self.disable_offload:
            return
        while self.issued_group <= group_id and self.issued_group < self.group_num:
            self.group.onload_issue(self.issued_group)
            self.issued_group += 1

    def __exit__(self, *exc_info):
        if self.disable_offload:
            return False
        while self.issued_group < self.group_num:
            self.group.onload_issue(self.issued_group)
            self.issued_group += 1
        self.group.onload_epilogue()
        _GROUPS.pop(self.key, None)
        return False


def get_offload_nstages():
    return _offload_group_count()


offload_ctx = None
reload_ctx = None


def issue_loads(stage):
    global offload_ctx
    global reload_ctx
    if not _offload_enabled():
        return
    if reload_ctx is not None:
        reload_ctx.issue(stage)
    if offload_ctx is not None:
        offload_ctx.issue(stage)


@dataclass
class _ForwardMicrobatchContext:
    key: object
    group_num: int
    record_ctx: object = None

    def __enter__(self):
        self.record_ctx = record(self.key, self.group_num)
        self.record_ctx.__enter__()
        return self

    def __exit__(self, *exc_info):
        self.record_ctx.__exit__(*exc_info)
        if exc_info[0] is None:
            offload = OffloadAsync(self.key, self.group_num)
            offload.__enter__()
            offload.__exit__(None, None, None)
        return False


@dataclass
class _BackwardMicrobatchContext:
    key: object
    group_num: int

    def __enter__(self):
        self.onload = OnloadAsync(self.key, self.group_num)
        self.onload.__enter__()
        self.onload.__exit__(None, None, None)
        return self

    def __exit__(self, *exc_info):
        return False


class PipelineActivationOffloadRuntime:
    def enabled(self):
        return _offload_enabled()

    def forward_microbatch(
        self,
        *,
        phase,
        virtual_microbatch_id,
        model_chunk_id,
        enabled=True,
        offload_key=None,
    ):
        if not self.enabled() or not enabled:
            return contextlib.nullcontext()
        return _ForwardMicrobatchContext(
            offload_key
            if offload_key is not None
            else _make_key(virtual_microbatch_id, model_chunk_id),
            _offload_group_count(),
        )

    def backward_microbatch(
        self,
        *,
        phase,
        virtual_microbatch_id,
        model_chunk_id,
        enabled=True,
        offload_key=None,
    ):
        if not self.enabled() or not enabled:
            return contextlib.nullcontext()
        return _BackwardMicrobatchContext(
            offload_key
            if offload_key is not None
            else _make_key(virtual_microbatch_id, model_chunk_id),
            _offload_group_count(),
        )


_RUNTIME = PipelineActivationOffloadRuntime()


def get_pipeline_offload_runtime():
    return _RUNTIME
