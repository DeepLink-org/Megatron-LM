"""Apply deeplink_megatron patches to Megatron-LM.

Importing this module is intentionally narrow: it only patches Megatron argument
registration/validation and the pipeline scheduler selector.
"""

from functools import wraps

from deeplink_megatron.adapter.patch_utils import MegatronPatchesManager
from deeplink_megatron.arguments import chain_extra_args_provider
from deeplink_megatron.core.pipeline_parallel.selector import get_forward_backward_func_wrapper


def apply_numpy_compatibility_patches():
    import numpy as np

    if not hasattr(np, "product") and hasattr(np, "prod"):
        np.product = np.prod


def parse_args_wrapper(parse_args):
    @wraps(parse_args)
    def wrapper(extra_args_provider=None, ignore_unknown_args=False):
        return parse_args(
            extra_args_provider=chain_extra_args_provider(extra_args_provider),
            ignore_unknown_args=ignore_unknown_args,
        )

    return wrapper


def validate_args_wrapper(validate_args):
    @wraps(validate_args)
    def wrapper(args, defaults=None):
        if defaults is None:
            defaults = {}

        args = validate_args(args, defaults)
        backend = getattr(args, "pipeline_schedule_backend", "vanilla")

        if backend == "interleaved_1f1b":
            if args.pipeline_model_parallel_size <= 1:
                raise AssertionError(
                    "interleaved_1f1b requires --pipeline-model-parallel-size > 1"
                )
            if args.virtual_pipeline_model_parallel_size is None:
                raise AssertionError(
                    "interleaved_1f1b requires virtual pipeline parallelism"
                )

        if getattr(args, "deeplink_megatron_enable_activation_offload", False):
            if args.deeplink_megatron_offload_min_bytes < 0:
                raise AssertionError("--deeplink_megatron-offload-min-bytes must be non-negative")
            if not 0.0 <= args.deeplink_megatron_activation_offload_ratio <= 1.0:
                raise AssertionError("--deeplink_megatron-activation-offload-ratio must be in [0, 1]")
            if args.deeplink_megatron_per_batch_offload_size < 0:
                raise AssertionError("--deeplink_megatron-per-batch-offload-size must be non-negative")
            if args.deeplink_megatron_activation_offload_stages < 1:
                raise AssertionError("--deeplink_megatron-activation-offload-stages must be >= 1")

        return args

    return wrapper


def apply_patches():
    apply_numpy_compatibility_patches()

    MegatronPatchesManager.register_patch(
        "megatron.training.arguments.parse_args",
        parse_args_wrapper,
        apply_wrapper=True,
    )
    MegatronPatchesManager.register_patch(
        "megatron.training.arguments.validate_args",
        validate_args_wrapper,
        apply_wrapper=True,
    )
    MegatronPatchesManager.register_patch(
        "megatron.core.pipeline_parallel.schedules.get_forward_backward_func",
        get_forward_backward_func_wrapper,
        apply_wrapper=True,
    )
    MegatronPatchesManager.apply_patches()


apply_patches()
