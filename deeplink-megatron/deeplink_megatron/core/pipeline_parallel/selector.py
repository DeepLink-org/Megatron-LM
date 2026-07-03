"""Pipeline scheduler selector patch."""

from functools import wraps


def _get_backend(args):
    return getattr(args, "pipeline_schedule_backend", "vanilla")


def get_forward_backward_func_wrapper(get_forward_backward_func):
    @wraps(get_forward_backward_func)
    def wrapper():
        from megatron.training import get_args

        args = get_args()
        backend = _get_backend(args)

        if backend == "vanilla":
            if getattr(args, "deeplink_megatron_enable_activation_offload", False):
                from deeplink_megatron.core.pipeline_parallel.schedules import (
                    get_forward_backward_func as get_deeplink_megatron_forward_backward_func,
                )

                return get_deeplink_megatron_forward_backward_func()
            return get_forward_backward_func()
        if backend == "interleaved_1f1b":
            from deeplink_megatron.core.pipeline_parallel.schedules import (
                forward_backward_pipelining_with_interleaving,
            )

            return forward_backward_pipelining_with_interleaving

        raise ValueError(f"Unsupported pipeline schedule backend: {backend}")

    return wrapper
