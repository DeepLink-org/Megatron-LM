"""deeplink_megatron Megatron-LM pipeline scheduler extension.

Import ``deeplink_megatron.megatron_adaptor`` or ``deeplink_megatron.adapter.megatron_adaptor`` from a
Megatron entrypoint to apply runtime patches.
"""

__all__ = ["megatron_adaptor"]


def __getattr__(name):
    if name == "megatron_adaptor":
        from .adapter import megatron_adaptor

        return megatron_adaptor
    raise AttributeError(name)
