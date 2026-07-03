"""Argument registration for the deeplink_megatron pipeline scheduler extension."""


def _has_option(parser, option_string):
    return option_string in parser._option_string_actions


def add_deeplink_megatron_args(parser):
    """Add deeplink_megatron scheduler/offload arguments to a Megatron-LM parser."""
    group = parser.add_argument_group(title="deeplink_megatron pipeline scheduler")

    if not _has_option(parser, "--pipeline-schedule-backend"):
        group.add_argument(
            "--pipeline-schedule-backend",
            "--deeplink_megatron-pipeline-schedule-backend",
            "--schedule-method",
            dest="pipeline_schedule_backend",
            default="vanilla",
            choices=["vanilla", "interleaved_1f1b"],
            help="Select Megatron vanilla scheduling or the deeplink_megatron interleaved scheduler.",
        )

    if not _has_option(parser, "--deeplink_megatron-enable-activation-offload"):
        group.add_argument(
            "--deeplink_megatron-enable-activation-offload",
            action="store_true",
            default=False,
            help="Enable deeplink_megatron scheduler-scoped activation offload/reload hooks.",
        )

    if not _has_option(parser, "--deeplink_megatron-offload-min-bytes"):
        group.add_argument(
            "--deeplink_megatron-offload-min-bytes",
            type=int,
            default=1024 * 1024,
            help="Minimum CUDA tensor size, in bytes, eligible for deeplink_megatron activation offload.",
        )

    if not _has_option(parser, "--deeplink_megatron-offload-non-contiguous"):
        group.add_argument(
            "--deeplink_megatron-offload-non-contiguous",
            action="store_true",
            default=False,
            help=(
                "Allow deeplink_megatron activation offload for non-contiguous or view-backed saved "
                "tensors. Disabled by default because some fused kernels require the "
                "original saved-tensor layout in backward."
            ),
        )

    if not _has_option(parser, "--deeplink_megatron-no-offload-pin-memory"):
        group.add_argument(
            "--deeplink_megatron-no-offload-pin-memory",
            action="store_false",
            dest="deeplink_megatron_offload_pin_memory",
            default=True,
            help="Use regular CPU memory instead of pinned memory for deeplink_megatron activation offload.",
        )

    if not _has_option(parser, "--deeplink_megatron-activation-offload-ratio"):
        group.add_argument(
            "--deeplink_megatron-activation-offload-ratio",
            type=float,
            default=1.0,
            help="Fraction of eligible saved activations to offload for each scheduler microbatch.",
        )

    if not _has_option(parser, "--deeplink_megatron-per-batch-offload-size"):
        group.add_argument(
            "--deeplink_megatron-per-batch-offload-size",
            type=float,
            default=0.0,
            help="Fixed per-microbatch activation offload budget in MiB; 0 uses ratio-based sizing.",
        )

    if not _has_option(parser, "--deeplink_megatron-activation-offload-stages"):
        group.add_argument(
            "--deeplink_megatron-activation-offload-stages",
            type=int,
            default=1,
            help="Number of copy issue groups used by the deeplink_megatron offload runtime.",
        )

    return parser


def chain_extra_args_provider(extra_args_provider):
    """Return an extra-args provider that preserves the caller provider."""

    def provider(parser):
        if extra_args_provider is not None:
            parser = extra_args_provider(parser)
        return add_deeplink_megatron_args(parser)

    return provider
