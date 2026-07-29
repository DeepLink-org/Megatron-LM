import argparse
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import Mock, patch, sentinel

DEEPLINK_MEGATRON_ROOT = Path(__file__).parents[3] / "deeplink-megatron"
sys.path.insert(0, str(DEEPLINK_MEGATRON_ROOT))

from deeplink_megatron.arguments import add_deeplink_megatron_args  # noqa: E402
from deeplink_megatron.core.pipeline_parallel.selector import (  # noqa: E402
    get_forward_backward_func_wrapper,
)


def _training_module(backend, *, activation_offload=False):
    training = types.ModuleType("megatron.training")
    training.get_args = lambda: SimpleNamespace(
        pipeline_schedule_backend=backend,
        deeplink_megatron_enable_activation_offload=activation_offload,
    )
    return training


class TestArguments(TestCase):
    def test_argument_options_use_hyphens(self):
        parser = add_deeplink_megatron_args(argparse.ArgumentParser())

        deeplink_options = [
            option
            for option in parser._option_string_actions
            if option.startswith("--deeplink")
        ]
        self.assertTrue(deeplink_options)
        self.assertTrue(
            all("deeplink_megatron" not in option for option in deeplink_options)
        )

        args = parser.parse_args(
            [
                "--deeplink-megatron-enable-activation-offload",
                "--deeplink-megatron-offload-min-bytes",
                "42",
            ]
        )
        self.assertTrue(args.deeplink_megatron_enable_activation_offload)
        self.assertEqual(args.deeplink_megatron_offload_min_bytes, 42)


class TestSelector(TestCase):
    def test_vanilla_selector_forwards_arguments_to_upstream(self):
        upstream = Mock(return_value=sentinel.schedule)
        modules = {"megatron.training": _training_module("vanilla")}

        with patch.dict(sys.modules, modules):
            result = get_forward_backward_func_wrapper(upstream)(4, vp_size=None)

        self.assertIs(result, sentinel.schedule)
        upstream.assert_called_once_with(4, vp_size=None)

    def test_activation_offload_selector_forwards_arguments(self):
        deeplink_selector = Mock(return_value=sentinel.schedule)
        schedules = types.ModuleType("deeplink_megatron.core.pipeline_parallel.schedules")
        schedules.get_forward_backward_func = deeplink_selector
        modules = {
            "megatron.training": _training_module("vanilla", activation_offload=True),
            "deeplink_megatron.core.pipeline_parallel.schedules": schedules,
        }

        with patch.dict(sys.modules, modules):
            result = get_forward_backward_func_wrapper(Mock())(pp_size=4, vp_size=2)

        self.assertIs(result, sentinel.schedule)
        deeplink_selector.assert_called_once_with(pp_size=4, vp_size=2)

    def test_interleaved_backend_returns_deeplink_schedule(self):
        interleaved_schedule = Mock()
        schedules = types.ModuleType("deeplink_megatron.core.pipeline_parallel.schedules")
        schedules.forward_backward_pipelining_with_interleaving = interleaved_schedule
        modules = {
            "megatron.training": _training_module("interleaved_1f1b"),
            "deeplink_megatron.core.pipeline_parallel.schedules": schedules,
        }
        upstream = Mock()

        for call_args in ((), (4,)):
            with self.subTest(call_args=call_args), patch.dict(sys.modules, modules):
                result = get_forward_backward_func_wrapper(upstream)(*call_args)

            self.assertIs(result, interleaved_schedule)
        upstream.assert_not_called()


if __name__ == "__main__":
    main()
