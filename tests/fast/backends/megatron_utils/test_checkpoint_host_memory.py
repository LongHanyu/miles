"""CPU-only save-boundary tests; real pinned-memory reclamation needs a GPU run."""

import ast
import importlib.util
import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import Mock, call, patch

ROOT = Path(__file__).resolve().parents[4]


class CheckpointHostMemoryTests(unittest.TestCase):
    def setUp(self):
        # Execute the actual methods without importing Megatron/CUDA extensions.
        path = ROOT / "miles/backends/megatron_utils/actor.py"
        tree = ast.parse(path.read_text())
        actor = next(
            node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MegatronTrainRayActor"
        )
        methods = [
            node
            for node in actor.body
            if isinstance(node, ast.FunctionDef) and node.name in {"save_model", "_finalize_pending_async_save"}
        ]
        for method in methods:
            method.decorator_list = []
        self.events = Mock()
        namespace = {name: getattr(self.events, name) for name in ("save", "clear_memory", "save_hf_model")}
        namespace["maybe_finalize_async_save"] = self.events.finalize
        exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), "exec"), namespace)
        self.save_model = namespace["save_model"]
        self.actor = SimpleNamespace(
            args=SimpleNamespace(
                debug_rollout_only=False, async_save=False, save_hf=None, custom_megatron_post_save_hook_path=None
            ),
            model=object(),
            optimizer=object(),
            opt_param_scheduler=object(),
            role="actor",
            snapshot_publisher=object(),
            _heartbeat=Mock(),
        )
        self.actor._finalize_pending_async_save = MethodType(namespace["_finalize_pending_async_save"], self.actor)
        self.save_call = call.save(
            16,
            self.actor.model,
            self.actor.optimizer,
            self.actor.opt_param_scheduler,
            snapshot_publisher=self.actor.snapshot_publisher,
        )
        self.flush = call.clear_memory(clear_host_memory=True)
        self.finalize = call.finalize(blocking=True)

    def test_synchronous_save_releases_cache_after_return(self):
        for force_sync in (False, True):
            with self.subTest(force_sync=force_sync):
                self.events.reset_mock()
                self.save_model(self.actor, 16, force_sync=force_sync)
                self.assertEqual(self.events.mock_calls, [self.save_call, self.flush])

    def test_async_save_reclaims_previous_cache_without_waiting_for_new_save(self):
        self.actor.args.async_save = True
        self.save_model(self.actor, 16)
        self.assertEqual(self.events.mock_calls, [self.finalize, self.flush, self.save_call])

    def test_forced_async_save_reclaims_both_completed_saves(self):
        self.actor.args.async_save = True
        self.save_model(self.actor, 16, force_sync=True)
        self.assertEqual(
            self.events.mock_calls, [self.finalize, self.flush, self.save_call, self.finalize, self.flush]
        )

    def test_save_failure_propagates_without_post_save_cleanup(self):
        for async_save in (False, True):
            with self.subTest(async_save=async_save):
                self.events.reset_mock()
                self.actor.args.async_save = async_save
                self.events.save.side_effect = RuntimeError("save failed")
                with self.assertRaisesRegex(RuntimeError, "save failed"):
                    self.save_model(self.actor, 16, force_sync=True)
                before = [self.finalize, self.flush] if async_save else []
                self.assertEqual(self.events.mock_calls, before + [self.save_call])

    def test_failed_async_finalization_does_not_flush_its_staging(self):
        self.actor.args.async_save = True
        self.events.finalize.side_effect = RuntimeError("finalize failed")
        with self.assertRaisesRegex(RuntimeError, "finalize failed"):
            self.save_model(self.actor, 16)
        self.assertEqual(self.events.mock_calls, [self.finalize])

    def test_failed_forced_finalization_does_not_flush_new_staging(self):
        self.actor.args.async_save = True
        self.events.finalize.side_effect = [None, RuntimeError("finalize failed")]
        with self.assertRaisesRegex(RuntimeError, "finalize failed"):
            self.save_model(self.actor, 16, force_sync=True)
        self.assertEqual(self.events.mock_calls, [self.finalize, self.flush, self.save_call, self.finalize])

    def test_debug_rollout_does_not_save_or_flush(self):
        self.actor.args.debug_rollout_only = True
        self.save_model(self.actor, 16)
        self.assertEqual(self.events.mock_calls, [])

    def test_hf_export_keeps_publisher_and_finalize_order(self):
        self.actor.args.save_hf = "test-hf"
        self.actor.args.async_save = True
        export = call.save_hf_model(self.actor.args, 16, self.actor.model, publisher=self.actor.snapshot_publisher)
        self.save_model(self.actor, 16, force_sync=True)
        self.assertEqual(
            self.events.mock_calls, [self.finalize, self.flush, self.save_call, export, self.finalize, self.flush]
        )

    def test_export_failure_is_not_masked_by_cleanup(self):
        self.actor.args.save_hf = "test-hf"
        self.events.save_hf_model.side_effect = RuntimeError("export failed")
        with self.assertRaisesRegex(RuntimeError, "export failed"):
            self.save_model(self.actor, 16, force_sync=True)
        self.events.clear_memory.assert_not_called()


class HostMemoryHelperTests(unittest.TestCase):
    def test_existing_helper_orders_sync_gc_and_cache_release(self):
        path = ROOT / "miles/utils/memory_utils.py"
        spec = importlib.util.spec_from_file_location("checkpoint_test_memory_utils", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for host in (False, True):
            with self.subTest(host=host):
                events = Mock()
                with patch.object(module, "torch", events.torch), patch.object(module, "gc", events.gc):
                    module.clear_memory(clear_host_memory=host)
                expected = [call.torch.cuda.synchronize(), call.gc.collect(), call.torch.cuda.empty_cache()]
                if host:
                    expected.append(call.torch._C._host_emptyCache())
                self.assertEqual(events.mock_calls, expected)


if __name__ == "__main__":
    unittest.main()
