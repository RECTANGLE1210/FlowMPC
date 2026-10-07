"""CPU-only regression tests for selected replay and offline trainer hooks."""

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
import gc
import weakref
from unittest.mock import Mock, patch

import torch
from tensordict import TensorDict
from torchrl.data.replay_buffers import LazyTensorStorage

from common.buffer import Buffer
from common import TASK_SET
from common.parser import parse_cfg
from hydra import compose, initialize_config_dir
from trainer.offline_trainer import OfflineTrainer
from .fm.data import load_selected_buffer, load_selection_metadata, select_episodes
from .trainer import FlowMPCOfflineTrainer
from .replay import build_flowmpc_replay, validate_flowmpc_cfg


class TrainerTests(unittest.TestCase):
	def _trainer(self, cls, mode="frozen", num_eps=4):
		return cls(
			cfg=SimpleNamespace(flowmpc_train_mode=mode, eval_freq=2, steps=3, multitask=True, task="mt80",
				tasks=list(TASK_SET["mt80"]), flowmpc_train_task_ids=None, flowmpc_eval_task_ids=None),
			env=None, agent=SimpleNamespace(model="test model", update=Mock(return_value={})),
			buffer=SimpleNamespace(num_eps=num_eps), logger=Mock(),
		)

	def test_initial_evaluation_and_inherited_loop(self):
		self.assertIs(FlowMPCOfflineTrainer.train, OfflineTrainer.train)
		for cls, expected in ((OfflineTrainer, [1, 3]), (FlowMPCOfflineTrainer, [3])):
			trainer = self._trainer(cls)
			calls = []
			trainer.eval = lambda: calls.append(trainer.agent.update.call_count) or {}
			with patch.object(OfflineTrainer, "_load_dataset"):
				trainer.train()  # Mock updates only; no environment or model training.
			self.assertEqual(calls, expected)
			self.assertEqual(trainer.logger.log.call_count, 2)

	def test_frozen_uses_preloaded_buffer(self):
		trainer = self._trainer(FlowMPCOfflineTrainer)
		buffer = trainer.buffer
		with patch.object(OfflineTrainer, "_load_dataset") as full_loader:
			trainer._load_dataset()
			full_loader.assert_not_called()
		self.assertIs(trainer.buffer, buffer)
		for empty in (None, SimpleNamespace(num_eps=0)):
			trainer.buffer = empty
			with self.assertRaises(AssertionError):
				trainer._load_dataset()

	def test_partial_full_delegate(self):
		for mode in ("partial", "full"):
			trainer = self._trainer(FlowMPCOfflineTrainer, mode)
			with patch.object(OfflineTrainer, "_load_dataset", autospec=True) as full_loader:
				trainer._load_dataset()
				full_loader.assert_called_once_with(trainer)

	def test_subset_trainer_keeps_preloaded_replay(self):
		for mode in ("partial", "full"):
			trainer = self._trainer(FlowMPCOfflineTrainer, mode)
			trainer.cfg.flowmpc_train_task_ids = [30, 71]
			buffer = trainer.buffer
			with patch.object(OfflineTrainer, "_load_dataset") as full_loader:
				trainer._load_dataset()
				full_loader.assert_not_called()
			self.assertIs(trainer.buffer, buffer)

	def test_evaluation_task_index_hook_and_loop(self):
		standard = self._trainer(OfflineTrainer)
		flowmpc = self._trainer(FlowMPCOfflineTrainer)
		self.assertIs(FlowMPCOfflineTrainer.eval, OfflineTrainer.eval)
		self.assertEqual(list(standard._eval_task_indices()), list(range(80)))
		self.assertEqual(list(flowmpc._eval_task_indices()), list(range(80)))
		flowmpc.cfg.flowmpc_eval_task_ids = [71, 30]
		self.assertEqual(flowmpc._eval_task_indices(), [71, 30])
		flowmpc.cfg.eval_episodes = 1
		flowmpc.env = Mock()
		flowmpc.env.reset.return_value = torch.zeros(3)
		flowmpc.env.step.return_value = (torch.zeros(3), 1., True, {"success": 0.})
		flowmpc.agent.act = Mock(return_value=torch.zeros(2))
		with patch("torch.compiler.cudagraph_mark_step_begin"):
			results = flowmpc.eval()
		self.assertEqual([call.args[0] for call in flowmpc.env.reset.call_args_list], [71, 30])
		self.assertEqual([call.kwargs["task"] for call in flowmpc.agent.act.call_args_list], [71, 30])
		self.assertEqual(set(results), {
			f"{metric}+{TASK_SET['mt80'][task_id]}"
			for metric in ("episode_reward", "episode_success") for task_id in (71, 30)
		})
		self.assertEqual(flowmpc.cfg.tasks, TASK_SET["mt80"])


class ConfigTests(unittest.TestCase):
	def test_hydra_cli_keeps_global_mt80_ordering(self):
		with initialize_config_dir(version_base=None, config_dir=str(Path(__file__).resolve().parents[1])):
			cfg = compose(config_name="flowmpc/config", overrides=[
				"task=mt80", "model_size=48", "obs=state", "flowmpc_train_mode=full",
				"flowmpc_train_task_ids=[71,30]", "flowmpc_eval_task_ids=[30]",
			])
			self.assertNotIn("flowmpc", cfg)
			with patch("hydra.utils.get_original_cwd", return_value=str(Path.cwd())):
				parsed = parse_cfg(cfg)
		validate_flowmpc_cfg(parsed)
		self.assertEqual(parsed.tasks, TASK_SET["mt80"])
		self.assertEqual(parsed.flowmpc_train_task_ids, [71, 30])
		self.assertEqual(parsed.flowmpc_eval_task_ids, [30])

	def test_invalid_mode_and_task_ids(self):
		cfg = SimpleNamespace(task="mt80", tasks=list(TASK_SET["mt80"]), flowmpc_train_mode="frozen",
			flowmpc_train_task_ids=None, flowmpc_eval_task_ids=None)
		for mode in ("unknown", None, []):
			cfg.flowmpc_train_mode = mode
			with self.assertRaises(ValueError):
				validate_flowmpc_cfg(cfg)
		cfg.flowmpc_train_mode = "frozen"
		for name in ("flowmpc_train_task_ids", "flowmpc_eval_task_ids"):
			for ids in ([30, 30], [-1], [80], [True], ["30"], [], 30):
				with self.subTest(name=name, ids=ids):
					setattr(cfg, name, ids)
					with self.assertRaises(ValueError):
						validate_flowmpc_cfg(cfg)
			setattr(cfg, name, None)
		cfg.tasks = list(reversed(cfg.tasks))
		with self.assertRaises(ValueError):
			validate_flowmpc_cfg(cfg)


def _cpu_storage(buffer, _):
	return buffer._reserve_buffer(LazyTensorStorage(buffer.capacity, device="cpu"))


class TaskSubsetReplayTests(unittest.TestCase):
	def setUp(self):
		self.directory = TemporaryDirectory()
		self.addCleanup(self.directory.cleanup)
		root = Path(self.directory.name)
		data_dir = root / "data"
		data_dir.mkdir()
		self.cfg = SimpleNamespace(
			task="mt80", tasks=list(TASK_SET["mt80"]), data_dir=str(data_dir), multitask=True,
			flowmpc_train_mode="frozen", flowmpc_train_task_ids=None, flowmpc_eval_task_ids=None,
			flow_quality_fraction=.5, flow_max_episodes_per_task=None,
			flow_horizon=3, flow_batch_size=4, horizon=2, batch_size=2,
			buffer_size=100, steps=3, fm_checkpoint=str(root / "fm.pt"),
		)
		for chunk in range(2):
			ids = torch.arange(80)
			values = ids + chunk * 80
			td = TensorDict({
				"obs": values[:, None, None].expand(80, 101, 3).float().clone(),
				"action": torch.zeros(80, 101, 2),
				"reward": values[:, None].expand(80, 101).float().clone(),
				"task": ids[:, None].expand(80, 101).clone(),
			}, batch_size=[80, 101])
			torch.save(td, data_dir / f"chunk{chunk}.pt")
		select_episodes(self.cfg, root / "fm_selection.pt")

	def _build(self):
		with patch.object(Buffer, "_init", _cpu_storage):
			buffer = build_flowmpc_replay(self.cfg)
		buffer._device = torch.device("cpu")
		return buffer

	def _assert_subset(self, buffer, episode_count, values):
		self.assertEqual(buffer.num_eps, episode_count)
		self.assertEqual(buffer.capacity, episode_count * 101)
		td = buffer._buffer._storage[:]
		self.assertEqual(set(td["task"].tolist()), {30, 71})
		self.assertEqual(set(td["obs"][:, 0].unique().tolist()), set(values))
		self.assertEqual(td["episode"].unique(return_counts=True)[1].tolist(), [101] * episode_count)
		_, action, _, _, tasks = buffer.sample()
		self.assertEqual(action.shape, (self.cfg.horizon, self.cfg.batch_size, 2))
		self.assertTrue(set(tasks.tolist()).issubset({30, 71}))
		self.assertEqual(self.cfg.tasks, TASK_SET["mt80"])
		self.assertEqual(self.cfg.steps, 3)

	def test_partial_full_subsets_keep_all_episodes_and_ids(self):
		for mode in ("partial", "full"):
			self.cfg.flowmpc_train_mode = mode
			self.cfg.flowmpc_train_task_ids = [71, 30]
			self._assert_subset(self._build(), 4, [30, 71, 110, 151])

	def test_frozen_selected_subset(self):
		self.cfg.flowmpc_train_task_ids = [71, 30]
		self._assert_subset(self._build(), 2, [110, 151])

	def test_frozen_null_keeps_all_selected_tasks(self):
		buffer = self._build()
		self.assertEqual(buffer.num_eps, 80)
		self.assertEqual(buffer.capacity, 80 * 101)
		self.assertEqual(set(buffer._buffer._storage[:]["task"].tolist()), set(range(80)))
		self.assertEqual((buffer.cfg.horizon, buffer.cfg.batch_size), (2, 2))

	def test_null_partial_full_requests_official_loader(self):
		for mode in ("partial", "full"):
			self.cfg.flowmpc_train_mode = mode
			with patch("flowmpc.replay.Buffer") as buffer_constructor:
				self.assertIsNone(build_flowmpc_replay(self.cfg))
				buffer_constructor.assert_not_called()
			trainer = FlowMPCOfflineTrainer(self.cfg, None, SimpleNamespace(model="test model"), None, Mock())
			with patch.object(OfflineTrainer, "_load_dataset", autospec=True) as loader:
				trainer._load_dataset()
				loader.assert_called_once_with(trainer)

	def test_official_loader_capacity_is_unchanged(self):
		self.cfg.flowmpc_train_mode = "full"
		trainer = FlowMPCOfflineTrainer(self.cfg, None, SimpleNamespace(model="test model"), None, Mock())
		buffer = Mock(num_eps=0)
		def record_load(td):
			buffer.num_eps += len(td)
		with patch("trainer.offline_trainer.Buffer", return_value=buffer) as constructor:
			buffer.load.side_effect = record_load
			trainer._load_dataset()
		buffer_cfg = constructor.call_args.args[0]
		self.assertEqual(buffer_cfg.buffer_size, 550_450_000)
		self.assertEqual(buffer_cfg.steps, 550_450_000)
		self.assertEqual(buffer_cfg.episode_length, 101)
		self.assertEqual(buffer.num_eps, 160)
		self.assertEqual(self.cfg.steps, 3)

	def test_subset_releases_each_source_chunk(self):
		self.cfg.flowmpc_train_mode = "partial"
		self.cfg.flowmpc_train_task_ids = [30, 71]
		original_load = torch.load
		refs = []
		def tracked_load(*args, **kwargs):
			gc.collect()
			self.assertTrue(all(reference() is None for reference in refs))
			td = original_load(*args, **kwargs)
			refs.append(weakref.ref(td))
			return td
		with patch("flowmpc.replay.torch.load", side_effect=tracked_load):
			self._build()
		gc.collect()
		self.assertTrue(all(reference() is None for reference in refs))


class SelectedReplayTests(unittest.TestCase):
	def setUp(self):
		self.directory = TemporaryDirectory()
		self.addCleanup(self.directory.cleanup)
		root = Path(self.directory.name)
		data_dir = root / "data"
		data_dir.mkdir()
		self.cfg = SimpleNamespace(
			tasks=["task-a", "task-b"], data_dir=str(data_dir), multitask=True,
			flow_quality_fraction=.5, flow_max_episodes_per_task=2,
			flow_horizon=3, flow_batch_size=4, horizon=2, batch_size=2,
			buffer_size=100, steps=3,
		)
		for chunk in range(2):
			ids = torch.arange(6) + chunk * 6
			td = TensorDict({
				"obs": ids[:, None, None].expand(6, 5, 3).float().clone(),
				"action": torch.zeros(6, 5, 2),
				"reward": ids[:, None].expand(6, 5).float().clone(),
				"task": (ids % 2)[:, None].expand(6, 5).clone(),
			}, batch_size=[6, 5])
			torch.save(td, data_dir / f"chunk{chunk}.pt")
		self.metadata_path = root / "fm_selection.pt"
		self.metadata = select_episodes(self.cfg, self.metadata_path)

	def test_metadata_validation(self):
		metadata = load_selection_metadata(self.cfg, self.metadata_path)
		self.assertEqual(metadata["tasks"], self.cfg.tasks)
		for change in (
			{"tasks": list(reversed(self.cfg.tasks))}, {"format_version": -1},
			{"source_identity": []}, {"episodes_per_task": 0},
			{"files": {"chunk1.pt": torch.tensor([0, 0, 1, 2])}},
			{"files": {"chunk1.pt": torch.tensor([0, 1])}},
		):
			with self.subTest(change=change):
				bad = {**self.metadata, **change}
				torch.save(bad, self.metadata_path)
				with self.assertRaises(ValueError):
					load_selection_metadata(self.cfg, self.metadata_path)

	def test_fm_defaults_and_world_model_overrides(self):
		def cpu_storage(buffer, _):
			return buffer._reserve_buffer(LazyTensorStorage(buffer.capacity, device="cpu"))
		original = deepcopy(vars(self.cfg))
		for overrides, horizon, batch_size in (
			({}, self.cfg.flow_horizon, self.cfg.flow_batch_size),
			({"horizon": self.cfg.horizon, "batch_size": self.cfg.batch_size}, self.cfg.horizon, self.cfg.batch_size),
		):
			with self.subTest(overrides=overrides), patch.object(Buffer, "_init", cpu_storage):
				buffer = load_selected_buffer(self.cfg, self.metadata, **overrides)
				buffer._device = torch.device("cpu")  # Test-only: production Buffer is unchanged.
				self.assertEqual(buffer.num_eps, 4)
				self.assertEqual(buffer.capacity, 20)
				self.assertEqual((buffer.cfg.horizon, buffer.cfg.batch_size), (horizon, batch_size))
				obs, action, reward, _, task = buffer.sample()
				self.assertEqual(obs.shape, (horizon + 1, batch_size, 3))
				self.assertEqual(action.shape, (horizon, batch_size, 2))
				self.assertEqual(reward.shape, (horizon, batch_size, 1))
				self.assertEqual(task.shape, (batch_size,))
				loaded = buffer._buffer._storage[:]["obs"][:, 0].unique().sort().values
				selected = torch.cat([
					torch.load(Path(self.cfg.data_dir) / name, weights_only=False)[indices]["obs"][:, 0, 0]
					for name, indices in self.metadata["files"].items()
				]).sort().values
				self.assertTrue(torch.equal(loaded, selected))
		self.assertEqual(vars(self.cfg), original)


if __name__ == "__main__":
	unittest.main()
