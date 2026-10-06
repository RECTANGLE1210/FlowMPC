"""CPU-only regression tests for selected replay and offline trainer hooks."""

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from tensordict import TensorDict
from torchrl.data.replay_buffers import LazyTensorStorage

from common.buffer import Buffer
from trainer.offline_trainer import OfflineTrainer
from .fm.data import load_selected_buffer, load_selection_metadata, select_episodes
from .trainer import FlowMPCOfflineTrainer


class TrainerTests(unittest.TestCase):
	def _trainer(self, cls, mode="frozen", num_eps=4):
		return cls(
			cfg=SimpleNamespace(flowmpc_train_mode=mode, eval_freq=2, steps=3, multitask=True, task="mt80"),
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
