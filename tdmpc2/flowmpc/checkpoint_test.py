"""Synthetic CPU checkpoint, resume, sampler-RNG and terminal logging tests."""

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import io
from pathlib import Path
import random
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictParams
from torchrl.data.replay_buffers import LazyTensorStorage

from common import TASK_SET
from common.buffer import Buffer
from common.logger import Logger
from common.scale import RunningScale
from trainer.offline_trainer import OfflineTrainer
from .checkpoint import capture_rng, load_training_checkpoint, restore_rng, save_training_checkpoint, training_provenance
from .terminal import terminal_log
from .trainer import FlowMPCOfflineTrainer


class _Model(torch.nn.Module):
	def __init__(self):
		super().__init__()
		self.encoder = torch.nn.Linear(2, 2).requires_grad_(False)
		self._pi_bc = torch.nn.Linear(2, 1)
		self._target_Qs_params = TensorDictParams(TensorDict({"weight": torch.tensor([7.])}, []), no_convert=True)


class _Agent:
	def __init__(self, cfg):
		self.model = _Model()
		self.optim = torch.optim.Adam(self.model._pi_bc.parameters(), lr=.01)
		# Use the real RunningScale state API, constructing its buffers on CPU for this test.
		self.scale = RunningScale.__new__(RunningScale)
		torch.nn.Module.__init__(self.scale)
		self.scale.cfg = cfg
		self.scale.value = torch.nn.Buffer(torch.ones(1))
		self.scale._percentiles = torch.nn.Buffer(torch.tensor([5., 95.]))
		self.fm_policy = torch.nn.Linear(2, 1).eval().requires_grad_(False)
		self.fm_policy.load_state_dict(torch.load(cfg.fm_checkpoint, weights_only=True))
		self.calls = 0

	def update(self, buffer):
		obs, action, _, _, _ = buffer.sample()
		noise = (random.random() + np.random.rand() + torch.rand(())) * .01
		loss = (self.model._pi_bc(self.model.encoder(obs[:-1])) - action + noise).square().mean()
		self.optim.zero_grad(set_to_none=True)
		loss.backward()
		self.optim.step()
		self.scale.value.add_(.25)
		self.calls += 1
		return {}


def _same(test, left, right):
	if torch.is_tensor(left):
		test.assertTrue(torch.equal(left, right))
	elif isinstance(left, dict):
		test.assertEqual(left.keys(), right.keys())
		for key in left:
			_same(test, left[key], right[key])
	elif isinstance(left, (tuple, list)):
		test.assertEqual(len(left), len(right))
		for a, b in zip(left, right):
			_same(test, a, b)
	else:
		test.assertEqual(left, right)


class CheckpointTests(unittest.TestCase):
	def setUp(self):
		self.directory = TemporaryDirectory()
		self.addCleanup(self.directory.cleanup)
		self.root = Path(self.directory.name)
		(self.root / "data").mkdir()
		torch.save(torch.nn.Linear(2, 1).state_dict(), self.root / "fm.pt")
		torch.save({"selection": "test fixture"}, self.root / "fm_selection.pt")
		self.td = TensorDict({
			"obs": torch.arange(60).view(6, 5, 2).float() / 60,
			"action": torch.zeros(6, 5, 1), "reward": torch.zeros(6, 5),
			"task": torch.tensor([30, 71, 30, 71, 30, 71])[:, None].expand(6, 5).clone(),
		}, [6, 5])
		torch.save(self.td, self.root / "data" / "chunk.pt")
		self.cuda = patch("torch.cuda.is_available", return_value=False)
		self.cuda.start()
		self.addCleanup(self.cuda.stop)

	def cfg(self, **overrides):
		values = dict(
			task="mt80", tasks=list(TASK_SET["mt80"]), multitask=True, flowmpc_train_mode="frozen",
			flowmpc_train_task_ids=[30, 71], flowmpc_eval_task_ids=None, flowmpc_save_freq=10000,
			flowmpc_resume_checkpoint=None, work_dir=self.root / "work", data_dir=str(self.root / "data"),
			fm_checkpoint=str(self.root / "fm.pt"), model_size=48, obs="state", horizon=2, batch_size=2,
			steps=20, eval_freq=1000000, eval_episodes=1, enable_wandb=False, tau=.01, seed=1,
			obs_shape={"state": (2,)}, action_dim=1, task_title="MT80", exp_name="test",
			save_csv=False, save_agent=True, save_video=False, wandb_project="debug", wandb_entity="debug",
		)
		values.update(overrides)
		cfg = SimpleNamespace(**values)
		cfg.get = lambda key, default=None: getattr(cfg, key, default)
		return cfg

	def buffer(self, cfg):
		buffer_cfg = deepcopy(cfg)
		buffer_cfg.buffer_size = buffer_cfg.steps = 30
		def cpu_storage(buffer, _):
			buffer._storage_device = torch.device("cpu")
			return buffer._reserve_buffer(LazyTensorStorage(buffer.capacity, device="cpu"))
		with patch.object(Buffer, "_init", cpu_storage):
			buffer = Buffer(buffer_cfg)
			buffer.load(self.td.clone())
		buffer._device = torch.device("cpu")
		return buffer

	def trainer(self, cfg, agent=None, logger=None):
		return FlowMPCOfflineTrainer(cfg, None, agent or _Agent(cfg), self.buffer(cfg), logger or Mock())

	def test_periodic_latest_only_and_failed_save_preserves_previous(self):
		cfg = self.cfg()
		trainer = self.trainer(cfg)
		trainer._initial_update_index()
		trainer._after_update(10000)
		trainer._after_update(20000)
		path = cfg.work_dir / "models" / "latest.pt"
		payload = torch.load(path, weights_only=False)
		self.assertEqual(payload["update"], 20000)
		self.assertEqual(list(path.parent.iterdir()), [path])
		self.assertTrue(any("_target_Qs_params" in key for key in payload["model"]))
		self.assertTrue(any("_pi_bc" in key for key in payload["model"]))
		self.assertFalse(any("fm_policy" in key for key in payload["model"]))
		before = path.read_bytes()
		def incomplete_save(_, file):
			file.write(b"incomplete")
			raise OSError("synthetic disk failure")
		with patch("flowmpc.checkpoint.torch.save", side_effect=incomplete_save):
			with self.assertRaises(OSError):
				trainer._after_update(30000)
		self.assertEqual(path.read_bytes(), before)
		self.assertEqual(list(path.parent.iterdir()), [path])
		with patch("flowmpc.checkpoint.os.replace", side_effect=OSError("synthetic rename failure")):
			with self.assertRaises(OSError):
				trainer._after_update(30000)
		self.assertEqual(path.read_bytes(), before)
		self.assertEqual(list(path.parent.iterdir()), [path])

	def test_final_non_multiple_and_wandb_disabled(self):
		cfg = self.cfg(steps=3, flowmpc_save_freq=2)
		logger = Logger(cfg)
		trainer = self.trainer(cfg, logger=logger)
		with patch.object(logger, "save_agent", wraps=logger.save_agent) as standard_save:
			trainer.train()
			standard_save.assert_called_once_with(None)
		path = cfg.work_dir / "models" / "latest.pt"
		self.assertEqual(torch.load(path, weights_only=False)["update"], 3)
		self.assertEqual(list(path.parent.iterdir()), [path])

	def test_zero_frequency_still_saves_final(self):
		cfg = self.cfg(steps=1, flowmpc_save_freq=0)
		trainer = self.trainer(cfg)
		trainer.train()
		self.assertEqual(torch.load(cfg.work_dir / "models" / "latest.pt", weights_only=False)["update"], 1)
		trainer.logger.finish.assert_called_once_with()
		trainer.logger.save_agent.assert_not_called()

	def test_resume_weights_optimizer_scale_count_and_frozen_fm(self):
		cfg = self.cfg()
		trainer = self.trainer(cfg)
		trainer.agent.update(trainer.buffer)
		trainer.agent.model._target_Qs_params.get("weight").fill_(12.)
		trainer.agent.scale._percentiles.fill_(17.)
		path = save_training_checkpoint(cfg, trainer.agent, trainer.buffer, 7, trainer._provenance())
		resumed = self.trainer(self.cfg(flowmpc_resume_checkpoint=str(path)))
		fm_before = deepcopy(resumed.agent.fm_policy.state_dict())
		self.assertEqual(resumed._initial_update_index(), 7)
		_same(self, trainer.agent.model.state_dict(), resumed.agent.model.state_dict())
		_same(self, trainer.agent.optim.state_dict(), resumed.agent.optim.state_dict())
		_same(self, trainer.agent.scale.state_dict(), resumed.agent.scale.state_dict())
		_same(self, fm_before, resumed.agent.fm_policy.state_dict())
		self.assertFalse(resumed.agent.fm_policy.training)
		self.assertFalse(any(value.requires_grad for value in resumed.agent.fm_policy.parameters()))

	def test_resume_200000_runs_only_remaining_312000(self):
		cfg = self.cfg(steps=512000, flowmpc_save_freq=0)
		trainer = self.trainer(cfg)
		path = save_training_checkpoint(cfg, trainer.agent, trainer.buffer, 200000, trainer._provenance())
		resumed = self.trainer(self.cfg(steps=512000, flowmpc_save_freq=0, flowmpc_resume_checkpoint=str(path)))
		def count_only(_):
			resumed.agent.calls += 1  # No model training: synthetic update counter only.
			return {}
		resumed.agent.update = count_only
		resumed.train()
		self.assertEqual(resumed.agent.calls, 312000)
		self.assertEqual(resumed._completed_updates, 512000)

	def test_incompatible_mode_subset_groups_and_fm_identity(self):
		cfg = self.cfg()
		trainer = self.trainer(cfg)
		path = save_training_checkpoint(cfg, trainer.agent, trainer.buffer, 1, trainer._provenance())
		for change in ({"flowmpc_train_mode": "full"}, {"flowmpc_train_task_ids": [30]}, {"tasks": list(reversed(cfg.tasks))}):
			bad_cfg = self.cfg(**change)
			bad_agent = _Agent(bad_cfg)
			with self.assertRaises(ValueError):
				load_training_checkpoint(path, bad_cfg, bad_agent, training_provenance(bad_cfg, bad_agent))
		bad_agent = _Agent(cfg)
		bad_agent.optim.param_groups[0]["lr"] = .02
		with self.assertRaises(ValueError):
			load_training_checkpoint(path, cfg, bad_agent, training_provenance(cfg, bad_agent))
		torch.save(torch.nn.Linear(2, 1).state_dict(), cfg.fm_checkpoint)
		with self.assertRaises(ValueError):
			load_training_checkpoint(path, cfg, trainer.agent, training_provenance(cfg, trainer.agent))

	def test_cpu_continuation_matches_uninterrupted_updates(self):
		def seed():
			random.seed(9)
			np.random.seed(9)
			torch.manual_seed(9)
		seed()
		continuous = self.trainer(self.cfg(steps=10, flowmpc_save_freq=0, work_dir=self.root / "continuous"))
		continuous.train()
		seed()
		first = self.trainer(self.cfg(steps=4, flowmpc_save_freq=0))
		first.train()
		path = first.cfg.work_dir / "models" / "latest.pt"
		random.random(); np.random.rand(); torch.rand(5)  # Simulate initialization consuming RNG.
		resumed = self.trainer(self.cfg(steps=10, flowmpc_save_freq=0, flowmpc_resume_checkpoint=str(path)))
		resumed.train()
		self.assertEqual(resumed.agent.calls, 6)
		_same(self, continuous.agent.model.state_dict(), resumed.agent.model.state_dict())
		_same(self, continuous.agent.optim.state_dict(), resumed.agent.optim.state_dict())
		_same(self, continuous.agent.scale.state_dict(), resumed.agent.scale.state_dict())

	def test_sampler_rng_continuation_global_and_explicit_generator(self):
		cfg = self.cfg()
		for private in (False, True):
			buffer = self.buffer(cfg)
			if private:
				buffer._buffer.set_rng(torch.Generator().manual_seed(5))
			buffer.sample()  # Warm SliceSampler cache before checkpoint.
			rng = capture_rng(buffer)
			expected = buffer.sample()[0]
			expected_random = (random.random(), np.random.rand(), torch.rand(3))
			fresh = self.buffer(cfg)
			random.random(); np.random.rand(); torch.rand(5)
			restore_rng(rng, fresh)
			self.assertTrue(torch.equal(fresh.sample()[0], expected))
			self.assertEqual(random.random(), expected_random[0])
			self.assertEqual(np.random.rand(), expected_random[1])
			self.assertTrue(torch.equal(torch.rand(3), expected_random[2]))

	def test_standard_trainer_saves_and_finalizes_as_before(self):
		cfg = self.cfg(steps=5, eval_freq=2)
		trainer = OfflineTrainer(cfg, None, _Agent(cfg), self.buffer(cfg), Mock())
		trainer.eval = Mock(return_value={})
		trainer.agent.update = Mock(return_value={})
		with patch.object(OfflineTrainer, "_load_dataset"):
			trainer.train()
		self.assertEqual(trainer.eval.call_count, 3)
		self.assertEqual([call.kwargs["identifier"] for call in trainer.logger.save_agent.call_args_list], ["2", "4"])
		trainer.logger.finish.assert_called_once_with(trainer.agent)
		self.assertIs(FlowMPCOfflineTrainer.train, OfflineTrainer.train)


@unittest.skipUnless(torch.cuda.is_available(), "Actual FlowMPC checkpoint round trip requires CUDA")
class CudaCheckpointTests(unittest.TestCase):
	def test_actual_model_optimizer_and_rng_resume_all_modes(self):
		from .agent import FlowMPC
		from .fm.policy import MultiTaskFlowMatchingPolicy
		from .smoke_test import _cfg

		with TemporaryDirectory() as directory:
			root = Path(directory)
			(root / "data").mkdir()
			torch.save({}, root / "data" / "chunk.pt")
			torch.save({}, root / "fm_selection.pt")
			MultiTaskFlowMatchingPolicy(_cfg()).save(root / "fm.pt")
			for mode in ("frozen", "partial", "full"):
				cfg = _cfg(mode, root / "fm.pt")
				cfg.work_dir, cfg.data_dir = root / mode, str(root / "data")
				cfg.steps = cfg.buffer_size = 20
				agent = FlowMPC(cfg)
				# Initialize real capturable CUDA Adam state without running training.
				for group in agent.optim.param_groups:
					for value in group["params"]:
						value.grad = torch.ones_like(value)
				agent.optim.step()
				agent.optim.zero_grad(set_to_none=True)
				with torch.no_grad():
					for value in agent.model._target_Qs_params.values(include_nested=True, leaves_only=True):
						value.add_(1.)
				agent.scale.value.fill_(3.)
				agent.scale._percentiles.fill_(7.)
				buffer = Buffer(cfg)
				buffer.load(TensorDict({
					"obs": torch.zeros(2, 5, 3), "action": torch.zeros(2, 5, 4),
					"reward": torch.zeros(2, 5), "task": torch.tensor([0, 1])[:, None].expand(2, 5),
				}, [2, 5]))
				buffer.sample()  # Warm the real CUDA-storage SliceSampler cache.
				path = save_training_checkpoint(cfg, agent, buffer, 7, training_provenance(cfg, agent))
				expected_sample = buffer.sample()[0]
				expected_rng = torch.rand(3, device="cuda")
				restored = FlowMPC(cfg)
				count, rng = load_training_checkpoint(path, cfg, restored, training_provenance(cfg, restored))
				self.assertEqual(count, 7)
				_same(self, agent.model.state_dict(), restored.model.state_dict())
				_same(self, agent.optim.state_dict(), restored.optim.state_dict())
				_same(self, agent.scale.state_dict(), restored.scale.state_dict())
				_same(self, agent.fm_policy.state_dict(), restored.fm_policy.state_dict())
				for key, value in agent.model._target_Qs_params.items(include_nested=True, leaves_only=True):
					self.assertTrue(torch.equal(value, restored.model._target_Qs_params.get(key)))
				for key, value in restored.model._detach_Qs_params.items(include_nested=True, leaves_only=True):
					self.assertTrue(torch.equal(value, restored.model._Qs.params.get(key)))
				restore_rng(rng, buffer)
				self.assertTrue(torch.equal(expected_sample, buffer.sample()[0]))
				self.assertTrue(torch.equal(expected_rng, torch.rand(3, device="cuda")))
				self.assertFalse(restored.fm_policy.training)
				self.assertFalse(any(value.requires_grad for value in restored.fm_policy.parameters()))


class TerminalTests(unittest.TestCase):
	def test_stdout_stderr_traceback_separate_logs_and_stream_cleanup(self):
		with TemporaryDirectory() as directory:
			out, err = io.StringIO(), io.StringIO()
			with redirect_stdout(out), redirect_stderr(err):
				with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
					with terminal_log(directory) as first:
						print("synthetic stdout")
						print("synthetic stderr", file=sys.stderr)
						self.assertIn("synthetic stdout", first.read_text())  # Flushed immediately.
						self.assertEqual(sys.stdout.isatty(), out.isatty())
						raise RuntimeError("synthetic failure")
				self.assertIs(sys.stdout, out)
				self.assertIs(sys.stderr, err)
				with terminal_log(directory) as second:
					print("second launch")
				self.assertIs(sys.stdout, out)
				self.assertIs(sys.stderr, err)
			text = first.read_text()
			for content in ("synthetic stdout", "synthetic stderr", "Traceback", "synthetic failure", "FlowMPC run failed"):
				self.assertIn(content, text)
			self.assertIn("synthetic stdout", out.getvalue())
			self.assertIn("synthetic stderr", err.getvalue())
			self.assertNotEqual(first, second)
			self.assertEqual(len(list((Path(directory) / "terminal").glob("*.log"))), 2)
			self.assertIn("completed successfully", second.read_text())


if __name__ == "__main__":
	unittest.main()
