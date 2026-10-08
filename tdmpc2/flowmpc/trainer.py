"""Offline TD-MPC2 training with FlowMPC's preselected frozen-mode replay."""

from trainer.offline_trainer import OfflineTrainer
from .replay import resolve_task_ids, validate_flowmpc_cfg
from .checkpoint import load_training_checkpoint, restore_rng, save_training_checkpoint, training_provenance


class FlowMPCOfflineTrainer(OfflineTrainer):
	def _load_dataset(self):
		validate_flowmpc_cfg(self.cfg)
		if self.cfg.flowmpc_train_mode != "frozen" and getattr(self.cfg, "flowmpc_train_task_ids", None) is None:
			super()._load_dataset()
		else:
			assert self.buffer is not None and self.buffer.num_eps > 0, "FlowMPC requires a nonempty preloaded subset/selected buffer"

	def _eval_task_indices(self):
		if getattr(self.cfg, "flowmpc_eval_task_ids", None) is None:
			return super()._eval_task_indices()
		return resolve_task_ids(self.cfg, "flowmpc_eval_task_ids")

	def _should_eval(self, i):
		return i > 0 and super()._should_eval(i)

	def _provenance(self):
		if not hasattr(self, "_checkpoint_provenance"):
			self._checkpoint_provenance = training_provenance(self.cfg, self.agent)
		return self._checkpoint_provenance

	def _initial_update_index(self):
		self._completed_updates = 0
		resume = getattr(self.cfg, "flowmpc_resume_checkpoint", None)
		if resume is not None:
			self._completed_updates, rng = load_training_checkpoint(resume, self.cfg, self.agent, self._provenance())
			print(f"Resuming {resume}: {self._completed_updates:,} completed / {self.cfg.steps:,} total updates")
			print("Resume restores training and sampler RNG state; compile/CUDA/environment bitwise determinism is not guaranteed")
			restore_rng(rng, self.buffer)
		else:
			print(f"Starting at 0 completed / {self.cfg.steps:,} total updates")
		return self._completed_updates

	def _after_update(self, completed_updates):
		self._completed_updates = completed_updates
		frequency = getattr(self.cfg, "flowmpc_save_freq", 10000)
		if frequency > 0 and completed_updates % frequency == 0:
			save_training_checkpoint(self.cfg, self.agent, self.buffer, completed_updates, self._provenance())

	def _save_eval_checkpoint(self, i):
		pass  # latest.pt is independent of evaluation scores and W&B artifacts.

	def _finalize(self):
		try:
			save_training_checkpoint(self.cfg, self.agent, self.buffer, self._completed_updates, self._provenance())
		finally:
			self.logger.finish()  # Do not ask the standard logger to write final.pt.
