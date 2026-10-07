"""Offline TD-MPC2 training with FlowMPC's preselected frozen-mode replay."""

from trainer.offline_trainer import OfflineTrainer
from .replay import resolve_task_ids, validate_flowmpc_cfg


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
