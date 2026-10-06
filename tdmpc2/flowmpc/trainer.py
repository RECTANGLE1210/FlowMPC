"""Offline TD-MPC2 training with FlowMPC's preselected frozen-mode replay."""

from trainer.offline_trainer import OfflineTrainer


class FlowMPCOfflineTrainer(OfflineTrainer):
	def _load_dataset(self):
		if self.cfg.flowmpc_train_mode == "frozen":
			assert self.buffer is not None and self.buffer.num_eps > 0, "Frozen FlowMPC requires a preloaded selected buffer"
		else:
			super()._load_dataset()

	def _should_eval(self, i):
		return i > 0 and super()._should_eval(i)
