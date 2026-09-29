"""FlowMPC training behavior built from minimal TD-MPC2 extension hooks."""

import torch
from pathlib import Path

from common.layers import api_model_conversion
from tdmpc2 import TDMPC2

from .world_model import FlowMPCWorldModel
from .fm.policy import MultiTaskFlowMatchingPolicy


class FlowMPC(TDMPC2):
	"""TD-MPC2 with pi_BC training and TD/planning bootstrap actions from pi_BC."""

	def __init__(self, cfg):
		super().__init__(cfg)
		self._validate_planner_config()
		if not getattr(self.cfg, "fm_checkpoint", None):
			raise ValueError("fm_checkpoint must point to a trained Flow Matching checkpoint")
		checkpoint = Path(self.cfg.fm_checkpoint)
		if not checkpoint.is_file():
			raise FileNotFoundError(f"FM checkpoint does not exist: {checkpoint}")
		self.fm_policy = MultiTaskFlowMatchingPolicy(self.cfg).to(self.device)
		self.fm_policy.load(checkpoint, map_location=self.device, inference_only=True)
		self.fm_policy.eval()
		self.fm_policy.requires_grad_(False)
		if self.fm_policy._action_masks.shape != self.model._action_masks.shape:
			raise ValueError("FM and world-model action masks have incompatible shapes")
		if not torch.equal(self.fm_policy._action_masks, self.model._action_masks):
			raise ValueError("FM and world-model action masks differ; task/action ordering is incompatible")

	def train(self, mode=True):
		super().train(mode)
		if hasattr(self, "fm_policy"):
			self.fm_policy.eval()
		return self

	def _validate_planner_config(self):
		if self.cfg.mpc is not True:
			raise ValueError("FlowMPC requires mpc=true")
		if self.cfg.num_pi_trajs <= 0:
			raise ValueError("num_pi_trajs must be > 0 for FlowMPC")
		if self.cfg.num_pi_trajs > self.cfg.num_samples:
			raise ValueError("num_pi_trajs must not exceed num_samples")
		if not 0 < self.cfg.num_elites <= self.cfg.num_samples:
			raise ValueError("num_elites must be in [1, num_samples]")
		if self.cfg.flow_horizon < self.cfg.horizon:
			raise ValueError("flow_horizon must be at least the TD-MPC2 planning horizon")
		if self.cfg.flow_inference_steps <= 0:
			raise ValueError("flow_inference_steps must be positive")

	def _build_model(self, cfg):
		return FlowMPCWorldModel(cfg)

	def _configure_trainable_params(self):
		"""Set all FlowMPC trainability from the single configured training mode."""
		mode = self.cfg.flowmpc_train_mode
		if mode not in {"frozen", "partial", "full"}:
			raise ValueError("flowmpc_train_mode must be one of: frozen, partial, full")
		for parameter in self.model.parameters():
			parameter.requires_grad_(False)

		train_modules = [self.model._pi_bc]
		if mode in {"partial", "full"}:
			train_modules += [self.model._dynamics, self.model._reward, self.model._Qs]
			if self.model._termination is not None:
				train_modules.append(self.model._termination)
		if mode == "full":
			train_modules += [self.model._encoder]
			if self.cfg.multitask:
				train_modules.append(self.model._task_emb)
		for module in train_modules:
			for parameter in module.parameters():
				parameter.requires_grad_(True)

		if self.cfg.multitask:
			self.model._task_emb.max_norm = 1 if mode == "full" else None
		return train_modules

	def _configure_optimizers(self):
		train_modules = self._configure_trainable_params()
		groups = []
		for module in train_modules:
			params = [parameter for parameter in module.parameters() if parameter.requires_grad]
			if params:
				groups.append({
					"params": params,
					"lr": self.cfg.lr * self.cfg.enc_lr_scale if module is self.model._encoder else self.cfg.lr,
				})
		if not groups:
			raise RuntimeError("FlowMPC has no trainable parameters")
		optimizer_ids = {id(parameter) for group in groups for parameter in group["params"]}
		expected_ids = {id(parameter) for module in train_modules for parameter in module.parameters() if parameter.requires_grad}
		if optimizer_ids != expected_ids or not all(parameter.requires_grad for group in groups for parameter in group["params"]):
			raise RuntimeError("FlowMPC optimizer parameters do not match configured trainable parameters")
		self.optim = torch.optim.Adam(groups, lr=self.cfg.lr, capturable=True)
		self.pi_optim = None

	def _bootstrap_action(self, z, task):
		return self.model.pi_bc(z, task)
	def _proposal_trajectories(self, obs, z, task):
		proposals = self.fm_policy.sample(state=obs, task=task, num_samples=self.cfg.num_pi_trajs)
		expected_shape = (self.cfg.flow_horizon, self.cfg.num_pi_trajs, self.cfg.action_dim)
		if tuple(proposals.shape) != expected_shape:
			raise ValueError(f"FM proposal shape {tuple(proposals.shape)} does not match {expected_shape}")
		return proposals[:self.cfg.horizon]


	def _extra_model_loss(self, zs, action, task):
		prediction = self.model.pi_bc(zs, task)
		mask = self.model._action_masks[task.long()]
		squared_error = (prediction - action).square() * mask
		per_sample = squared_error.sum(dim=-1) / mask.sum(dim=-1).unsqueeze(0)
		rho = torch.pow(self.cfg.rho, torch.arange(zs.shape[0], device=zs.device, dtype=zs.dtype))
		bc_loss = (per_sample.mean(dim=1) * rho).mean()
		return self.cfg.bc_coef * bc_loss, {"bc_loss": bc_loss}

	def _after_model_update(self, zs, action, task):
		return {}
	def _update_target_q(self):
		if self.cfg.flowmpc_train_mode != "frozen":
			super()._update_target_q()

	def load_pretrained_tdmpc(self, fp):
		"""Load inherited TD-MPC2 weights while leaving the new pi_BC initialized."""
		checkpoint = torch.load(fp, map_location=self.device, weights_only=False)
		state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
		state_dict = api_model_conversion(self.model.state_dict(), dict(state_dict))
		result = self.model.load_state_dict(state_dict, strict=False)
		expected_missing = {key for key in self.model.state_dict() if key.startswith("_pi_bc.")}
		if set(result.missing_keys) != expected_missing or result.unexpected_keys:
			raise RuntimeError(
				f"Incompatible TD-MPC2 checkpoint: missing={result.missing_keys}, unexpected={result.unexpected_keys}"
			)
