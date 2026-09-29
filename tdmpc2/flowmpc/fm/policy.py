"""Standalone state-based multi-task Flow Matching policy.

Adapted from Feebami/FlowMPC/flow_match/flow_match.py. Changes: mt80 task
conditioning, padded-action masks, tensor-based updates, and masked loss.
"""

from pathlib import Path

import torch
import torch.nn as nn
from torch.optim.swa_utils import AveragedModel, get_ema_avg_fn

from .layers import ConditionalUnet1D


class MultiTaskFlowMatchingPolicy(nn.Module):
	"""Generate padded action trajectories conditioned on state and task ID."""

	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		self.inference_steps = cfg.flow_inference_steps
		self._pi = ConditionalUnet1D(cfg)
		self.ema_model = AveragedModel(self._pi, avg_fn=get_ema_avg_fn()).requires_grad_(False)
		self.optim = torch.optim.Adam(self._pi.parameters(), lr=cfg.flow_lr)
		self.register_buffer("_action_masks", torch.zeros(len(cfg.tasks), cfg.action_dim))
		for index, action_dim in enumerate(cfg.action_dims):
			self._action_masks[index, :action_dim] = 1.

	def _mask(self, task):
		return self._action_masks[task.long()]

	@torch.no_grad()
	def _renormalize_task_embedding(self):
		self._pi._task_emb.weight.renorm_(p=2, dim=0, maxnorm=1.)

	def _interpolate(self, action, task, timestep):
		"""Build the masked linear path and its constant target velocity."""
		mask = self._mask(task).unsqueeze(0)
		x1 = action * mask
		x0 = torch.randn_like(x1) * mask
		t = timestep.view(1, -1, 1)
		return (1 - t) * x0 + t * x1, x1 - x0, mask

	def loss(self, state, action, task):
		"""Return masked, per-active-dimension normalized flow-matching MSE."""
		if state.ndim != 2 or action.ndim != 3:
			raise ValueError("state and action must have shapes [B, state_dim] and [T, B, action_dim]")
		horizon, batch_size, action_dim = action.shape
		if horizon != self.cfg.flow_horizon or state.shape[0] != batch_size or action_dim != self.cfg.action_dim:
			raise ValueError("state or action has an incompatible shape")
		task = torch.as_tensor(task, device=action.device).long().reshape(-1)
		if task.numel() != batch_size:
			raise ValueError("task must have shape [B]")
		timestep = torch.rand(batch_size, device=action.device, dtype=action.dtype)
		x_t, velocity_target, mask = self._interpolate(action, task, timestep)
		velocity_pred = self._pi(x_t, timestep, state, task)
		squared_error = (velocity_pred - velocity_target).square() * mask
		return squared_error.sum(dim=(0, 2)).div(mask.sum(dim=(0, 2)) * horizon).mean()

	def update(self, state, action, task):
		"""Perform one FM optimisation step and update the EMA inference model."""
		loss = self.loss(state, action, task)
		self.optim.zero_grad(set_to_none=True)
		loss.backward()
		grad_norm = torch.nn.utils.clip_grad_norm_(self._pi.parameters(), 1.0)
		self.optim.step()
		self._renormalize_task_embedding()
		self.ema_model.update_parameters(self._pi)
		return {"flow_loss": loss.detach().item(), "flow_grad_norm": grad_norm.detach().item()}

	@torch.no_grad()
	def sample(self, state, task, num_samples=1):
		"""Generate ``[flow_horizon, B * num_samples, action_dim]`` trajectories."""
		if state.ndim != 2 or num_samples < 1:
			raise ValueError("state must have shape [B, state_dim] and num_samples must be positive")
		batch_size = state.shape[0]
		task = torch.as_tensor(task, device=state.device).long().reshape(-1)
		if task.numel() != batch_size:
			raise ValueError("task must have shape [B]")
		state = state.repeat_interleave(num_samples, dim=0)
		task = task.repeat_interleave(num_samples, dim=0)
		mask = self._mask(task).unsqueeze(0)
		x = torch.randn(self.cfg.flow_horizon, task.numel(), self.cfg.action_dim, device=state.device, dtype=state.dtype) * mask
		for step in range(self.inference_steps):
			timestep = torch.full((task.numel(),), step / self.inference_steps, device=state.device, dtype=state.dtype)
			x = (x + self.ema_model(x, timestep, state, task) / self.inference_steps) * mask
		return x.clamp(-1, 1) * mask

	def save(self, path):
		"""Save the EMA inference model, including its learned task embedding."""
		torch.save({
			"format_version": 1,
			"metadata": self._checkpoint_metadata(),
			"ema_model": self.ema_model.state_dict(),
		}, Path(path))

	def _checkpoint_metadata(self):
		return {
			"tasks": list(self.cfg.tasks),
			"task_dim": int(self.cfg.task_dim),
			"state_dim": int(self.cfg.obs_shape["state"][0]),
			"action_dim": int(self.cfg.action_dim),
			"action_dims": list(self.cfg.action_dims),
			"flow_horizon": int(self.cfg.flow_horizon),
			"flow_t_emb_dim": int(self.cfg.flow_t_emb_dim),
			"unet_down_dims": list(self.cfg.unet_down_dims),
			"unet_kernel_size": int(self.cfg.unet_kernel_size),
			"unet_n_groups": int(self.cfg.unet_n_groups),
		}

	def _validate_checkpoint_metadata(self, metadata):
		for key, expected in self._checkpoint_metadata().items():
			if metadata.get(key) != expected:
				raise ValueError(
					f"FM checkpoint metadata mismatch for {key}: "
					f"expected {expected!r}, got {metadata.get(key)!r}"
				)

	def load(self, path, map_location=None, inference_only=False):
		"""Load an EMA checkpoint, optionally releasing all trainable state."""
		checkpoint = torch.load(Path(path), map_location=map_location, weights_only=True)
		if not isinstance(checkpoint, dict) or checkpoint.get("format_version") != 1:
			raise ValueError("Unsupported FM checkpoint format; expected format_version=1")
		if "metadata" not in checkpoint or "ema_model" not in checkpoint:
			raise ValueError("FM checkpoint is missing metadata or EMA weights")
		self._validate_checkpoint_metadata(checkpoint["metadata"])
		self.ema_model.load_state_dict(checkpoint["ema_model"])
		if inference_only:
			del self._pi
			del self.optim
			self.eval()
			self.requires_grad_(False)
