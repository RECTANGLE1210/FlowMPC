"""State-only temporal Flow Matching layers.

Adapted from Feebami/FlowMPC/flow_match/layers.py. Changes: state-only mt80
conditioning, task embedding, padded actions, and TD-MPC2-style action masks.
"""

import math

import torch
import torch.nn as nn


class SinusoidalPosEmb(nn.Module):
	"""Sinusoidal embedding for scalar flow times."""

	def __init__(self, dim):
		super().__init__()
		if dim < 4 or dim % 2:
			raise ValueError("flow_t_emb_dim must be an even integer of at least 4")
		self.dim = dim

	def forward(self, x):
		half_dim = self.dim // 2
		scale = math.log(10000) / (half_dim - 1)
		freq = torch.exp(torch.arange(half_dim, device=x.device, dtype=x.dtype) * -scale)
		emb = x[:, None] * freq[None, :]
		return torch.cat((emb.sin(), emb.cos()), dim=-1)


class Downsample1d(nn.Module):
	def __init__(self, dim):
		super().__init__()
		self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

	def forward(self, x):
		return self.conv(x)


class Upsample1d(nn.Module):
	def __init__(self, dim):
		super().__init__()
		self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

	def forward(self, x):
		return self.conv(x)


class Conv1dBlock(nn.Module):
	def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
		super().__init__()
		self.block = nn.Sequential(
			nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False),
			nn.GroupNorm(n_groups, out_channels),
			nn.Mish(),
		)

	def forward(self, x):
		return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
	"""Residual temporal block with FiLM conditioning."""

	def __init__(self, in_channels, out_channels, cond_dim, kernel_size=3, n_groups=8):
		super().__init__()
		self.blocks = nn.ModuleList((
			Conv1dBlock(in_channels, out_channels, kernel_size, n_groups),
			Conv1dBlock(out_channels, out_channels, kernel_size, n_groups),
		))
		self.out_channels = out_channels
		self.cond_encoder = nn.Sequential(
			nn.Mish(), nn.Linear(cond_dim, out_channels * 2), nn.Unflatten(-1, (-1, 1))
		)
		self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

	def forward(self, x, cond):
		out = self.blocks[0](x)
		scale, bias = self.cond_encoder(cond).view(cond.shape[0], 2, self.out_channels, 1).unbind(1)
		out = scale * out + bias
		return self.blocks[1](out) + self.residual_conv(x)


class ConditionalUnet1D(nn.Module):
	"""Predict an action-trajectory velocity from time, state, and task ID."""

	def __init__(self, cfg):
		super().__init__()
		self.action_dim = cfg.action_dim
		self.state_dim = cfg.obs_shape["state"][0]
		self._task_emb = nn.Embedding(len(cfg.tasks), cfg.task_dim)
		dims = [cfg.action_dim] + list(cfg.unet_down_dims)
		if len(dims) < 2:
			raise ValueError("unet_down_dims must contain at least one dimension")
		cond_dim = cfg.flow_t_emb_dim + self.state_dim + cfg.task_dim
		self.diffusion_step_encoder = nn.Sequential(
			SinusoidalPosEmb(cfg.flow_t_emb_dim),
			nn.Linear(cfg.flow_t_emb_dim, cfg.flow_t_emb_dim * 4),
			nn.Mish(),
			nn.Linear(cfg.flow_t_emb_dim * 4, cfg.flow_t_emb_dim),
		)

		in_out = list(zip(dims[:-1], dims[1:]))
		mid_dim = dims[-1]
		self.mid_modules = nn.ModuleList((
			ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim, cfg.unet_kernel_size, cfg.unet_n_groups),
			ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim, cfg.unet_kernel_size, cfg.unet_n_groups),
		))
		self.down_modules = nn.ModuleList()
		for index, (dim_in, dim_out) in enumerate(in_out):
			is_last = index == len(in_out) - 1
			self.down_modules.append(nn.ModuleList((
				ConditionalResidualBlock1D(dim_in, dim_out, cond_dim, cfg.unet_kernel_size, cfg.unet_n_groups),
				ConditionalResidualBlock1D(dim_out, dim_out, cond_dim, cfg.unet_kernel_size, cfg.unet_n_groups),
				Downsample1d(dim_out) if not is_last else nn.Identity(),
			)))
		self.up_modules = nn.ModuleList()
		for index, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
			is_last = index >= len(in_out) - 1
			self.up_modules.append(nn.ModuleList((
				ConditionalResidualBlock1D(dim_out * 2, dim_in, cond_dim, cfg.unet_kernel_size, cfg.unet_n_groups),
				ConditionalResidualBlock1D(dim_in, dim_in, cond_dim, cfg.unet_kernel_size, cfg.unet_n_groups),
				Upsample1d(dim_in) if not is_last else nn.Identity(),
			)))
		self.final_conv = nn.Sequential(
			Conv1dBlock(dims[1], dims[1], cfg.unet_kernel_size, cfg.unet_n_groups),
			nn.Conv1d(dims[1], cfg.action_dim, 1),
		)

	def forward(self, sample, timestep, state, task):
		"""Return velocity with the same ``[T, B, action_dim]`` shape as sample."""
		if sample.ndim != 3:
			raise ValueError("sample must have shape [T, B, action_dim]")
		_, batch_size, action_dim = sample.shape
		if action_dim != self.action_dim or state.shape != (batch_size, self.state_dim):
			raise ValueError("sample or state has an incompatible shape")
		task = torch.as_tensor(task, device=sample.device).long().reshape(-1)
		if task.numel() != batch_size:
			raise ValueError("task must have shape [B]")
		timestep = torch.as_tensor(timestep, device=sample.device, dtype=sample.dtype)
		if timestep.ndim == 0:
			timestep = timestep.expand(batch_size)
		else:
			timestep = timestep.reshape(-1)
			if timestep.numel() == 1:
				timestep = timestep.expand(batch_size)
			elif timestep.numel() != batch_size:
				raise ValueError("timestep must be scalar or have shape [B]")
		cond = torch.cat((self.diffusion_step_encoder(timestep), state, self._task_emb(task)), dim=-1)
		x = sample.permute(1, 2, 0)
		skips = []
		for resnet, resnet2, downsample in self.down_modules:
			x = resnet2(resnet(x, cond), cond)
			skips.append(x)
			x = downsample(x)
		for module in self.mid_modules:
			x = module(x, cond)
		for resnet, resnet2, upsample in self.up_modules:
			x = torch.cat((x, skips.pop()), dim=1)
			x = upsample(resnet2(resnet(x, cond), cond))
		return self.final_conv(x).permute(2, 0, 1)
