"""FlowMPC-specific delta on top of TD-MPC2's world model."""

import torch
import torch.nn as nn

from common import init, layers
from common.world_model import WorldModel


class FlowMPCWorldModel(WorldModel):
	"""TD-MPC2 world model with an additional deterministic BC policy."""

	def __init__(self, cfg):
		super().__init__(cfg)
		self._pi_bc = nn.Sequential(
			layers.mlp(cfg.latent_dim + cfg.task_dim, 2 * [cfg.mlp_dim], cfg.action_dim),
			nn.Tanh(),
		)
		self._pi_bc.apply(init.weight_init)

	def pi_bc(self, z, task):
		"""Return masked deterministic behavior-cloning actions for latent states."""
		if self.cfg.multitask:
			z = self.task_emb(z, task)
		action = self._pi_bc(z)
		if self.cfg.multitask:
			action = action * self._action_masks[task.long()]
		return action
