"""Lightweight GPU smoke tests for FlowMPC's TD-MPC2 world-model adaptation."""

from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from types import SimpleNamespace

import torch

from common.world_model import WorldModel
from tdmpc2 import TDMPC2

from .agent import FlowMPC
from .fm.policy import MultiTaskFlowMatchingPolicy
from .world_model import FlowMPCWorldModel


def _cfg(mode="frozen", fm_checkpoint=None):
	return SimpleNamespace(
		multitask=True, tasks=["task-a", "task-b"], task_dim=4,
		obs_shape={"state": (3,)}, action_dim=4, action_dims=[2, 4],
		episodic=False, latent_dim=8, mlp_dim=8, enc_dim=8, num_enc_layers=2,
		num_q=2, num_bins=5, dropout=0., simnorm_dim=2, log_std_min=-10.,
		log_std_max=2., lr=1e-3, enc_lr_scale=.5, horizon=2, batch_size=2,
		iterations=1, episode_lengths=[10, 10], episode_length=10,
		discount_denom=5, discount_min=.95, discount_max=.995, compile=False,
		rho=.5, entropy_coef=1e-4, grad_clip_norm=10., flowmpc_train_mode=mode,
		bc_coef=1., tau=.01, mpc=True, num_samples=2, num_pi_trajs=1, num_elites=1,
		max_std=1., min_std=.05, temperature=.5, flow_horizon=4,
		flow_inference_steps=2, flow_t_emb_dim=4, unet_down_dims=[4, 8],
		unet_kernel_size=3, unet_n_groups=4, flow_lr=1e-3,
		fm_checkpoint=fm_checkpoint,
	)


def _temporary_fm_checkpoint(cfg):
	with NamedTemporaryFile(suffix=".pt", delete=False) as file:
		path = Path(file.name)
	MultiTaskFlowMatchingPolicy(cfg).save(path)
	return path


def _parameter_ids(modules):
	return {id(parameter) for module in modules for parameter in module.parameters() if parameter.requires_grad}


def _expected_modules(agent, mode):
	modules = [agent.model._pi_bc]
	if mode in {"partial", "full"}:
		modules += [agent.model._dynamics, agent.model._reward, agent.model._Qs]
	if mode == "full":
		modules += [agent.model._encoder, agent.model._task_emb]
	return modules


def _tensor_snapshot(state):
	return {key: value.detach().clone() for key, value in state.items() if torch.is_tensor(value)}


def _state_equal(left, right):
	return torch.equal(left, right) if torch.is_tensor(left) else left == right


def _assert_target_q_update(agent, should_update):
	state = agent.model.state_dict()
	targets = {key: value for key, value in state.items() if "_target_Qs_params" in key and torch.is_tensor(value)}
	assert targets, "No target-Q tensors captured"
	# Make targets differ from online Q so a real soft update is observable.
	with torch.no_grad():
		for value in targets.values():
			value.add_(1.)
	before = _tensor_snapshot(targets)
	agent._update_target_q()
	after = agent.model.state_dict()
	for key, value in before.items():
		expected = value.lerp(state[key.replace("_target_Qs_params", "_detach_Qs_params")], agent.cfg.tau) if should_update else value
		assert torch.allclose(after[key], expected) if should_update else torch.equal(after[key], expected)
	if should_update:
		assert any(not torch.equal(after[key], value) for key, value in before.items())


def run_smoke_test():
	if not torch.cuda.is_available():
		raise RuntimeError("FlowMPC smoke test requires CUDA, matching TD-MPC2's cuda:0 agent")
	torch.manual_seed(0)
	cfg = _cfg()
	fm_checkpoint = _temporary_fm_checkpoint(cfg)
	try:
		cfg.fm_checkpoint = fm_checkpoint
		model = FlowMPCWorldModel(cfg).cuda()
		assert isinstance(model, WorldModel)
		z = torch.randn(2, cfg.latent_dim, device="cuda")
		task = torch.tensor([0, 1], device="cuda")
		assert model.pi_bc(z, task).shape == (2, cfg.action_dim)
		assert model.pi_bc(z.unsqueeze(0).repeat(cfg.horizon, 1, 1), task).shape == (cfg.horizon, 2, cfg.action_dim)
		assert torch.equal(model.pi_bc(z, task)[0, 2:], torch.zeros(2, device="cuda"))

		for mode in ("frozen", "partial", "full"):
			agent = FlowMPC(_cfg(mode, fm_checkpoint))
			expected_ids = _parameter_ids(_expected_modules(agent, mode))
			trainable_ids = {id(parameter) for parameter in agent.model.parameters() if parameter.requires_grad}
			optimizer_ids = {id(parameter) for group in agent.optim.param_groups for parameter in group["params"]}
			assert trainable_ids == expected_ids == optimizer_ids
			assert not any(parameter.requires_grad for parameter in agent.model._pi.parameters())
			if mode in {"frozen", "partial"}:
				before = agent.model._task_emb.weight.detach().clone()
				agent.model.pi_bc(z, task)
				assert torch.equal(agent.model._task_emb.weight, before)
			if mode == "frozen":
				assert trainable_ids == _parameter_ids([agent.model._pi_bc])
			if mode == "partial":
				assert not any(parameter.requires_grad for parameter in agent.model._encoder.parameters())
				assert not any(parameter.requires_grad for parameter in agent.model._task_emb.parameters())
			_assert_target_q_update(agent, should_update=mode != "frozen")

		agent = FlowMPC(_cfg("frozen", fm_checkpoint))
		zs = torch.randn(cfg.horizon, 2, cfg.latent_dim, device="cuda")
		assert hasattr(agent, "fm_policy") and not agent.fm_policy.training
		assert not any(parameter.requires_grad for parameter in agent.fm_policy.parameters())
		assert not hasattr(agent.fm_policy, "_pi") and not hasattr(agent.fm_policy, "optim")
		optimizer_ids = {id(parameter) for group in agent.optim.param_groups for parameter in group["params"]}
		assert not optimizer_ids.intersection({id(parameter) for parameter in agent.fm_policy.parameters()})
		assert torch.equal(agent.fm_policy._action_masks, agent.model._action_masks)
		obs = torch.randn(1, cfg.obs_shape["state"][0], device="cuda")
		proposals = agent._proposal_trajectories(obs, z[:1], task[:1])
		assert proposals.shape == (cfg.horizon, cfg.num_pi_trajs, cfg.action_dim)
		assert torch.equal(proposals[..., 2:], torch.zeros_like(proposals[..., 2:]))
		frozen_fm_state = _tensor_snapshot(agent.fm_policy.state_dict())
		assert frozen_fm_state, "No FM tensors captured"
		agent._proposal_trajectories(obs, z[:1], task[:1])
		agent._proposal_trajectories(obs, z[:1], task[:1])
		assert all(torch.equal(value, agent.fm_policy.state_dict()[key]) for key, value in frozen_fm_state.items())
		agent.train()
		assert not agent.fm_policy.training
		agent.eval()
		assert not agent.fm_policy.training

		task_zero = torch.zeros(2, dtype=torch.long, device="cuda")
		action = torch.randn(cfg.horizon, 2, cfg.action_dim, device="cuda")
		_, metrics = agent._extra_model_loss(zs, action, task_zero)
		modified_action = action.clone()
		modified_action[..., 2:] = 1e9
		_, modified_metrics = agent._extra_model_loss(zs, modified_action, task_zero)
		assert torch.isfinite(metrics["bc_loss"])
		assert torch.equal(metrics["bc_loss"], modified_metrics["bc_loss"])
		assert torch.equal(agent._bootstrap_action(z, task), agent.model.pi_bc(z, task))

		standard = TDMPC2(_cfg())
		assert type(standard.model) is WorldModel and standard.pi_optim is not None
		zero_loss, zero_metrics = standard._extra_model_loss(zs, action, task_zero)
		standard_proposals = standard._proposal_trajectories(obs, z[:1], task[:1])
		assert standard_proposals.shape == (cfg.horizon, cfg.num_pi_trajs, cfg.action_dim)
		legacy_action = torch.full((2, cfg.action_dim), .25, device="cuda")
		standard.model.pi = lambda states, tasks: (legacy_action, {})
		assert torch.equal(standard._bootstrap_action(z, task), legacy_action)
		assert zero_loss == 0 and zero_metrics == {}
		standard.update_pi = lambda states, tasks: {"hook": torch.tensor(1., device="cuda")}
		assert "hook" in standard._after_model_update(zs, action, task_zero)
		_assert_target_q_update(standard, should_update=True)

		with TemporaryDirectory() as directory:
			source = WorldModel(_cfg()).cuda()
			tdmpc_checkpoint = Path(directory) / "tdmpc.pt"
			torch.save({"model": source.state_dict()}, tdmpc_checkpoint)
			pi_bc_before = _tensor_snapshot(agent.model._pi_bc.state_dict())
			assert pi_bc_before, "No pi_BC tensors captured"
			agent.load_pretrained_tdmpc(tdmpc_checkpoint)
			for key, value in source.state_dict().items():
				assert _state_equal(value, agent.model.state_dict()[key])
			assert all(torch.equal(value, agent.model._pi_bc.state_dict()[key]) for key, value in pi_bc_before.items())
			flowmpc_checkpoint = Path(directory) / "flowmpc.pt"
			agent.save(flowmpc_checkpoint)
			restored = FlowMPC(_cfg("frozen", fm_checkpoint))
			restored.load(flowmpc_checkpoint)
			assert agent.model.state_dict().keys() == restored.model.state_dict().keys()
			assert all(_state_equal(value, restored.model.state_dict()[key]) for key, value in agent.model.state_dict().items())

		bad_mpc = _cfg("frozen", fm_checkpoint)
		bad_mpc.mpc = False
		try:
			FlowMPC(bad_mpc)
		except ValueError:
			pass
		else:
			raise AssertionError("mpc=false should fail FlowMPC validation")
		bad_elites = _cfg("frozen", fm_checkpoint)
		bad_elites.num_elites = bad_elites.num_samples + 1
		try:
			FlowMPC(bad_elites)
		except ValueError:
			pass
		else:
			raise AssertionError("num_elites > num_samples should fail validation")

		bad_horizon = _cfg("frozen", fm_checkpoint)
		bad_horizon.flow_horizon = bad_horizon.horizon - 1
		try:
			FlowMPC(bad_horizon)
		except ValueError:
			pass
		else:
			raise AssertionError("flow_horizon < horizon should fail validation")
		bad_count = _cfg("frozen", fm_checkpoint)
		bad_count.num_pi_trajs = bad_count.num_samples + 1
		try:
			FlowMPC(bad_count)
		except ValueError:
			pass
		else:
			raise AssertionError("num_pi_trajs > num_samples should fail validation")
	finally:
		fm_checkpoint.unlink(missing_ok=True)

	missing_checkpoint = _cfg("frozen", fm_checkpoint.with_name("missing_fm.pt"))
	try:
		FlowMPC(missing_checkpoint)
	except FileNotFoundError:
		pass
	else:
		raise AssertionError("missing FM checkpoint should fail validation")
	return {"bc_loss": metrics["bc_loss"].item()}


if __name__ == "__main__":
	print(run_smoke_test())