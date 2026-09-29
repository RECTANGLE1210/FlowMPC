"""Minimal standalone smoke test for the state-based multi-task FM policy."""

from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch

from .policy import MultiTaskFlowMatchingPolicy


def _assert_metadata_mismatch(cfg, checkpoint):
	try:
		MultiTaskFlowMatchingPolicy(cfg).load(checkpoint)
	except ValueError:
		return
	raise AssertionError("incompatible FM checkpoint metadata should fail")


def run_smoke_test():
	torch.manual_seed(0)
	cfg = SimpleNamespace(
		tasks=["task-a", "task-b"], task_dim=8, obs_shape={"state": (5,)},
		action_dim=4, action_dims=[2, 4], flow_horizon=8, flow_inference_steps=2,
		flow_t_emb_dim=8, unet_down_dims=[8, 16], unet_kernel_size=3,
		unet_n_groups=4, flow_lr=1e-3,
	)
	policy = MultiTaskFlowMatchingPolicy(cfg)
	state = torch.randn(3, 5)
	action = torch.randn(8, 3, 4)
	task = torch.tensor([0, 1, 0])
	loss = policy.loss(state, action, task)
	assert torch.isfinite(loss)
	metrics = policy.update(state, action, task)
	assert torch.isfinite(torch.tensor(metrics["flow_loss"]))
	samples = policy.sample(state, task, num_samples=2)
	assert samples.shape == (8, 6, 4)
	assert torch.equal(samples[:, [0, 1, 4, 5], 2:], torch.zeros_like(samples[:, [0, 1, 4, 5], 2:]))
	with TemporaryDirectory() as directory:
		checkpoint = f"{directory}/fm.pt"
		policy.save(checkpoint)
		payload = torch.load(checkpoint, weights_only=True)
		assert payload["format_version"] == 1
		assert payload["metadata"] == policy._checkpoint_metadata()
		inference_policy = MultiTaskFlowMatchingPolicy(cfg)
		inference_policy.load(checkpoint, inference_only=True)
		assert not hasattr(inference_policy, "_pi") and not hasattr(inference_policy, "optim")
		assert not inference_policy.training and not any(parameter.requires_grad for parameter in inference_policy.parameters())
		assert inference_policy.sample(state, task).shape == (8, 3, 4)
		frozen_state = {key: value.clone() for key, value in inference_policy.state_dict().items()}
		inference_policy.sample(state, task)
		assert all(torch.equal(value, frozen_state[key]) for key, value in inference_policy.state_dict().items())
		_assert_metadata_mismatch(SimpleNamespace(**{**vars(cfg), "tasks": ["task-b", "task-a"]}), checkpoint)
		_assert_metadata_mismatch(SimpleNamespace(**{**vars(cfg), "action_dims": [4, 2]}), checkpoint)
		_assert_metadata_mismatch(SimpleNamespace(**{**vars(cfg), "obs_shape": {"state": (6,)}}), checkpoint)
		_assert_metadata_mismatch(SimpleNamespace(**{**vars(cfg), "flow_horizon": 9}), checkpoint)
	return metrics


if __name__ == "__main__":
	print(run_smoke_test())