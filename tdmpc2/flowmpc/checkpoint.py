"""Atomic FlowMPC training checkpoints (not TD-MPC2's model-only save API)."""

import hashlib
import os
from pathlib import Path
import random
from tempfile import NamedTemporaryFile

import numpy as np
import torch

from .fm.data import _chunk_paths, _source_identity

_FORMAT = "flowmpc-training"
_VERSION = 1
_CRITICAL_CFG = (
	"task", "tasks", "flowmpc_train_task_ids", "flowmpc_train_mode", "model_size", "obs", "multitask",
	"obs_shape", "obs_shapes", "action_dim", "action_dims", "episode_lengths", "episodic",
	"horizon", "batch_size", "latent_dim", "enc_dim", "mlp_dim", "num_enc_layers", "num_channels",
	"task_dim", "num_q", "num_bins", "vmin", "vmax", "simnorm_dim", "dropout", "log_std_min", "log_std_max",
	"lr", "enc_lr_scale", "tau", "rho", "bc_coef", "grad_clip_norm", "consistency_coef", "reward_coef",
	"value_coef", "termination_coef", "discount_denom", "discount_min", "discount_max", "entropy_coef",
	"flow_horizon", "flow_inference_steps", "flow_t_emb_dim", "unet_down_dims", "unet_kernel_size", "unet_n_groups",
)


def _file_identity(path):
	path = Path(path).resolve()
	digest = hashlib.sha256()
	with path.open("rb") as file:
		for block in iter(lambda: file.read(1024 * 1024), b""):
			digest.update(block)
	return {"path": str(path), "size": path.stat().st_size, "sha256": digest.hexdigest()}


def _optimizer_signature(agent):
	names = {id(value): name for name, value in agent.model.named_parameters()}
	return [{
		"parameters": [(names[id(value)], tuple(value.shape), str(value.dtype)) for value in group["params"]],
		"options": {key: value for key, value in group.items() if key != "params"},
	} for group in agent.optim.param_groups]


def training_provenance(cfg, agent):
	"""Hash the frozen FM file once per trainer; do not store its network weights."""
	result = {
		"config": {key: getattr(cfg, key, None) for key in _CRITICAL_CFG},
		"optimizer_groups": _optimizer_signature(agent),
		"model_spec": [(key, tuple(value.shape), str(value.dtype)) for key, value in agent.model.state_dict().items() if torch.is_tensor(value)],
		"fm_checkpoint": _file_identity(cfg.fm_checkpoint),
		"source_files": _source_identity(_chunk_paths(cfg.data_dir)),
	}
	if cfg.flowmpc_train_mode == "frozen":
		result["fm_selection"] = _file_identity(Path(cfg.fm_checkpoint).with_name("fm_selection.pt"))
	return result


def capture_rng(buffer):
	"""SliceSampler 0.8.1 uses global torch RNG unless a generator is supplied."""
	sampler = buffer._sampler
	generator = sampler._rng
	return {
		"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
		"cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
		"sampler_type": f"{type(sampler).__module__}.{type(sampler).__qualname__}",
		"sampler_state": sampler.state_dict(),
		"sampler_generator": None if generator is None else {"device": str(generator.device), "state": generator.get_state()},
		"storage_device": str(getattr(buffer, "_storage_device", "unknown")),
		"num_eps": buffer.num_eps, "capacity": buffer.capacity,
	}


def restore_rng(state, buffer):
	"""Call after initialization, immediately before continuing the inherited loop."""
	sampler = buffer._sampler
	name = f"{type(sampler).__module__}.{type(sampler).__qualname__}"
	if state["sampler_type"] != name or state["num_eps"] != buffer.num_eps or state["capacity"] != buffer.capacity:
		raise ValueError("Resume replay sampler/count/capacity is incompatible")
	if state["storage_device"] != str(getattr(buffer, "_storage_device", "unknown")):
		print("WARNING: replay storage device changed; sampled RNG sequence may differ")
	sampler.load_state_dict(state["sampler_state"])
	saved_generator = state["sampler_generator"]
	generator = None
	if saved_generator is not None:
		generator = torch.Generator(device=saved_generator["device"])
		generator.set_state(saved_generator["state"].cpu())
	buffer._buffer.set_rng(generator)
	random.setstate(state["python"])
	np.random.set_state(state["numpy"])
	torch.set_rng_state(state["torch"].cpu())
	if state["cuda"] is not None:
		if not torch.cuda.is_available() or len(state["cuda"]) != torch.cuda.device_count():
			raise ValueError("Resume CUDA RNG device count is incompatible")
		torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def save_training_checkpoint(cfg, agent, buffer, completed_updates, provenance):
	path = Path(cfg.work_dir) / "models" / "latest.pt"
	path.parent.mkdir(parents=True, exist_ok=True)
	payload = {
		"format": _FORMAT, "version": _VERSION, "update": completed_updates, "next_iteration": completed_updates,
		"model": agent.model.state_dict(), "optimizer": agent.optim.state_dict(), "scale": agent.scale.state_dict(),
		"rng": capture_rng(buffer), "provenance": provenance,
		"run": {key: getattr(cfg, key, None) for key in ("steps", "seed", "compile", "eval_freq", "eval_episodes", "flowmpc_eval_task_ids", "flowmpc_save_freq", "tdmpc_checkpoint", "flowmpc_resume_checkpoint")},
	}
	temporary = None
	try:
		with NamedTemporaryFile(mode="wb", dir=path.parent, prefix=".latest-", suffix=".tmp", delete=False) as file:
			temporary = Path(file.name)
			torch.save(payload, file)
			file.flush()
			os.fsync(file.fileno())
		os.replace(temporary, path)
	finally:
		if temporary is not None:
			temporary.unlink(missing_ok=True)
	print(f"Saved {completed_updates:,} completed updates to {path}")
	return path


def load_training_checkpoint(path, cfg, agent, provenance):
	"""Load trusted training files; leave RNG restoration until initialization ends."""
	payload = torch.load(path, map_location="cpu", weights_only=False)
	if not isinstance(payload, dict) or payload.get("format") != _FORMAT or payload.get("version") != _VERSION:
		raise ValueError("Not a supported FlowMPC training checkpoint (model-only files cannot resume training)")
	completed = payload.get("update")
	if type(completed) is not int or completed < 0 or payload.get("next_iteration") != completed or completed > cfg.steps:
		raise ValueError("Invalid completed-update count, or steps is less than the resume point")
	saved = payload.get("provenance", {})
	for section in ("config", "optimizer_groups", "model_spec", "source_files"):
		if saved.get(section) != provenance[section]:
			raise ValueError(f"Resume checkpoint has incompatible {section}")
	for section in ("fm_checkpoint", "fm_selection"):
		if section in provenance:
			identity = saved.get(section, {})
			if any(identity.get(key) != provenance[section][key] for key in ("size", "sha256")):
				raise ValueError(f"Resume checkpoint has incompatible {section} identity")
	for key in ("model", "optimizer", "scale", "rng"):
		if key not in payload:
			raise ValueError(f"Resume checkpoint is missing {key}")
	groups = payload["optimizer"].get("param_groups", [])
	expected = provenance["optimizer_groups"]
	if len(groups) != len(expected) or any(
		len(group["params"]) != len(spec["parameters"]) or {key: value for key, value in group.items() if key != "params"} != spec["options"]
		for group, spec in zip(groups, expected)
	):
		raise ValueError("Resume optimizer param groups are incompatible")
	agent.model.load_state_dict(payload["model"], strict=True)
	agent.optim.load_state_dict(payload["optimizer"])
	agent.scale.load_state_dict(payload["scale"])
	agent.fm_policy.eval().requires_grad_(False)
	return completed, payload["rng"]
