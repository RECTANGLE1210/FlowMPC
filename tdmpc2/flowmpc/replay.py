"""Config-driven mt80 replay routing; task IDs always retain their original rows."""

from copy import deepcopy
from pathlib import Path

import torch

from common import TASK_SET
from common.buffer import Buffer
from .fm.data import _chunk_paths, load_selected_buffer, load_selection_metadata


def resolve_task_ids(cfg, name):
	"""Resolve null to all tasks and reject ambiguous or invalid subset IDs."""
	ids = getattr(cfg, name, None)
	if ids is None:
		return list(range(len(cfg.tasks)))
	if not isinstance(ids, (list, tuple)) or not ids:
		raise ValueError(f"{name} must be null or a nonempty list of original mt80 task IDs")
	if any(type(task_id) is not int or not 0 <= task_id < len(cfg.tasks) for task_id in ids):
		raise ValueError(f"{name} contains non-integer or out-of-range task IDs")
	if len(set(ids)) != len(ids):
		raise ValueError(f"{name} contains duplicate task IDs")
	return list(ids)


def validate_flowmpc_cfg(cfg):
	if cfg.flowmpc_train_mode not in ("frozen", "partial", "full"):
		raise ValueError("flowmpc_train_mode must be one of: frozen, partial, full")
	if cfg.task != "mt80" or list(cfg.tasks) != TASK_SET["mt80"]:
		raise ValueError("FlowMPC requires cfg.task=mt80 and the complete official task ordering")
	resolve_task_ids(cfg, "flowmpc_train_task_ids")
	resolve_task_ids(cfg, "flowmpc_eval_task_ids")


def _episode_indices(td, task_ids, selected=None):
	tasks = td["task"][:, 0].reshape(td.shape[0], -1)[:, 0]
	if selected is not None:
		tasks = tasks[selected]
	mask = torch.isin(tasks, torch.tensor(task_ids, device=tasks.device, dtype=tasks.dtype))
	return selected[mask] if selected is not None else torch.where(mask)[0]


def _build_subset(cfg, task_ids, metadata=None):
	"""Count before allocation; retain indices only, never source replay chunks."""
	paths = _chunk_paths(cfg.data_dir)
	episode_length = metadata["episode_length"] if metadata is not None else 101
	if cfg.horizon >= episode_length:
		raise ValueError("Replay episodes must be longer than the sampling horizon")
	files = {}
	counts = torch.zeros(len(cfg.tasks), dtype=torch.long)
	bytes_per_episode = None
	for path in paths:
		selected = metadata["files"].get(path.name) if metadata is not None else None
		if metadata is not None and selected is None:
			continue
		td = torch.load(path, map_location="cpu", weights_only=False)
		if td.ndim != 2 or td.shape[1] != episode_length:
			raise ValueError(f"{path} must contain episodes of length {episode_length}")
		indices = _episode_indices(td, task_ids, selected)
		if indices.numel():
			files[path.name] = indices
			tasks = td["task"][indices, 0].reshape(indices.numel(), -1)[:, 0].long()
			counts += torch.bincount(tasks, minlength=len(cfg.tasks))
			if bytes_per_episode is None:
				bytes_per_episode = sum(value.numel() * value.element_size() for value in td[0].values(include_nested=True, leaves_only=True))
				if "episode" not in td.keys():
					bytes_per_episode += episode_length * 8  # Buffer adds int64 episode IDs.
			del tasks
		del td, indices
	if (counts[task_ids] == 0).any():
		raise ValueError("Replay has no episodes for one or more requested training tasks")
	if metadata is not None and not (counts[task_ids] == metadata["episodes_per_task"]).all():
		raise ValueError("Selected replay is not balanced according to FM selection metadata")
	episode_count = int(counts.sum().item())
	buffer_cfg = deepcopy(cfg)
	buffer_cfg.episode_length = episode_length
	buffer_cfg.buffer_size = episode_count * episode_length
	buffer_cfg.steps = buffer_cfg.buffer_size
	print(f"Selected episodes: {episode_count:,}; buffer capacity: {buffer_cfg.buffer_size:,}")
	print(f"Estimated replay size: {episode_count * bytes_per_episode / 1e9:.2f} GB")
	buffer = Buffer(buffer_cfg)
	for name, indices in files.items():
		td = torch.load(Path(cfg.data_dir) / name, map_location="cpu", weights_only=False)
		buffer.load(td[indices])
		del td
	if buffer.num_eps != episode_count:
		raise RuntimeError(f"Loaded {buffer.num_eps} episodes, expected {episode_count}")
	return buffer


def build_flowmpc_replay(cfg):
	"""Return preloaded replay, or None to request the unchanged official loader."""
	validate_flowmpc_cfg(cfg)
	task_ids = resolve_task_ids(cfg, "flowmpc_train_task_ids")
	print(f"FlowMPC replay mode: {cfg.flowmpc_train_mode}")
	print(f"Training tasks (original IDs): {[(task_id, cfg.tasks[task_id]) for task_id in task_ids]}")
	if getattr(cfg, "flowmpc_eval_task_ids", None) is not None:
		print(f"Evaluation task IDs: {resolve_task_ids(cfg, 'flowmpc_eval_task_ids')}")
	metadata = None
	if cfg.flowmpc_train_mode == "frozen":
		metadata = load_selection_metadata(cfg, Path(cfg.fm_checkpoint).with_name("fm_selection.pt"))
		if getattr(cfg, "flowmpc_train_task_ids", None) is None:
			count = len(cfg.tasks) * metadata["episodes_per_task"]
			print(f"Selected episodes: {count:,}; buffer capacity: {count * metadata['episode_length']:,}")
			print("Replay size/device will be reported by TD-MPC2 Buffer before allocation")
			return load_selected_buffer(cfg, metadata, horizon=cfg.horizon, batch_size=cfg.batch_size)
	elif getattr(cfg, "flowmpc_train_task_ids", None) is None:
		print("Official full mt80 replay: episode count/capacity/size will be reported by the TD-MPC2 loader")
		return None
	return _build_subset(cfg, task_ids, metadata)
