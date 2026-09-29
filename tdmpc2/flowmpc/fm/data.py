"""Episode selection for mt80 FlowMPC training.

Per-task return filtering and balancing are new mt80 adaptations; they are
not part of the original FlowMPC implementation.
"""

from copy import deepcopy
from glob import glob
from pathlib import Path

import torch

from common.buffer import Buffer

_SELECTION_FORMAT_VERSION = 1


def _chunk_paths(data_dir):
	paths = sorted(Path(path) for path in glob(str(Path(data_dir) / "*.pt")))
	if not paths:
		raise FileNotFoundError(f"No .pt dataset chunks found in {data_dir}")
	return paths


def _episode_metadata(td, path):
	if "reward" not in td.keys() or "task" not in td.keys():
		raise KeyError(f"{path} must contain reward and task tensors")
	if td.ndim < 2:
		raise ValueError(f"{path} must have [episodes, timesteps, ...] batch dimensions")
	episodes, episode_length = td.shape[:2]
	reward = td.get("reward")
	task = td.get("task")
	if reward.shape[:2] != (episodes, episode_length) or task.shape[:2] != (episodes, episode_length):
		raise ValueError(f"{path} reward/task shapes do not match its episode dimensions")
	returns = reward[:, 1:].reshape(episodes, episode_length - 1, -1).sum(dim=(1, 2))
	if not torch.isfinite(returns).all():
		raise ValueError(f"{path} contains non-finite episode returns")
	returns = returns.cpu()
	task_ids = task[:, 0].reshape(episodes, -1)[:, 0].long().cpu()
	return returns, task_ids, episode_length


def _source_identity(paths):
	return [{"name": path.name, "size": path.stat().st_size} for path in paths]


def _compatible(metadata, cfg, paths):
	return (
		metadata.get("format_version") == _SELECTION_FORMAT_VERSION
		and metadata.get("tasks") == list(cfg.tasks)
		and metadata.get("quality_fraction") == cfg.flow_quality_fraction
		and metadata.get("max_episodes_per_task") == cfg.flow_max_episodes_per_task
		and metadata.get("num_tasks") == len(cfg.tasks)
		and metadata.get("source_files") == [path.name for path in paths]
		and metadata.get("source_identity") == _source_identity(paths)
	)

def select_episodes(cfg, metadata_path):
	"""Return cached or newly computed balanced high-return episode metadata."""
	paths = _chunk_paths(cfg.data_dir)
	if not 0 < cfg.flow_quality_fraction <= 1:
		raise ValueError("flow_quality_fraction must be in (0, 1]")
	metadata_path = Path(metadata_path)
	if metadata_path.is_file():
		metadata = torch.load(metadata_path, map_location="cpu", weights_only=True)
		if _compatible(metadata, cfg, paths):
			print(f"Reusing FM episode selection from {metadata_path}")
			return metadata

	returns_by_task = [[] for _ in cfg.tasks]
	chunks_by_task = [[] for _ in cfg.tasks]
	episodes_by_task = [[] for _ in cfg.tasks]
	episode_length = None
	for chunk_index, path in enumerate(paths):
		td = torch.load(path, map_location="cpu", weights_only=False)
		returns, task_ids, chunk_length = _episode_metadata(td, path)
		if episode_length is None:
			episode_length = chunk_length
		elif episode_length != chunk_length:
			raise ValueError("All selected dataset chunks must have the same episode length")
		if (task_ids < 0).any() or (task_ids >= len(cfg.tasks)).any():
			raise ValueError(f"{path} contains task IDs outside the configured task range")
		for task_id in task_ids.unique(sorted=True).tolist():
			indices = torch.where(task_ids == task_id)[0]
			returns_by_task[task_id].append(returns[indices])
			chunks_by_task[task_id].append(torch.full_like(indices, chunk_index))
			episodes_by_task[task_id].append(indices)

	task_counts = torch.tensor([sum(part.numel() for part in parts) for parts in returns_by_task])
	if (task_counts == 0).any():
		missing = [cfg.tasks[index] for index in torch.where(task_counts == 0)[0].tolist()]
		raise ValueError(f"Dataset has no episodes for tasks: {missing}")
	candidate_counts = torch.clamp((task_counts.float() * cfg.flow_quality_fraction).floor().long(), min=1)
	episodes_per_task = int(candidate_counts.min().item())
	if cfg.flow_max_episodes_per_task is not None:
		episodes_per_task = min(episodes_per_task, int(cfg.flow_max_episodes_per_task))
	if episodes_per_task < 1:
		raise ValueError("flow_max_episodes_per_task must be positive when provided")

	selected_by_chunk = {path.name: [] for path in paths}
	thresholds = torch.empty(len(cfg.tasks))
	for task_id in range(len(cfg.tasks)):
		returns = torch.cat(returns_by_task[task_id])
		chunk_indices = torch.cat(chunks_by_task[task_id])
		episode_indices = torch.cat(episodes_by_task[task_id])
		order = torch.argsort(returns, descending=True, stable=True)[:episodes_per_task]
		thresholds[task_id] = returns[order[-1]]
		for chunk_index in chunk_indices[order].unique(sorted=True).tolist():
			selected_by_chunk[paths[chunk_index].name].append(
				episode_indices[order][chunk_indices[order] == chunk_index]
			)

	files = {
		name: torch.cat(indices).sort().values
		for name, indices in selected_by_chunk.items() if indices
	}
	metadata = {
		"format_version": _SELECTION_FORMAT_VERSION,
		"tasks": list(cfg.tasks),
		"quality_fraction": float(cfg.flow_quality_fraction),
		"max_episodes_per_task": cfg.flow_max_episodes_per_task,
		"episodes_per_task": episodes_per_task,
		"episode_length": episode_length,
		"num_tasks": len(cfg.tasks),
		"source_files": [path.name for path in paths],
		"source_identity": _source_identity(paths),
		"files": files,
		"task_counts": task_counts,
		"return_thresholds": thresholds,
	}
	metadata_path.parent.mkdir(parents=True, exist_ok=True)
	torch.save(metadata, metadata_path)
	return metadata


def load_selected_buffer(cfg, metadata):
	"""Load only the selected chunk-relative episode indices into TD-MPC2 Buffer."""
	selected_episode_count = len(cfg.tasks) * metadata["episodes_per_task"]
	buffer_cfg = deepcopy(cfg)
	buffer_cfg.horizon = cfg.flow_horizon
	buffer_cfg.batch_size = cfg.flow_batch_size
	buffer_cfg.buffer_size = selected_episode_count * metadata["episode_length"]
	buffer_cfg.steps = buffer_cfg.buffer_size
	buffer = Buffer(buffer_cfg)
	data_dir = Path(cfg.data_dir)
	for name, indices in metadata["files"].items():
		td = torch.load(data_dir / name, map_location="cpu", weights_only=False)
		buffer.load(td[indices.long()])
	if buffer.num_eps != selected_episode_count:
		raise RuntimeError(f"Loaded {buffer.num_eps} selected episodes, expected {selected_episode_count}")
	return buffer
