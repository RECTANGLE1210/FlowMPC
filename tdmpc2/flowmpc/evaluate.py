"""Hydra entry point for mt80 FlowMPC evaluation."""

import os
os.environ["MUJOCO_GL"] = os.getenv("MUJOCO_GL", "egl")
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import hydra
import torch
from hydra.utils import to_absolute_path

from common.parser import parse_cfg
from common.seed import set_seed
from envs import make_env
from trainer.offline_trainer import OfflineTrainer

from .agent import FlowMPC


torch.backends.cudnn.benchmark = True


def _resolve_path(cfg, name):
	value = getattr(cfg, name, None)
	if not value:
		raise ValueError(f"{name} must be provided")
	path = Path(to_absolute_path(str(value)))
	setattr(cfg, name, str(path))
	return path


def _require_file(cfg, name):
	path = _resolve_path(cfg, name)
	if not path.is_file():
		raise FileNotFoundError(f"{name} does not exist: {path}")


def _validate_evaluation_cfg(cfg):
	if not torch.cuda.is_available():
		raise RuntimeError("FlowMPC evaluation requires CUDA")
	if cfg.task != "mt80" or cfg.obs != "state":
		raise ValueError("FlowMPC evaluation requires task=mt80 and obs=state")
	if cfg.eval_episodes <= 0:
		raise ValueError("eval_episodes must be positive")
	_require_file(cfg, "checkpoint")
	_require_file(cfg, "fm_checkpoint")


def _print_mt80_summary(cfg, results):
	scores = []
	for task in cfg.tasks:
		reward = results[f"episode_reward+{task}"]
		success = results[f"episode_success+{task}"]
		scores.append(success * 100 if task.startswith("mw-") else reward / 10)
		print(f"  {task:<22}\tR: {reward:.01f}  S: {success:.02f}")
	print(f"Normalized score: {sum(scores) / len(scores):.02f}")


@hydra.main(version_base=None, config_path="..", config_name="flowmpc/config")
def evaluate(cfg):
	cfg = parse_cfg(cfg)
	_validate_evaluation_cfg(cfg)
	cfg.flowmpc_train_mode = "frozen"
	set_seed(cfg.seed)
	env = make_env(cfg)
	agent = FlowMPC(cfg)
	agent.load(cfg.checkpoint)
	agent.eval()
	agent.requires_grad_(False)
	results = OfflineTrainer(
		cfg=cfg,
		env=env,
		agent=agent,
		buffer=None,
		logger=None,
	).eval()
	_print_mt80_summary(cfg, results)
	return results


if __name__ == "__main__":
	evaluate()