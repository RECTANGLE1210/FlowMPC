"""Hydra entry point for offline mt80 FlowMPC training."""

import os
os.environ["MUJOCO_GL"] = os.getenv("MUJOCO_GL", "egl")
os.environ["LAZY_LEGACY_OP"] = "0"
os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"
os.environ["TORCH_LOGS"] = "+recompiles"
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import hydra
import torch
from hydra.utils import to_absolute_path

from common.logger import Logger
from common.parser import parse_cfg
from common.seed import set_seed
from envs import make_env

from .agent import FlowMPC
from .replay import build_flowmpc_replay, validate_flowmpc_cfg
from .trainer import FlowMPCOfflineTrainer


torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")


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


def _validate_training_cfg(cfg):
	validate_flowmpc_cfg(cfg)
	if cfg.model_size != 48:
		raise ValueError("FlowMPC mt80 training requires model_size=48 for mt80-48M initialization")
	if min(cfg.steps, cfg.batch_size, cfg.horizon, cfg.eval_freq, cfg.eval_episodes) < 1:
		raise ValueError("Training steps, batch size, horizon and evaluation settings must be positive")
	if not torch.cuda.is_available():
		raise RuntimeError("FlowMPC training requires CUDA")
	if cfg.task != "mt80" or cfg.obs != "state":
		raise ValueError("FlowMPC training requires task=mt80 and obs=state")
	data_dir = _resolve_path(cfg, "data_dir")
	if not data_dir.is_dir() or not any(data_dir.glob("*.pt")):
		raise FileNotFoundError(f"data_dir must contain mt80 .pt chunks: {data_dir}")
	_require_file(cfg, "tdmpc_checkpoint")
	_require_file(cfg, "fm_checkpoint")
	if cfg.flowmpc_train_mode == "frozen":
		selection_path = Path(cfg.fm_checkpoint).with_name("fm_selection.pt")
		if not selection_path.is_file():
			raise FileNotFoundError(f"FM selection metadata does not exist: {selection_path}")


@hydra.main(version_base=None, config_path="..", config_name="flowmpc/config")
def train(cfg):
	cfg = parse_cfg(cfg)
	_validate_training_cfg(cfg)
	set_seed(cfg.seed)
	env = make_env(cfg)
	agent = FlowMPC(cfg)
	agent.load_pretrained_tdmpc(cfg.tdmpc_checkpoint)
	buffer = build_flowmpc_replay(cfg)
	FlowMPCOfflineTrainer(
		cfg=cfg,
		env=env,
		agent=agent,
		buffer=buffer,
		logger=Logger(cfg),
	).train()


if __name__ == "__main__":
	train()
