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

from common.buffer import Buffer
from common.logger import Logger
from common.parser import parse_cfg
from common.seed import set_seed
from envs import make_env
from trainer.offline_trainer import OfflineTrainer

from .agent import FlowMPC


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
	if not torch.cuda.is_available():
		raise RuntimeError("FlowMPC training requires CUDA")
	if cfg.task != "mt80" or cfg.obs != "state":
		raise ValueError("FlowMPC training requires task=mt80 and obs=state")
	data_dir = _resolve_path(cfg, "data_dir")
	if not data_dir.is_dir() or not any(data_dir.glob("*.pt")):
		raise FileNotFoundError(f"data_dir must contain mt80 .pt chunks: {data_dir}")
	_require_file(cfg, "tdmpc_checkpoint")
	_require_file(cfg, "fm_checkpoint")


@hydra.main(version_base=None, config_path="..", config_name="flowmpc/config")
def train(cfg):
	cfg = parse_cfg(cfg)
	_validate_training_cfg(cfg)
	set_seed(cfg.seed)
	env = make_env(cfg)
	try:
		agent = FlowMPC(cfg)
		agent.load_pretrained_tdmpc(cfg.tdmpc_checkpoint)
		OfflineTrainer(
			cfg=cfg,
			env=env,
			agent=agent,
			buffer=Buffer(cfg),
			logger=Logger(cfg),
		).train()
	finally:
		env.close()


if __name__ == "__main__":
	train()