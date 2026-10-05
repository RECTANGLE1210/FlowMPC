"""Standalone mt80 Flow Matching training entrypoint."""

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
from omegaconf import OmegaConf
from tqdm import trange

from common.parser import parse_cfg
from common.seed import set_seed
from envs import make_env

from .data import load_selected_buffer, select_episodes
from .policy import MultiTaskFlowMatchingPolicy


torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")


def _validate_training_cfg(cfg):
	if cfg.task != "mt80" or cfg.obs != "state":
		raise ValueError("FM training requires task=mt80 and obs=state")
	if not torch.cuda.is_available():
		raise RuntimeError("FM training requires CUDA because TD-MPC2 Buffer uses cuda:0")
	data_dir = Path(to_absolute_path(str(cfg.data_dir)))
	if not data_dir.is_dir() or not any(data_dir.glob("*.pt")):
		raise FileNotFoundError(f"data_dir must contain mt80 .pt chunks: {data_dir}")
	cfg.data_dir = str(data_dir)


@hydra.main(version_base=None, config_path="../..", config_name="flowmpc/fm/config")
def train(cfg):
	"""Select balanced mt80 replay episodes and train the standalone FM policy."""
	resolved_cfg = cfg
	cfg = parse_cfg(cfg)
	_validate_training_cfg(cfg)
	set_seed(cfg.seed)

	env = make_env(cfg)
	output_dir = Path(cfg.work_dir) / "flowmpc" / "fm"
	output_dir.mkdir(parents=True, exist_ok=True)
	OmegaConf.save(resolved_cfg, output_dir / "config.yaml", resolve=True)
	metadata = select_episodes(cfg, output_dir / "fm_selection.pt")
	buffer = load_selected_buffer(cfg, metadata)
	del env

	policy = MultiTaskFlowMatchingPolicy(cfg).to("cuda:0")
	scheduler = torch.optim.lr_scheduler.OneCycleLR(
		policy.optim, max_lr=cfg.flow_lr, total_steps=cfg.flow_training_steps
	)
	progress = trange(cfg.flow_training_steps, desc="fm_train")
	for step in progress:
		policy.train()
		obs, action, _, _, task = buffer.sample()
		metrics = policy.update(state=obs[0], action=action, task=task)
		scheduler.step()
		if (step + 1) % cfg.flow_log_every == 0 or step == 0:
			metrics["flow_lr"] = scheduler.get_last_lr()[0]
			progress.set_postfix(metrics)
		if (step + 1) % cfg.flow_save_every == 0:
			policy.save(output_dir / f"flow_matching_model_{step + 1}.pt")
	policy.save(output_dir / "flow_matching_model.pt")


if __name__ == "__main__":
	train()