# Training checkpoints and logs

Use `python -m flowmpc.train` from `tdmpc2/`. `flowmpc_save_freq=10000`
saves after that many completed updates; `0` disables periodic saves only.
Normal completion always writes `cfg.work_dir/models/latest.pt`. Each save
writes and fsyncs a same-directory temporary file, then atomically replaces
`latest.pt`. Ordinary write/replace failures retain the previous checkpoint.

Set `flowmpc_resume_checkpoint=/path/to/latest.pt` to restore model (including
target Q and pi_BC), optimizer, RunningScale, completed-update count, and RNG.
`steps` is the TOTAL target: resuming 200000 with `steps=512000` runs 312000
more updates. Initialization and replay loading happen before RNG restoration.
Only load trusted training checkpoint files. Model-only inference checkpoints
cannot resume optimizer training. Frozen FM weights are loaded separately;
SHA-256 verifies their identity without embedding them in `latest.pt`.

Compatibility checks cover task ordering/subset, mode, model/loss/optimizer
configuration, optimizer parameter groups, dataset file names/sizes and frozen
selection identity. Runtime logging/saving settings and total steps may change.
Changing evaluation settings can change subsequent training RNG consumption.

Pinned TorchRL 0.8.1 SliceSampler uses global torch RNG by default and has an
empty sampler state_dict. Checkpoints save global RNG plus any explicit sampler
generator; its caches are rebuilt from unchanged replay. Synthetic CPU tests
check sampler sequence continuation. This is not a full reproducibility promise:
torch.compile/CUDA, different devices/library versions, asynchronous libraries,
and environment-local evaluation RNG/state can prevent bitwise continuation.
Replay storage-device changes are warned about. No replay data is checkpointed.

Each training invocation automatically tees Python stdout/stderr to a new UTC
timestamped file under `cfg.work_dir/terminal/`, flushing every write. Exceptions
and tracebacks are captured before Hydra handles them. Existing logs are never
overwritten. Abrupt kills cannot guarantee the last OS/filesystem write survives;
native code writing directly to OS file descriptors may bypass the Python tee.
