"""Flush-through terminal logging for the FlowMPC Hydra entrypoint."""

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import os
import sys
import threading
import traceback
import uuid


class _Tee:
	def __init__(self, stream, logfile, lock):
		self.stream, self.logfile, self.lock = stream, logfile, lock

	def write(self, text):
		with self.lock:
			result = self.stream.write(text)
			self.logfile.write(text)
			self.stream.flush()
			self.logfile.flush()
		return len(text) if result is None else result

	def flush(self):
		with self.lock:
			self.stream.flush()
			self.logfile.flush()

	def __getattr__(self, name):
		return getattr(self.stream, name)


@contextmanager
def terminal_log(work_dir):
	"""Keep terminal output and capture failures before Hydra handles the exception."""
	directory = Path(work_dir) / "terminal"
	directory.mkdir(parents=True, exist_ok=True)
	stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
	path = directory / f"{stamp}-{os.getpid()}-{uuid.uuid4().hex[:8]}.log"
	stdout, stderr = sys.stdout, sys.stderr
	lock = threading.RLock()
	with path.open("x", encoding="utf-8", errors="backslashreplace", buffering=1) as logfile:
		sys.stdout, sys.stderr = _Tee(stdout, logfile, lock), _Tee(stderr, logfile, lock)
		try:
			print(f"Terminal log: {path}")
			yield path
		except BaseException:
			print("FlowMPC run failed:", file=sys.stderr)
			traceback.print_exc(file=sys.stderr)
			raise
		else:
			print("FlowMPC run completed successfully")
		finally:
			try:
				sys.stdout.flush()
				sys.stderr.flush()
			finally:
				sys.stdout, sys.stderr = stdout, stderr
