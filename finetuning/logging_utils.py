# coding=utf-8
"""Copy everything a script prints (stdout, stderr, warnings, tracebacks) into a log file.

    from logging_utils import setup_logging
    setup_logging("run.log", args)

Console output is unchanged. In the file every line gets a timestamp, and the file is flushed on each write, so it
is complete even if the run is killed. The file is opened in append mode, so reruns add to it.
"""
import atexit
import datetime
import os
import re
import subprocess
import sys


class _Tee:
    def __init__(self, stream, file):
        self._stream = stream
        self._file = file
        self._at_line_start = True

    def write(self, data):
        self._stream.write(data)
        for part in data.splitlines(keepends=True):
            if self._at_line_start and part.strip():
                self._file.write(datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S] "))
            self._file.write(part)
            self._at_line_start = part.endswith(("\n", "\r"))
        self._file.flush()
        return len(data)

    def flush(self):
        self._stream.flush()
        self._file.flush()

    def __getattr__(self, name):  # isatty(), fileno(), encoding, ...
        return getattr(self._stream, name)


def _git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                             cwd=os.path.dirname(os.path.abspath(__file__)), timeout=5)
        return out.stdout.strip() or None
    except Exception:
        return None


def setup_logging(log_file, args=None):
    """Tee stdout and stderr into `log_file`. Returns the path, or None on non-zero ranks of a distributed run."""
    if os.environ.get("RANK", "0") not in ("0", ""):
        return None  # only the main process writes the log

    log_file = os.path.abspath(log_file)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    file = open(log_file, "a", encoding="utf-8", buffering=1)

    original_stdout, original_stderr = sys.stdout, sys.stderr
    sys.stdout = _Tee(original_stdout, file)
    sys.stderr = _Tee(original_stderr, file)

    print("=" * 80)
    print(f"Run started {datetime.datetime.now().isoformat(timespec='seconds')}")
    print(f"Command: {' '.join(sys.argv)}")
    commit = _git_commit()
    if commit:
        print(f"Git commit: {commit}")
    if args is not None:
        print("Arguments:")
        for key, value in sorted(vars(args).items()):
            print(f"  {key}: {value}")
    print(f"Log file: {log_file}")
    print("=" * 80)

    def _close():
        print(f"Run finished {datetime.datetime.now().isoformat(timespec='seconds')}")
        sys.stdout.flush()
        sys.stderr.flush()
        # put the real streams back before closing the file, so the interpreter's final flush cannot hit a closed file
        sys.stdout, sys.stderr = original_stdout, original_stderr
        file.close()

    atexit.register(_close)
    return log_file
