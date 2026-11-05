"""External tool invocation: sanitized environment, live pass-through, per-invocation log."""

import os
import shlex
import signal
import subprocess
import threading
from datetime import datetime

from .status import InputError, write_stderr, write_stdout

DROPPED_VARS = ("PYTHONHOME", "PYTHONPATH", "PYTHONNOUSERSITE", "ITK_AUTOLOAD_PATH")
SLICER_FILTERED_VARS = ("LD_LIBRARY_PATH", "PATH", "QT_PLUGIN_PATH")


def _under(path, root):
    path = os.path.normpath(os.path.abspath(path))
    root = os.path.normpath(os.path.abspath(root))
    return path == root or path.startswith(root.rstrip(os.sep) + os.sep)


def child_env(environ=None, extra=None):
    """The environment every child runs with, identical from Slicer and from a terminal.

    Drops the Python and ITK variables Slicer sets, removes entries under ``$SLICER_HOME`` from
    ``LD_LIBRARY_PATH``, ``PATH`` and ``QT_PLUGIN_PATH`` (dropping a variable left empty) and
    drops ``SLICER_HOME``.
    """
    env = dict(os.environ if environ is None else environ)
    for name in DROPPED_VARS:
        env.pop(name, None)
    slicer_home = env.pop("SLICER_HOME", "")
    for name in SLICER_FILTERED_VARS:
        if name not in env:
            continue
        entries = [e for e in env[name].split(os.pathsep) if e]
        if slicer_home:
            entries = [e for e in entries if not _under(e, slicer_home)]
        if entries:
            env[name] = os.pathsep.join(entries)
        else:
            env.pop(name)
    env["PYTHONUNBUFFERED"] = "1"
    if extra:
        env.update(extra)
    return env


def new_log_path(project, scope, step, suffix=""):
    """``<project>/logs/<scope>/<step>/<YYYYmmdd_HHMMSS>_<step>[_<suffix>].log`` (never an existing file)."""
    folder = os.path.join(project, "logs", scope, step)
    os.makedirs(folder, exist_ok=True)
    stem = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{step}" + (f"_{suffix}" if suffix else "")
    path = os.path.join(folder, stem + ".log")
    n = 2
    while os.path.exists(path):
        path = os.path.join(folder, f"{stem}_{n}.log")
        n += 1
    return path


class _Interrupted(Exception):
    pass


def run_tool(argv, cwd, log_path, env=None, on_stdout_line=None, on_stderr_line=None):
    """Run ``argv``; its stdout/stderr pass through unchanged and are teed to ``log_path``.

    The log starts with the exact argv and cwd. ``on_stdout_line``/``on_stderr_line`` receive each
    decoded line. SIGTERM or Ctrl-C stop the child's whole process group. Returns the exit status.
    """
    argv = [str(a) for a in argv]
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log = open(log_path, "wb")
    log.write(f"argv: {shlex.join(argv)}\ncwd: {cwd}\n\n".encode("utf-8"))
    log.flush()
    log_lock = threading.Lock()
    try:
        proc = subprocess.Popen(argv, cwd=cwd, env=child_env() if env is None else env,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    except OSError as exc:
        log.write(f"cannot start: {exc}\n".encode("utf-8"))
        log.close()
        raise InputError(f"cannot run {argv[0]}: {exc.strerror or exc}")

    def pump(stream, write, callback):
        for raw in iter(stream.readline, b""):
            with log_lock:
                log.write(raw)
                log.flush()
            write(raw)
            if callback is not None:
                callback(raw.decode("utf-8", errors="replace").rstrip("\r\n"))
        stream.close()

    threads = [threading.Thread(target=pump, args=(proc.stdout, write_stdout, on_stdout_line), daemon=True),
               threading.Thread(target=pump, args=(proc.stderr, write_stderr, on_stderr_line), daemon=True)]
    for t in threads:
        t.start()

    def on_term(signum, frame):
        raise _Interrupted()

    previous = None
    if threading.current_thread() is threading.main_thread():
        previous = signal.signal(signal.SIGTERM, on_term)
    try:
        rc = proc.wait()
        for t in threads:
            t.join()
    except (KeyboardInterrupt, _Interrupted):
        if previous is not None:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
        _stop_group(proc)
        for t in threads:
            t.join(timeout=5)
        with log_lock:
            log.write(b"\n[stopped by the caller]\n")
        log.close()
        raise KeyboardInterrupt()
    finally:
        if previous is not None:
            signal.signal(signal.SIGTERM, previous)
    with log_lock:
        log.write(f"\n[exit status {rc}]\n".encode("utf-8"))
    log.close()
    return rc


def _stop_group(proc):
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()
