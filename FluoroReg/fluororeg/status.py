"""Status lines and the errors that set the exit status.

Every event is one ``FLUOROREG {json}`` line on stdout with ``event`` (``start``, ``progress``,
``warning``, ``error``, ``result``) and ``step``. Child tool output goes to the same streams through
:func:`write_stdout` / :func:`write_stderr`, so status lines never split a child's line.
"""

import json
import sys
import threading

PREFIX = "FLUOROREG"

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_INVALID = 2
EXIT_ATTENTION = 3

_lock = threading.Lock()


class PipelineError(Exception):
    """A failure with a defined exit status; ``details`` (optional) go into the error event."""

    exit_code = EXIT_FAILURE
    kind = "failure"

    def __init__(self, message, details=None):
        super().__init__(message)
        self.details = details


class InputError(PipelineError):
    """Invalid input, configuration or missing tool path (exit 2)."""

    exit_code = EXIT_INVALID
    kind = "input"


class ToolError(PipelineError):
    """An external tool failed (exit 1)."""

    kind = "tool"


def _write(stream, data):
    with _lock:
        buffer = getattr(stream, "buffer", None)
        if buffer is not None:
            stream.flush()
            buffer.write(data)
            buffer.flush()
        else:
            stream.write(data.decode("utf-8", errors="replace"))
            stream.flush()


def write_stdout(data):
    _write(sys.stdout, data)


def write_stderr(data):
    _write(sys.stderr, data)


def emit(event, step, **fields):
    record = {"event": event, "step": step}
    record.update(fields)
    write_stdout(f"{PREFIX} {json.dumps(record, default=str)}\n".encode("utf-8"))


def parse_status_line(line, prefix=PREFIX):
    """The JSON record of a ``<prefix> {json}`` line, else None."""
    text = line.strip()
    if not text.startswith(prefix + " "):
        return None
    try:
        record = json.loads(text[len(prefix) + 1:])
    except ValueError:
        return None
    return record if isinstance(record, dict) else None
