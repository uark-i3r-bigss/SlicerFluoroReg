"""Machine-readable status lines for callers (the GUI parses them; people can read them).

Every line is ``BIGSS_POSE <json>`` on stdout, flushed. Events: ``start``, ``progress``,
``result``, ``warning``, ``error``. Human-oriented detail goes to stderr.
"""

from __future__ import annotations

import json
import sys
import time

PREFIX = "BIGSS_POSE"


def emit(event: str, **fields) -> None:
    rec = {"event": event, "time": round(time.time(), 3), **fields}
    print(f"{PREFIX} {json.dumps(rec, default=str, sort_keys=True)}", flush=True)


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)
