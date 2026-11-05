"""Pinned third-party weights. Downloaded by scripts/setup_env.sh; never committed."""

from __future__ import annotations

import os
from pathlib import Path

from .xreg_io import sha256_file

REPO_ROOT = Path(__file__).resolve().parents[2]

FOUNDATION = {
    "name": "xvr_wbct",
    "url": "https://huggingface.co/eigenvivek/xvr/resolve/main/wbct/model.pth",
    "sha256": "82a9fd05a61f22da949885e890bce9333e4d5abc091ec011ddfc4b3454d5ffba",
    "relpath": "xvr_wbct/model.pth",
    "notes": "xvr whole-body foundation pose regressor (resnet34, 300k iterations, MIT licence)",
}


def weights_dir() -> Path:
    return Path(os.environ.get("BIGSS_POSE_WEIGHTS_DIR", REPO_ROOT / "models"))


def foundation_path(verify: bool = True) -> Path:
    p = weights_dir() / FOUNDATION["relpath"]
    if not p.is_file():
        raise FileNotFoundError(f"{p} missing: run scripts/setup_env.sh to fetch the foundation weights")
    if verify and sha256_file(p) != FOUNDATION["sha256"]:
        raise ValueError(f"{p}: sha256 does not match the pinned foundation weights")
    return p
