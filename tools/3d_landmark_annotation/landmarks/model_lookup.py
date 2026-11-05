"""Locating the per-anatomy surface models a detector needs."""

import glob
import os


def find_model_by_name(models_dir, pattern):
    """Deterministic match for a model glob, e.g. ``'*hip_left.stl'``.

    Several files can match when a case carries both the target-anatomy and the all-label
    exports; the shortest name wins so the choice does not depend on directory order.
    """
    matches = sorted(glob.glob(os.path.join(models_dir, pattern)), key=len)
    return matches[0] if matches else None
