"""Deterministic random-number plumbing.

Every stochastic step in the pipeline draws from an explicitly passed
``numpy.random.Generator`` derived from a fixed root seed plus the case id and the stage
name.  Two runs over the same input therefore produce bit-identical output, which is the
property the pipeline claims (no training data, deterministic, auditable) and did not have:
`landmarks/femur_landmarks.py` previously called the global `np.random.choice`, and the
shipped sample records the consequence, a 0.17-0.92 mm femoral-head spread over five
repeats of the identical phantom (examples/README.md).

The reference implementations avoid the problem by being deterministic outright
(algebraic least squares rather than RANSAC), so there is no upstream convention to copy;
this module follows the NumPy `SeedSequence` spawning guidance instead.
"""

from __future__ import annotations

import hashlib

import numpy as np

#: Root entropy for the whole pipeline.  Change only with a recorded reason: every
#: published number is conditioned on it.
ROOT_SEED = 0xC17D


def _stable_hash(text: str) -> int:
    """A process-independent 32-bit hash (``hash()`` is salted per interpreter run)."""
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "big")


def rng_for(case_id: str, stage: str, root_seed: int = ROOT_SEED) -> np.random.Generator:
    """Return the generator for one (case, stage) pair.

    Stages are independent: re-running one stage cannot shift the draws of another, and a
    case processed alone gives the same result as the same case processed in a batch.
    """
    sequence = np.random.SeedSequence(
        entropy=root_seed,
        spawn_key=(_stable_hash(case_id or ""), _stable_hash(stage or "")),
    )
    return np.random.default_rng(sequence)
