"""Landmark status vocabulary shared by every detector.

The pipeline previously had no way to say "I don't know": each detector widened its search
window until it was non-empty and then returned an ``argmax``, and a failed sphere fit was
accepted whenever a weighted score exceeded 0.1.  A landmark returned for anatomy that is
outside the field of view silently poisons pose initialisation downstream, so absence has to
be representable and distinguishable from "not attempted".

Rules, enforced by the writers:

* ``position`` is ``None`` if and only if ``status`` is not ``OK`` or ``LOW_CONFIDENCE``.
  There is no sentinel coordinate anywhere - a missing landmark is never ``(0, 0, 0)``.
* only ``OK`` records reach the canonical FCSV (``LOW_CONFIDENCE`` behind an explicit flag);
* the debug JSON carries a record for every landmark the detector was asked for, including
  the ones that abstained, so a consumer can tell "not configured" from "not attempted" from
  "attempted and abstained".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


class Status:
    OK = "ok"
    LOW_CONFIDENCE = "low_confidence"
    ABSTAINED = "abstained"
    FAILED = "failed"
    NOT_ATTEMPTED = "not_attempted"


class Reason:
    EMPTY_REGION = "empty_region"                # candidate region had no vertices
    RESIDUAL_TOO_HIGH = "residual_too_high"      # fit RMS above tolerance
    OUT_OF_PRIOR = "out_of_prior"                # value outside the anatomical prior interval
    INSUFFICIENT_SUPPORT = "insufficient_support"  # too few inliers backing the estimate
    POOR_COVERAGE = "poor_coverage"              # support spans too small a solid angle
    UNSTABLE = "unstable"                        # bootstrap spread above tolerance
    AMBIGUOUS = "ambiguous"                      # competing candidates within tolerance
    MISSING_INPUT = "missing_input"              # a required mesh or landmark is absent
    FRAME_UNRESOLVED = "frame_unresolved"        # the anatomical frame could not be built
    NOT_CONFIGURED = "not_configured"            # the anatomy is not part of this run


@dataclass
class LandmarkRecord:
    """One detector's answer about one landmark."""

    landmark_id: str
    status: str
    position: Optional[np.ndarray] = None
    confidence: float = 0.0
    reason: Optional[str] = None
    residual_mm: Optional[float] = None
    support: int = 0
    provenance: str = ""
    metrics: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.status not in (Status.OK, Status.LOW_CONFIDENCE):
            self.position = None
            self.confidence = 0.0
        elif self.position is None:
            raise ValueError(f"{self.landmark_id}: {self.status} requires a position")

    @property
    def emitted(self):
        """Whether the record carries a usable coordinate."""
        return self.position is not None

    def to_json(self):
        return {
            "status": self.status,
            "position_lps": None if self.position is None else [float(v) for v in self.position],
            "confidence": round(float(self.confidence), 4),
            "reason": self.reason,
            "residual_mm": None if self.residual_mm is None else round(float(self.residual_mm), 4),
            "support": int(self.support),
            "provenance": self.provenance,
            "metrics": {
                key: (round(float(value), 4)
                      if isinstance(value, (int, float)) and not isinstance(value, bool)
                      else value)
                for key, value in self.metrics.items()
            },
        }


def ok(landmark_id, position, *, confidence=1.0, provenance="", **kwargs):
    return LandmarkRecord(landmark_id, Status.OK, np.asarray(position, dtype=float),
                          confidence=confidence, provenance=provenance, **kwargs)


def low_confidence(landmark_id, position, *, confidence, reason, provenance="", **kwargs):
    return LandmarkRecord(landmark_id, Status.LOW_CONFIDENCE, np.asarray(position, dtype=float),
                          confidence=confidence, reason=reason, provenance=provenance, **kwargs)


def abstained(landmark_id, reason, *, provenance="", **kwargs):
    return LandmarkRecord(landmark_id, Status.ABSTAINED, None, reason=reason,
                          provenance=provenance, **kwargs)


def not_attempted(landmark_id, reason=Reason.MISSING_INPUT, *, provenance="", **kwargs):
    """The detector never ran for this landmark - distinct from running and abstaining."""
    return LandmarkRecord(landmark_id, Status.NOT_ATTEMPTED, None, reason=reason,
                          provenance=provenance, **kwargs)


def records_to_json(records):
    return {record.landmark_id: record.to_json() for record in records.values()}


def positions(records):
    """The subset that carries a coordinate, as the plain ``{name: array}`` map detectors return."""
    return {rid: record.position for rid, record in records.items() if record.emitted}
