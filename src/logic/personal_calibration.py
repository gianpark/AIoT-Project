"""Session-based personal posture calibration.

Collects neck_forward_offset_m samples while the user is deliberately sitting
in a normal posture. The median is used as the personal baseline.

This module does not modify decision thresholds. It only returns a copied
feature dictionary with neck_forward_offset_m shifted toward target_nf_m.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median
from typing import Optional


@dataclass
class PersonalPostureCalibrator:
    """Collect and apply a session-only neck-forward baseline."""

    target_nf_m: float = 0.143
    required_samples: int = 150
    min_valid_nf_m: float = -0.20
    max_valid_nf_m: float = 0.50

    _samples: list[float] = field(default_factory=list, init=False)
    _collecting: bool = field(default=False, init=False)
    _baseline_nf_m: Optional[float] = field(default=None, init=False)
    _offset_m: Optional[float] = field(default=None, init=False)

    @property
    def is_collecting(self) -> bool:
        return self._collecting

    @property
    def is_calibrated(self) -> bool:
        return self._offset_m is not None

    @property
    def sample_count(self) -> int:
        return len(self._samples)

    @property
    def progress(self) -> float:
        return min(1.0, len(self._samples) / self.required_samples)

    @property
    def baseline_nf_m(self) -> Optional[float]:
        return self._baseline_nf_m

    @property
    def offset_m(self) -> Optional[float]:
        return self._offset_m

    def start(self) -> None:
        """Start or restart collection; previous calibration is discarded."""
        self._samples.clear()
        self._baseline_nf_m = None
        self._offset_m = None
        self._collecting = True

    def cancel(self) -> None:
        """Stop collection without changing an already completed calibration."""
        self._collecting = False

    def add_sample(self, value: object) -> bool:
        """Add one valid neck-forward sample. Returns True if accepted."""
        if not self._collecting or self.is_calibrated:
            return False

        try:
            sample = float(value)
        except (TypeError, ValueError):
            return False

        if not (-float("inf") < sample < float("inf")):
            return False
        if not self.min_valid_nf_m <= sample <= self.max_valid_nf_m:
            return False

        self._samples.append(sample)

        if len(self._samples) >= self.required_samples:
            self._baseline_nf_m = float(median(self._samples))
            self._offset_m = self._baseline_nf_m - self.target_nf_m
            self._collecting = False

        return True

    def apply(self, depth_features: Optional[dict]) -> Optional[dict]:
        """Return a copied feature dict with only neck_forward_offset_m adjusted.

        Before calibration, or when the feature is missing/invalid, the input
        values are passed through unchanged.
        """
        if depth_features is None:
            return None

        adjusted = dict(depth_features)
        if not self.is_calibrated:
            return adjusted

        value = adjusted.get("neck_forward_offset_m")
        try:
            current = float(value)
        except (TypeError, ValueError):
            return adjusted

        if not (-float("inf") < current < float("inf")):
            return adjusted

        adjusted["neck_forward_offset_m"] = current - self.offset_m
        return adjusted

    def status_text(self) -> str:
        if self.is_calibrated:
            return (
                f"Personal calibration: complete "
                f"({self.sample_count}/{self.required_samples}, "
                f"baseline={self._baseline_nf_m:.3f} m)"
            )
        if self.is_collecting:
            return (
                f"Personal calibration: collecting "
                f"{self.sample_count}/{self.required_samples}"
            )
        return "Personal calibration: press 'c' to calibrate"
