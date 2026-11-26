"""Session-based personal posture calibration.

Collects neck_forward_offset_m samples while the user is deliberately sitting
in a normal posture. The median is used as the personal baseline.

This module does not modify decision thresholds. It only returns a copied
feature dictionary with neck_forward_offset_m shifted toward target_nf_m.

Stability guards (11/26, replay evaluation experiments/calib_guard_eval.md):
- If the 10-90 percentile spread of the collected samples exceeds
  max_spread_m (0.04 m), the calibration is rejected and the user is asked
  to press 'c' again (p07: spread 0.077 m, calibration hurt detection).
- The applied shift is clipped to +/- max_offset_m (0.03 m) so that a large
  shift does not hide the opposite posture (the same feature drives both
  neck_forward and slouch_back in decision.judge).
Both thresholds are provisional values chosen from the current data.
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
    max_spread_m: Optional[float] = 0.04
    max_offset_m: Optional[float] = 0.03

    _samples: list[float] = field(default_factory=list, init=False)
    _collecting: bool = field(default=False, init=False)
    _baseline_nf_m: Optional[float] = field(default=None, init=False)
    _offset_m: Optional[float] = field(default=None, init=False)
    _spread_m: Optional[float] = field(default=None, init=False)
    _rejected: bool = field(default=False, init=False)
    _clipped: bool = field(default=False, init=False)

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

    @property
    def spread_m(self) -> Optional[float]:
        """10-90 percentile spread of the last completed collection."""
        return self._spread_m

    @property
    def is_rejected(self) -> bool:
        """True when the last collection was rejected as unstable."""
        return self._rejected

    @property
    def is_clipped(self) -> bool:
        """True when the applied shift was limited by max_offset_m."""
        return self._clipped

    def start(self) -> None:
        """Start or restart collection; previous calibration is discarded."""
        self._samples.clear()
        self._baseline_nf_m = None
        self._offset_m = None
        self._spread_m = None
        self._rejected = False
        self._clipped = False
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
            self._finish()

        return True

    def _finish(self) -> None:
        self._collecting = False
        self._spread_m = _percentile(self._samples, 90) - _percentile(self._samples, 10)
        if self.max_spread_m is not None and self._spread_m > self.max_spread_m:
            self._rejected = True  # unstable posture during collection: keep uncalibrated
            return
        self._baseline_nf_m = float(median(self._samples))
        offset = self._baseline_nf_m - self.target_nf_m
        if self.max_offset_m is not None and abs(offset) > self.max_offset_m:
            offset = self.max_offset_m if offset > 0 else -self.max_offset_m
            self._clipped = True
        self._offset_m = offset

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
                f"baseline={self._baseline_nf_m:.3f} m"
                f"{', shift clipped' if self._clipped else ''})"
            )
        if self.is_rejected:
            return (
                f"Personal calibration: rejected (unstable, spread="
                f"{self._spread_m:.3f} m > {self.max_spread_m:.2f} m) - "
                f"sit still and press 'c' again"
            )
        if self.is_collecting:
            return (
                f"Personal calibration: collecting "
                f"{self.sample_count}/{self.required_samples}"
            )
        return "Personal calibration: press 'c' to calibrate"


def _percentile(values: list[float], q: float) -> float:
    """Linear-interpolation percentile (same as numpy's default)."""
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    pos = (len(xs) - 1) * q / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    return float(xs[lo] + (xs[hi] - xs[lo]) * (pos - lo))
