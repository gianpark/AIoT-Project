"""src/logic/personal_calibration.py 단위 테스트 (11/26 선우 구현 검증)."""
import math

from src.logic.personal_calibration import PersonalPostureCalibrator


def _fill(cal, values):
    cal.start()
    for v in values:
        cal.add_sample(v)


def test_not_collecting_before_start():
    cal = PersonalPostureCalibrator(required_samples=3)
    assert cal.add_sample(0.15) is False
    assert not cal.is_calibrated


def test_median_and_offset():
    cal = PersonalPostureCalibrator(required_samples=5)
    _fill(cal, [0.16, 0.18, 0.17, 0.175, 0.165])
    assert cal.is_calibrated and not cal.is_collecting
    assert math.isclose(cal.baseline_nf_m, 0.17)
    assert not cal.is_clipped and not cal.is_rejected
    assert math.isclose(cal.offset_m, 0.17 - 0.143)


def test_invalid_samples_rejected():
    cal = PersonalPostureCalibrator(required_samples=2)
    cal.start()
    for bad in (None, "x", float("nan"), float("inf"), -0.5, 0.9):
        assert cal.add_sample(bad) is False
    assert cal.sample_count == 0


def test_apply_shifts_only_neck_forward():
    cal = PersonalPostureCalibrator(required_samples=1)
    _fill(cal, [0.163])
    feats = {"neck_forward_offset_m": 0.20, "torso_recline_offset_m": 0.05}
    out = cal.apply(feats)
    assert math.isclose(out["neck_forward_offset_m"], 0.20 - 0.02)
    assert out["torso_recline_offset_m"] == 0.05
    assert feats["neck_forward_offset_m"] == 0.20  # 원본은 그대로


def test_apply_passthrough_before_calibration_and_missing():
    cal = PersonalPostureCalibrator(required_samples=1)
    assert cal.apply(None) is None
    assert cal.apply({"neck_forward_offset_m": 0.2})["neck_forward_offset_m"] == 0.2
    _fill(cal, [0.15])
    assert cal.apply({"neck_forward_offset_m": None})["neck_forward_offset_m"] is None


def test_restart_discards_previous_calibration():
    cal = PersonalPostureCalibrator(required_samples=1)
    _fill(cal, [0.15])
    cal.start()
    assert not cal.is_calibrated and cal.is_collecting and cal.sample_count == 0


def test_unstable_collection_rejected():
    cal = PersonalPostureCalibrator(required_samples=5)
    _fill(cal, [0.10, 0.20, 0.16, 0.18, 0.17])  # 10~90% 폭 0.068 m > 0.04 m
    assert cal.is_rejected and not cal.is_calibrated and not cal.is_collecting
    assert cal.spread_m > 0.04
    assert cal.apply({"neck_forward_offset_m": 0.2})["neck_forward_offset_m"] == 0.2
    assert "press 'c' again" in cal.status_text()


def test_restart_after_rejection():
    cal = PersonalPostureCalibrator(required_samples=5)
    _fill(cal, [0.10, 0.20, 0.16, 0.18, 0.17])
    _fill(cal, [0.15] * 5)
    assert cal.is_calibrated and not cal.is_rejected


def test_offset_clipped_both_directions():
    hi = PersonalPostureCalibrator(required_samples=3)
    _fill(hi, [0.20] * 3)  # 이동량 0.057 m -> 0.03 m
    assert hi.is_clipped and math.isclose(hi.offset_m, 0.03)
    assert math.isclose(hi.apply({"neck_forward_offset_m": 0.20})["neck_forward_offset_m"], 0.17)
    lo = PersonalPostureCalibrator(required_samples=3)
    _fill(lo, [0.08] * 3)  # 이동량 -0.063 m -> -0.03 m
    assert lo.is_clipped and math.isclose(lo.offset_m, -0.03)


def test_guards_can_be_disabled():
    cal = PersonalPostureCalibrator(required_samples=5, max_spread_m=None, max_offset_m=None)
    _fill(cal, [0.10, 0.30, 0.25, 0.28, 0.27])
    assert cal.is_calibrated and math.isclose(cal.offset_m, 0.27 - 0.143)


def test_percentile_matches_numpy_default():
    import numpy as np
    from src.logic.personal_calibration import _percentile
    xs = [0.12, 0.31, 0.05, 0.22, 0.18, 0.09, 0.27]
    for q in (10, 50, 90):
        assert math.isclose(_percentile(xs, q), float(np.percentile(xs, q)))
