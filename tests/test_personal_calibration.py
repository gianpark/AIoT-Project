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
    _fill(cal, [0.10, 0.20, 0.16, 0.18, 0.17])
    assert cal.is_calibrated and not cal.is_collecting
    assert math.isclose(cal.baseline_nf_m, 0.17)
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
