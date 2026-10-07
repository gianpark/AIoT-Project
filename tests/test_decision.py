from src.logic.decision import (CAUTION, NORMAL, WARNING, AlertStateMachine, Judgement, judge)
from src.pose.movenet_keypoints import Keypoint


def kps(nose_x=0.5, shoulder_dx=0.0):
    """shoulder_dx: 어깨를 화면 가로로 밀어 몸통을 기울인 정도(화면 비율). 코는 nose_x로 따로 움직임."""
    base = {5: (0.42 + shoulder_dx, 0.40), 6: (0.58 + shoulder_dx, 0.40), 11: (0.44, 0.65), 12: (0.56, 0.65),
            0: (nose_x, 0.28)}
    return [Keypoint(name=f"k{i}", y=base.get(i, (0.5, 0.5))[1], x=base.get(i, (0.5, 0.5))[0], score=0.9)
            for i in range(17)]


def depth(head=0.55, chest=0.66, recline=-0.05):
    return {"head_depth_m": head, "chest_depth_m": chest, "hip_depth_m": chest + recline,
            "neck_forward_offset_m": chest - head, "torso_recline_offset_m": recline}


def test_normal_posture_is_normal():
    j = judge(kps(), depth())
    assert j.posture_level == NORMAL and not j.proximity and not j.needs_attention


def test_forward_slouch_and_proximity():
    j = judge(kps(), depth(head=0.35, chest=0.45, recline=0.17))
    assert j.posture_level == WARNING and j.posture_kind == "slouch_forward" and j.proximity


def test_slouch_back_warning():
    j = judge(kps(), depth(head=0.79, chest=0.82, recline=-0.30))
    assert j.posture_level == WARNING and j.posture_kind == "slouch_back"


def test_tilt_direction_uses_participant_side():
    # 이미지 왼쪽(x 작음) = 참가자 본인 기준 오른쪽
    left_img = judge(kps(shoulder_dx=-0.10), depth())
    right_img = judge(kps(shoulder_dx=0.10), depth())
    assert left_img.posture_kind == "tilt_right" and right_img.posture_kind == "tilt_left"


def test_head_only_tilt_is_not_a_tilt():
    """고개만 옆으로 기울이고 몸통이 곧으면 기울임이 아니다(클래스 정의: 몸통 기울임만 tilt)."""
    j = judge(kps(nose_x=0.75), depth())
    assert j.posture_level == NORMAL and j.posture_kind is None


def test_caution_between_thresholds():
    j = judge(kps(), depth(recline=0.075))  # 0.75x
    assert j.posture_level == CAUTION


def test_posture_alert_needs_three_consecutive_then_cooldown():
    sm = AlertStateMachine()
    w = Judgement(WARNING, "slouch_forward", False)
    assert sm.update(0, w) == [] and sm.update(5, w) == []
    assert sm.update(10, w) == ["posture:slouch_forward"]
    for t in (15, 20, 25):
        assert sm.update(t, w) == []  # 쿨다운 중
    assert sm.update(305, w) == []  # 쿨다운(300초) 아직 안 지남
    assert sm.update(310, w) == ["posture:slouch_forward"]  # 계속 나쁜 자세면 쿨다운 끝나자마자 재알림


def test_streak_resets_on_normal_and_proximity_cooldown():
    sm = AlertStateMachine()
    w, n = Judgement(WARNING, "tilt_left", False), Judgement(NORMAL, None, False)
    sm.update(0, w); sm.update(5, w); sm.update(10, n)
    assert sm.update(15, w) == []
    p = Judgement(NORMAL, None, True)
    assert sm.update(20, p) == ["proximity"] and sm.update(30, p) == []
    assert sm.update(80, p) == ["proximity"]


def test_proximity_estimator_recovers_depth_from_shoulder_width():
    from src.logic.decision import ProximityEstimator
    est = ProximityEstimator(min_samples=3)
    assert est.estimate_depth(0.3) is None  # 아직 보정 전
    for z in (0.6, 0.7, 0.8, 0.65):
        est.update(0.12 / z, z)  # w = 0.12 / Z
    assert est.calibrated
    assert abs(est.estimate_depth(0.12 / 0.3) - 0.3) < 0.02


def test_fallback_depth_triggers_proximity_only_when_depth_missing():
    near_missing = judge(kps(), {"head_depth_m": None, "chest_depth_m": None}, fallback_depth_m=0.30)
    assert near_missing.proximity
    valid_far = judge(kps(), depth(), fallback_depth_m=0.30)  # depth가 유효하면 depth 우선
    assert not valid_far.proximity


def test_back_threshold_matches_field_value():
    j = judge(kps(), depth(head=0.8, chest=0.8, recline=-0.15))
    assert j.posture_level == WARNING and j.posture_kind == "slouch_back"


def test_near_invalid_face_counts_as_forward_and_proximity():
    j = judge(kps(), {"head_depth_m": None, "chest_depth_m": None, "hip_depth_m": None,
                      "torso_recline_offset_m": None, "near_invalid": True})
    assert j.proximity and j.posture_level == WARNING and j.posture_kind == "slouch_forward"


def test_proximity_without_hip_depth_is_forward_lean():
    j = judge(kps(), {"head_depth_m": 0.35, "chest_depth_m": 0.5, "hip_depth_m": None,
                      "torso_recline_offset_m": None})
    assert j.proximity and j.posture_kind == "slouch_forward" and j.posture_level == WARNING


def test_back_lean_detected_from_neck_offset_without_hip():
    """10/6 로그 값: 정상 nf 0.153 → 정상, 기댐 nf 0.062 → 경고, 허리 depth 없어도."""
    base = {"head_depth_m": 0.74, "chest_depth_m": 0.80, "hip_depth_m": None, "torso_recline_offset_m": None}
    normal = judge(kps(), {**base, "neck_forward_offset_m": 0.153})
    back = judge(kps(), {**base, "neck_forward_offset_m": 0.062})
    assert normal.posture_kind is None
    assert back.posture_level == WARNING and back.posture_kind == "slouch_back"


def test_logged_recline_values():
    assert judge(kps(), depth(recline=0.054)).posture_kind is None
    assert judge(kps(), depth(recline=-0.117)).posture_kind == "slouch_back"


def test_forward_lean_not_mistaken_for_back_by_neck_offset():
    """10/6 forward 로그 값(머리 0.35m, 가슴 0.41m, nf 0.062)은 기댐이 아니라 앞숙임."""
    j = judge(kps(), {"head_depth_m": 0.348, "chest_depth_m": 0.412, "hip_depth_m": None,
                      "torso_recline_offset_m": None, "neck_forward_offset_m": 0.062})
    assert j.proximity and j.posture_kind == "slouch_forward"
    mild = judge(kps(), {"head_depth_m": 0.45, "chest_depth_m": 0.50, "hip_depth_m": None,
                         "torso_recline_offset_m": None, "neck_forward_offset_m": 0.05})
    assert mild.posture_kind != "slouch_back"


def test_issues_lists_all_compound_problems():
    # 뒤로 기댐(recline -0.16) + 옆 기울임(shoulder_dx 큼)이 동시에 -> 두 문제 모두 issues에 남는다
    j = judge(kps(shoulder_dx=0.12), depth(recline=-0.16))
    kinds = [k for k, _ in j.issues]
    assert "slouch_back" in kinds and any(k.startswith("tilt") for k in kinds)
    assert j.posture_kind == kinds[0]
    assert judge(kps(), depth()).issues == []
