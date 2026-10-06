from src.logic.decision import (CAUTION, NORMAL, WARNING, AlertStateMachine, Judgement, judge)
from src.pose.movenet_keypoints import Keypoint


def kps(nose_x=0.5):
    base = {5: (0.42, 0.40), 6: (0.58, 0.40), 11: (0.44, 0.65), 12: (0.56, 0.65), 0: (nose_x, 0.28)}
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
    j = judge(kps(), depth(head=0.79, chest=0.82, recline=-0.16))
    assert j.posture_level == WARNING and j.posture_kind == "slouch_back"


def test_tilt_direction_uses_participant_side():
    # 이미지 왼쪽(x 작음) = 참가자 본인 기준 오른쪽
    left_img = judge(kps(nose_x=0.30), depth())
    right_img = judge(kps(nose_x=0.70), depth())
    assert left_img.posture_kind == "tilt_right" and right_img.posture_kind == "tilt_left"


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
