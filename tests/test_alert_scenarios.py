"""T3: 알림 쿨다운 오프라인 시나리오 검증 (카메라 없이 합성 Judgement로 AlertStateMachine을 돌린다)."""
from src.logic.decision import CAUTION, NORMAL, WARNING, AlertStateMachine, Judgement

STEP = 5.0  # --sample-interval 기본값(초)


def J(level=NORMAL, kind=None, prox=False):
    return Judgement(level, kind, prox)


def run(seq, step=STEP):
    """seq: Judgement 리스트(5초 간격). 반환: [(t, alert), ...]"""
    sm = AlertStateMachine()
    out = []
    for i, j in enumerate(seq):
        for a in sm.update(i * step, j):
            out.append((i * step, a))
    return out


def times(events, prefix):
    return [t for t, a in events if a.startswith(prefix)]


def test_proximity_alert_immediate_then_60s_cooldown():
    ev = run([J(prox=True)] * 37)  # 3분(180s) 이상 지속
    ts = times(ev, "proximity")
    assert ts[0] == 0.0  # 감지 즉시
    assert all(b - a >= 60.0 for a, b in zip(ts, ts[1:]))
    assert ts == [0.0, 60.0, 120.0, 180.0]


def test_posture_alert_needs_three_consecutive_warnings():
    ev = run([J(WARNING, "neck_forward")] * 3)
    assert times(ev, "posture") == [10.0]  # t=0,5,10 → 10초(3번째 샘플)에 알림


def test_posture_two_warnings_then_break_resets_streak():
    seq = [J(WARNING, "neck_forward")] * 2 + [J()] + [J(WARNING, "neck_forward")] * 2
    assert times(run(seq), "posture") == []


def test_caution_does_not_count_toward_streak():
    seq = [J(WARNING, "slouch_back"), J(CAUTION, "slouch_back"), J(WARNING, "slouch_back")]
    assert times(run(seq), "posture") == []


def test_posture_cooldown_300s_over_7_minutes():
    ev = run([J(WARNING, "slouch_back")] * 85)  # 7분(420s)
    ts = times(ev, "posture")
    assert ts[0] == 10.0
    assert all(b - a >= 300.0 for a, b in zip(ts, ts[1:]))
    assert ts == [10.0, 310.0]


def test_posture_cooldown_blocks_even_after_streak_resets():
    seq = [J(WARNING, "chin_rest")] * 3 + [J()] + [J(WARNING, "chin_rest")] * 10
    ts = times(run(seq), "posture")
    assert ts == [10.0]  # 두 번째 연속 3회가 쿨다운(300s) 안이라 막힘


def test_posture_alert_after_cooldown_expires_when_warning_resumes():
    seq = [J(WARNING, "neck_forward")] * 3 + [J()] * 60 + [J(WARNING, "neck_forward")] * 3
    ts = times(run(seq), "posture")
    assert ts == [10.0, 325.0]  # 경고 재개(315~325s) 3번째 샘플에서 쿨다운(≥300s) 경과


def test_proximity_and_posture_cooldowns_are_independent():
    seq = [J(WARNING, "neck_forward", prox=True)] * 15  # 70초
    ev = run(seq)
    assert times(ev, "proximity") == [0.0, 60.0]
    assert times(ev, "posture") == [10.0]


def test_alert_gaps_never_below_cooldown_on_long_mixed_scenario():
    seq = ([J(prox=True)] * 20 + [J(WARNING, "slouch_forward")] * 80 + [J()] * 10
           + [J(WARNING, "tilt_left", prox=True)] * 100)
    ev = run(seq)
    prox, post = times(ev, "proximity"), times(ev, "posture")
    assert all(b - a >= 60.0 for a, b in zip(prox, prox[1:]))
    assert all(b - a >= 300.0 for a, b in zip(post, post[1:]))
