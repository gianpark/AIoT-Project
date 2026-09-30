"""
src/pose/movenet_keypoints.py의 KeypointOcclusionSmoother 단위 테스트.

카메라/모델 없이도 검증 가능하도록, MoveNet이 낸 것처럼 생긴 가짜 Keypoint 시퀀스로
"팔이 엉덩이를 가리는" 상황과 "머리가 순간적으로 어깨/엉덩이를 가리는" 상황을 흉내 낸다.
"""

from src.pose.movenet_keypoints import KEYPOINT_NAMES, Keypoint, KeypointOcclusionSmoother


def make_frame(overrides: dict[int, tuple[float, float, float]]) -> list[Keypoint]:
    """17개 keypoint를 기본값(화면 중앙, confidence 1.0)으로 채우고, overrides로 특정
    인덱스만 (x, y, score)를 덮어쓴다."""
    base = [Keypoint(name=KEYPOINT_NAMES[i], x=0.5, y=0.5, score=1.0) for i in range(17)]
    for idx, (x, y, score) in overrides.items():
        base[idx] = Keypoint(name=KEYPOINT_NAMES[idx], x=x, y=y, score=score)
    return base


def test_hip_tracks_shoulder_offset_when_occluded_for_a_long_time():
    """어깨는 계속 잘 보이는데 엉덩이가 팔에 의해 오래(hold_frames를 훌쩍 넘게) 가려지는
    상황 — 어깨 기준 오프셋을 학습해뒀으니, 그 이후 어깨가 옆으로 이동해도(=몸이 움직여도)
    엉덩이 추정 위치가 얼어붙지 않고 같이 따라가야 한다."""
    smoother = KeypointOcclusionSmoother(hold_frames=5, confidence_threshold=0.3)

    # 1) 여러 프레임 동안 양쪽 다 잘 보여서, "어깨 기준 엉덩이 오프셋"(여기서는 dx=0.05,
    #    dy=0.20)이 학습되게 한다. left_shoulder=(0.40,0.30), left_hip=(0.45,0.50) 고정.
    for _ in range(5):
        frame = make_frame({5: (0.40, 0.30, 0.9), 11: (0.45, 0.50, 0.9)})
        smoother.smooth(frame)

    # 2) 이제 엉덩이가 오래(hold_frames=5보다 훨씬 많은 20프레임) 가려진다고 가정하고,
    #    그동안 몸 전체가 오른쪽으로 이동해 어깨 x좌표가 0.40 -> 0.55로 바뀐다.
    result = None
    for step in range(20):
        shoulder_x = 0.40 + 0.15 * (step / 19)  # 0.40 -> 0.55로 서서히 이동
        frame = make_frame({
            5: (shoulder_x, 0.30, 0.9),   # 어깨는 계속 잘 보임
            11: (0.5, 0.5, 0.05),          # 엉덩이는 계속 저신뢰(가려짐)
        })
        result = smoother.smooth(frame)

    # hold_frames(5)를 훨씬 넘겼지만, 단순 '위치 얼리기'가 아니라 어깨를 따라간 추정이므로
    # 최종 어깨 위치(0.55) + 학습된 오프셋(~0.05) 근처(~0.60)에 있어야 한다 — 시작 시점
    # 위치(0.40+0.05=0.45)에 그대로 얼어붙어 있으면 안 된다.
    assert result is not None
    estimated_hip_x = result[11].x
    assert estimated_hip_x > 0.50  # 얼어붙은 시작 위치(0.45)보다는 분명히 커야 함
    assert estimated_hip_x < 0.62  # 최종 기대값(0.60) 근처
    # score는 원본 저신뢰 값을 그대로 유지해야 한다 (위치만 보정, confidence는 안 속임)
    assert result[11].score == 0.05


def test_hip_falls_back_to_raw_low_confidence_when_no_anchor_and_no_history():
    """엉덩이도 어깨도 둘 다 한 번도 confidence 충분했던 적이 없으면(추정 근거가 전혀
    없으면), 그냥 원본 저신뢰 값을 그대로 내보내야 한다 — 없는 정보를 지어내면 안 된다."""
    smoother = KeypointOcclusionSmoother(hold_frames=5, confidence_threshold=0.3)

    frame = make_frame({5: (0.4, 0.3, 0.1), 11: (0.4, 0.5, 0.05)})  # 둘 다 저신뢰
    result = smoother.smooth(frame)

    assert result[11].x == 0.4 and result[11].y == 0.5  # 보정 없이 원본 좌표 그대로
    assert result[11].score == 0.05


def test_nose_holds_last_good_position_briefly_then_releases():
    """코(머리)처럼 기준점이 없는 keypoint는 기존 방식대로 hold_frames 동안만 마지막
    위치를 유지하고, 그 이후엔 원본(저신뢰) 값으로 돌아가야 한다."""
    smoother = KeypointOcclusionSmoother(hold_frames=3, confidence_threshold=0.3)

    # 확실한 위치 한 번 학습
    smoother.smooth(make_frame({0: (0.3, 0.2, 0.9)}))

    # 가려짐 1~3프레임: hold_frames(3) 이내라 마지막 위치(0.3, 0.2)를 유지해야 한다
    for _ in range(3):
        result = smoother.smooth(make_frame({0: (0.9, 0.9, 0.05)}))
        assert result[0].x == 0.3 and result[0].y == 0.2

    # 4번째부터는 hold_frames를 넘겨서 원본(저신뢰) 값으로 돌아가야 한다
    result = smoother.smooth(make_frame({0: (0.9, 0.9, 0.05)}))
    assert result[0].x == 0.9 and result[0].y == 0.9
