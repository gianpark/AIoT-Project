#!/usr/bin/env python3
"""
바른자세 - MoveNet 17키포인트 추출 모듈

용도
----
1) 웹캠 또는 RealSense D455의 color 스트림(`--realsense`)에서 MoveNet Lightning으로
   17개 keypoint를 실시간 추출하고 화면에 스켈레톤을 그려 보여준다. 실제 자세 데이터
   수집은 최종 제품과 화각·렌즈 특성을 맞추기 위해 `--realsense`로 진행한다.
2) 각 keypoint의 confidence를 CSV로 기록한다 — "상반신만 보이는 책상 구도에서
   어떤 keypoint를 신뢰할 수 있는가"를 실측으로 검증하기 위한 기초 데이터다.
3) `--realsense`일 때는 머리(코)/가슴(어깨 중점)/허리(엉덩이 중점) 세 지점의 depth(m)와
   그 차이값(neck_forward_offset_m, torso_recline_offset_m)도 화면에 같이 표시하고,
   `--log`와 함께 쓰면 CSV에도 기록한다(5주차 "판정 로직과 스테레오+포즈 파이프라인
   1차 통합" — src/features/posture_features.py의 compute_depth_features 참고).
4) 화면에 여러 사람이 잡혀도 컴퓨터 사용자 한 명만 인식하도록 자동 인식 기능을 지원한다.
   시작 직후(또는 'r' 키로 재인식 시) 몇 프레임 연속으로 확실하게 잡힌 사람을 "컴퓨터
   사용자"로 보고, 그 사람의 keypoint 위치(ROI)와 — `--realsense`일 때는 — 깊이 범위까지
   자동으로 고정한다(수동으로 사각형을 그릴 필요 없음). 그 뒤로는 범위 밖(다른 사람,
   배경)을 검게 지워 MoveNet이 애초에 한 사람만 보게 한다. 최초 인식 후에도 사용자가
   움직이면(의자를 당기거나 몸을 기울이는 등) 매 프레임 위치를 다시 추정해 영역을 서서히
   따라가게 한다(`--no-track-subject`로 끄면 최초 위치에 고정). 자동 인식 자체를 끄려면
   `--no-auto-calibrate`(이 경우 `--subject-min-depth`/`--subject-max-depth`로 지정한
   거리 범위만 처음부터 고정 적용, `--no-subject-isolation`으로 그마저도 끌 수 있음).

특징 추출(정규화·각도 계산)은 src/features/posture_features.py로 분리되어 있다
— 이 모듈은 "포즈 추정" 담당, 그쪽은 "특징 엔지니어링" 담당으로 역할을 나눴다.

모델 준비
--------
이 스크립트는 TFLite 형식의 MoveNet SinglePose Lightning 모델 파일이 필요하다.
TensorFlow Hub의 모델 배포처가 Kaggle Models로 통합되어, 지금은 아래 Kaggle
페이지가 공식 다운로드 경로다:

    # https://www.kaggle.com/models/google/movenet/tfLite/singlepose-lightning-tflite-int8
    # 위 페이지에서 "Download" 받은 .tflite 파일을 models/ 아래에 둔다.
    # (float16 버전을 쓰고 싶으면 같은 페이지에서 singlepose-lightning-tflite-fp16 선택)
    #
    # 입출력 스펙(둘 다 입력은 uint8, 192x192, 출력은 [1,1,17,3]=y,x,score)은
    # TensorFlow 공식 튜토리얼에 정리되어 있다:
    # https://www.tensorflow.org/hub/tutorials/movenet

의존성
------
    pip install -r requirements.txt

실행 예시 (프로젝트 루트에서, 패키지로 실행)
--------
    # 웹캠으로 실시간 확인 (q로 종료)
    python -m src.pose.movenet_keypoints --model models/movenet_lightning_int8.tflite

    # RealSense D455의 color 스트림으로 실시간 확인 (실제 데이터 수집은 이 방식으로)
    python -m src.pose.movenet_keypoints --model models/movenet_lightning_int8.tflite --realsense

    # confidence 로그를 CSV로 저장 (카메라 프레이밍/keypoint 신뢰도 검증용)
    python -m src.pose.movenet_keypoints --model models/movenet_lightning_int8.tflite \
        --log keypoint_confidence_log.csv --camera 0

    # 데이터 수집용 촬영 (data_collection_protocol.md 4번 절차) — 's' 누르면 그 순간
    # 프레임을 data/raw/{참가자ID}/{참가자ID}_{클래스}_{일련번호}.jpg 로 저장한다.
    # 스크립트를 한 번만 켜두고, 원하는 만큼 찍은 뒤 'n' 키를 누르면 바로 다음 클래스로
    # 넘어간다 (normal → slouch_forward → slouch_back → tilt_left → tilt_right 순서).
    # 재시작 없이 5클래스 전부 촬영 가능.
    python -m src.pose.movenet_keypoints --model models/movenet_lightning_int8.tflite \
        --realsense --participant p01
"""

from __future__ import annotations

import argparse
import csv
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "opencv-python이 필요합니다: pip install opencv-python"
    ) from exc

from src.features.posture_features import compute_depth_features, compute_posture_features
from src.logic.decision import CAUTION, NORMAL, WARNING, AlertStateMachine, judge

DEPTH_FEATURE_KEYS = [
    "head_depth_m", "chest_depth_m", "hip_depth_m",
    "neck_forward_offset_m", "torso_recline_offset_m",
]


# ---------------------------------------------------------------------------
# MoveNet(COCO) 17키포인트 순서 — 모델 출력이 항상 이 순서로 나온다.
# 어떤 keypoint를 실제로 쓸지(상반신/전신)는 src/features/posture_features.py의
# UPPER_BODY_IDX/LOWER_BODY_IDX 참고.
# ---------------------------------------------------------------------------
VALID_LABELS = ["normal", "slouch_forward", "slouch_back", "tilt_left", "tilt_right"]

# data_collection_protocol.md 1번 표의 "촬영 시 지시 문구" — 촬영자(본인)가 카메라 앞에서
# 화면만 보고도 지금 무슨 자세를 취해야 하는지 바로 알 수 있게 그대로 가져다 쓴다.
LABEL_INSTRUCTIONS = {
    "normal": "평소 화면 볼 때처럼 편하게 앉아주세요",
    "slouch_forward": "화면에 좀 더 집중하듯이 고개를 앞으로 내밀어주세요",
    "slouch_back": "의자에 기대서 늘어지듯 앉아주세요",
    "tilt_left": "왼쪽 팔걸이 쪽으로 몸을 기울여주세요",
    "tilt_right": "오른쪽으로 몸을 기울여주세요",
}

KEYPOINT_NAMES = [
    "nose",
    "left_eye", "right_eye",
    "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]

# 스켈레톤을 그릴 때 이을 keypoint 쌍
SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),          # 얼굴
    (0, 5), (0, 6), (5, 6),                  # 목~어깨
    (5, 7), (7, 9), (6, 8), (8, 10),         # 팔
    (5, 11), (6, 12), (11, 12),              # 몸통
    (11, 13), (13, 15), (12, 14), (14, 16),  # 다리 (책상 구도에서는 대부분 미검출)
]


@dataclass
class Keypoint:
    name: str
    y: float  # 0~1 정규화 좌표 (프레임 세로 기준)
    x: float  # 0~1 정규화 좌표 (프레임 가로 기준)
    score: float  # confidence, 0~1


class MoveNetExtractor:
    """TFLite MoveNet SinglePose 모델을 감싸는 얇은 래퍼."""

    def __init__(self, model_path: str):
        self._interpreter = self._load_interpreter(model_path)
        self._interpreter.allocate_tensors()
        self._input_details = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()

        in_shape = self._input_details[0]["shape"]  # [1, H, W, 3]
        self.input_size = int(in_shape[1])
        self.input_dtype = self._input_details[0]["dtype"]

    @staticmethod
    def _load_interpreter(model_path: str):
        """tflite_runtime을 우선 쓰고, 없으면 tensorflow.lite로 폴백한다."""
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            try:
                from tensorflow.lite.python.interpreter import Interpreter
            except ImportError as exc:  # pragma: no cover
                raise SystemExit(
                    "tflite_runtime 또는 tensorflow 중 하나는 설치되어 있어야 합니다.\n"
                    "  pip install tflite-runtime\n"
                    "  또는 pip install tensorflow"
                ) from exc
        return Interpreter(model_path=model_path)

    def infer(self, frame_bgr: np.ndarray) -> list[Keypoint]:
        """BGR 프레임 한 장에서 17개 keypoint를 추출한다."""
        img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.input_size, self.input_size))

        if self.input_dtype == np.uint8:
            input_data = img.astype(np.uint8)
        else:
            # float32 모델은 0~1로 정규화된 입력을 기대한다.
            input_data = img.astype(np.float32) / 255.0

        input_data = np.expand_dims(input_data, axis=0)

        self._interpreter.set_tensor(self._input_details[0]["index"], input_data)
        self._interpreter.invoke()

        raw = self._interpreter.get_tensor(self._output_details[0]["index"])
        # raw shape: [1, 1, 17, 3] -> (y, x, score)
        raw = raw.reshape(17, 3)

        return [
            Keypoint(name=KEYPOINT_NAMES[i], y=float(raw[i, 0]), x=float(raw[i, 1]), score=float(raw[i, 2]))
            for i in range(17)
        ]


# 우리 특징 계산에 실제로 쓰이는 keypoint만 스무딩 대상으로 삼는다(posture_features.py의
# nose/shoulders/hips) — 팔·눈·무릎 등은 애초에 특징에 안 쓰여서 굳이 붙잡아둘 필요가 없다.
SMOOTHED_KEYPOINT_IDX = [0, 5, 6, 11, 12]  # nose, left_shoulder, right_shoulder, left_hip, right_hip


class KeypointOcclusionSmoother:
    """책상 구도에서 두 가지 자기 가림(self-occlusion) 패턴을 보정한다(5주차 실측으로 발견):

    1) 머리를 화면 쪽으로 내밀 때 머리가 카메라 시야상 어깨·엉덩이보다 앞쪽에 오면서
       그 지점을 순간적으로 가리는 경우 — 대부분 아주 짧게 지나간다.
    2) 턱을 괴는 등 팔이 엉덩이 앞을 오래(수 초 이상) 가리는 경우 — 1)과 달리 짧게
       안 지나가므로, "마지막 위치를 잠깐 유지"만으로는 hold_frames를 넘기면 다시
       저신뢰로 돌아가 버린다.

    대응 전략은 두 단계다.
    - **엉덩이(hip)**: 양쪽 어깨가 보이는 동안 "어깨 대비 엉덩이가 어디쯤 있었는지"
      오프셋을 지수이동평균(EMA)으로 계속 학습해둔다. 엉덩이 confidence가 떨어져도
      같은 쪽 어깨가 여전히 보이면, "지금 보이는 어깨 위치 + 학습해둔 오프셋"으로
      엉덩이 위치를 추정한다 — 가려진 동안 몸이 움직여도 어깨를 따라 같이 움직이므로
      단순히 위치를 얼려두는 것보다 오래 가려져도 안정적이다.
    - **그 외(코·어깨)**: 기준으로 삼을 다른 keypoint가 마땅치 않으므로, 기존처럼
      "마지막으로 확실했던 위치"를 최대 hold_frames 프레임만 유지한다.

    두 방법 모두 계속 가려져서 추정 근거(어깨든, 과거 위치든)조차 없으면 결국 원본
    저신뢰 값을 그대로 내보낸다 — 없는 정보를 영원히 지어내지 않기 위함. score는
    원본 값을 그대로 두므로(위치만 보정), confidence 기반 경고·로그는 정상 동작한다.
    """

    # hip index -> 같은 쪽 shoulder index (오프셋 추정의 기준점)
    _HIP_ANCHOR = {11: 5, 12: 6}  # left_hip <- left_shoulder, right_hip <- right_shoulder

    def __init__(self, hold_frames: int = 10, confidence_threshold: float = 0.3, offset_alpha: float = 0.15):
        self.hold_frames = hold_frames
        self.confidence_threshold = confidence_threshold
        self.offset_alpha = offset_alpha  # 어깨-엉덩이 오프셋 EMA 갱신 속도 (0~1, 클수록 최근 값에 민감)
        self._last_good: dict[int, Keypoint] = {}
        self._held_for: dict[int, int] = {}
        self._hip_offset: dict[int, tuple[float, float]] = {}  # hip idx -> (dx, dy) = hip - anchor_shoulder (EMA)

    def smooth(self, keypoints: list[Keypoint]) -> list[Keypoint]:
        # 1) 양쪽 다 잘 보이는 동안 어깨 기준 엉덩이 오프셋을 계속 갱신해둔다.
        for hip_idx, sh_idx in self._HIP_ANCHOR.items():
            hip_kp, sh_kp = keypoints[hip_idx], keypoints[sh_idx]
            if hip_kp.score >= self.confidence_threshold and sh_kp.score >= self.confidence_threshold:
                dx, dy = hip_kp.x - sh_kp.x, hip_kp.y - sh_kp.y
                prev = self._hip_offset.get(hip_idx)
                self._hip_offset[hip_idx] = (
                    (dx, dy) if prev is None else
                    (prev[0] + self.offset_alpha * (dx - prev[0]), prev[1] + self.offset_alpha * (dy - prev[1]))
                )

        result = list(keypoints)
        for idx in SMOOTHED_KEYPOINT_IDX:
            kp = keypoints[idx]
            if kp.score >= self.confidence_threshold:
                self._last_good[idx] = kp
                self._held_for[idx] = 0
                continue

            anchor_idx = self._HIP_ANCHOR.get(idx)
            if anchor_idx is not None and idx in self._hip_offset and keypoints[anchor_idx].score >= self.confidence_threshold:
                # 엉덩이인데 가려졌고, 기준 어깨는 지금 보임 -> 어깨 기준으로 추정
                sh = keypoints[anchor_idx]
                dx, dy = self._hip_offset[idx]
                self._held_for[idx] = 0
                result[idx] = Keypoint(name=kp.name, x=sh.x + dx, y=sh.y + dy, score=kp.score)
                continue

            held_so_far = self._held_for.get(idx, 0)
            if idx in self._last_good and held_so_far < self.hold_frames:
                anchor = self._last_good[idx]
                self._held_for[idx] = held_so_far + 1
                result[idx] = Keypoint(name=kp.name, x=anchor.x, y=anchor.y, score=kp.score)
            # else: 추정 근거(어깨든 과거 위치든)가 없으면 원본(저신뢰) 값 그대로 둔다

        return result


# OpenCV 기본 폰트(Hershey)는 한글을 그리지 못한다(깨지거나 아예 안 보임) — 촬영 중
# "지금 무슨 자세를 취해야 하는지" 화면에서 바로 읽을 수 있게, Pillow + 시스템 한글
# 폰트(Windows 기본 맑은 고딕)로 따로 그린다. Pillow나 한글 폰트가 없으면 조용히
# 건너뛴다(터미널에는 항상 출력되므로 기능 자체가 막히지는 않는다).
_KOREAN_FONT_CANDIDATES = [
    "C:/Windows/Fonts/malgun.ttf",           # Windows 기본 내장 (맑은 고딕)
    "C:/Windows/Fonts/malgunbd.ttf",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",  # macOS
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",  # Linux (설치돼 있는 경우)
]


def _load_korean_font(size: int):
    try:
        from PIL import ImageFont
    except ImportError:
        return None
    for path in _KOREAN_FONT_CANDIDATES:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return None


def draw_text_kr(frame_bgr: np.ndarray, text: str, org: tuple[int, int], font, color_bgr=(255, 255, 255)) -> np.ndarray:
    """한글이 섞인 텍스트를 프레임에 그린다. font는 _load_korean_font()로 미리 로드해둔
    것을 넘긴다 — None이면(Pillow/한글 폰트 없음) 아무것도 안 그리고 원본을 그대로 반환."""
    if font is None:
        return frame_bgr
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return frame_bgr

    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
    x, y = org
    for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):  # 가독성용 검은 외곽선
        draw.text((x + dx, y + dy), text, font=font, fill=(0, 0, 0))
    draw.text((x, y), text, font=font, fill=color_rgb)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def draw_skeleton(frame: np.ndarray, keypoints: list[Keypoint], threshold: float, show_labels: bool = False) -> np.ndarray:
    h, w = frame.shape[:2]
    for kp in keypoints:
        if kp.score < threshold:
            continue
        cx, cy = int(kp.x * w), int(kp.y * h)
        cv2.circle(frame, (cx, cy), 4, (0, 255, 170), -1)
        if show_labels:
            # 어떤 keypoint가 실제로 어디서 잡히는지 눈으로 바로 확인하기 위한 디버그 라벨
            # (책상 구도에서 keypoint 신뢰도 검증용 — 스크립트 상단 docstring 참고)
            cv2.putText(frame, kp.name, (cx + 6, cy - 6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (0, 255, 170), 1, cv2.LINE_AA)

    for i, j in SKELETON_EDGES:
        a, b = keypoints[i], keypoints[j]
        if a.score < threshold or b.score < threshold:
            continue
        pa = (int(a.x * w), int(a.y * h))
        pb = (int(b.x * w), int(b.y * h))
        cv2.line(frame, pa, pb, (255, 200, 0), 2)

    return frame


class _WebcamSource:
    """일반 웹캠(cv2.VideoCapture)에서 BGR 프레임을 읽는 소스."""

    def __init__(self, camera_index: int):
        self._cap = cv2.VideoCapture(camera_index)
        if not self._cap.isOpened():
            raise SystemExit(f"카메라(index={camera_index})를 열 수 없습니다.")

    def read(self):
        ok, frame = self._cap.read()
        return frame if ok else None

    def get_depth_lookup(self, frame_shape):
        return None  # 일반 웹캠은 depth가 없음

    def get_foreground_mask(self, min_depth_m: float, max_depth_m: float):
        return None  # 일반 웹캠은 depth가 없어 전경 분리 불가

    def release(self) -> None:
        self._cap.release()


class _RealSenseSource:
    """RealSense D455의 color+depth 스트림에서 프레임을 읽는 소스.

    5주차 "판정 로직과 스테레오+포즈 파이프라인 1차 통합"부터는 color 프레임에서 뽑은
    keypoint 좌표를 그대로 같은 순간의 depth 프레임에 대입해, 머리/가슴/허리 각각의
    실측 거리를 읽을 수 있게 한다(get_depth_lookup). posture_features.py는 카메라에
    의존하지 않는 순수 모듈이라, depth 조회는 여기서 클로저로 만들어 주입한다.
    """

    def __init__(self, fps: int):
        from src.capture.realsense_capture import (  # 여기서만 필요해 지연 import
            MAX_VALID_DEPTH_M,
            MIN_VALID_DEPTH_M,
            RealSenseCamera,
        )

        self._RealSenseCamera = RealSenseCamera
        self._min_valid = MIN_VALID_DEPTH_M
        self._max_valid = MAX_VALID_DEPTH_M
        self._cam = RealSenseCamera(fps=fps).start()
        self._last_depth_frame = None

    def read(self):
        result = self._cam.read()
        if result is None:
            return None
        color_image, depth_frame, _timestamp_ms = result
        self._last_depth_frame = depth_frame
        return color_image

    def get_depth_lookup(self, frame_shape):
        """정규화 좌표(x_norm, y_norm) -> depth(m) 콜백을 반환 (마지막 read() 프레임 기준)."""
        depth_frame = self._last_depth_frame
        if depth_frame is None:
            return None
        h, w = frame_shape[:2]
        get_distance_m = self._RealSenseCamera.get_distance_m
        min_valid, max_valid = self._min_valid, self._max_valid

        def _lookup(x_norm: float, y_norm: float):
            px = int(np.clip(x_norm, 0.0, 1.0) * (w - 1))
            py = int(np.clip(y_norm, 0.0, 1.0) * (h - 1))
            d = get_distance_m(depth_frame, px, py)
            return d if (min_valid <= d <= max_valid) else None

        return _lookup

    def get_foreground_mask(self, min_depth_m: float, max_depth_m: float):
        """[min_depth_m, max_depth_m] 범위 안에 있는 픽셀만 True인 마스크를 반환한다
        (마지막 read() 프레임 기준). MoveNet은 SinglePose 모델이라 사람이 둘 이상
        잡히면 어느 쪽을 추적할지 보장이 안 되는데(5주차 실측으로 발견), 촬영 대상은
        항상 카메라에서 가까운 거리(책상 앞)에 있다는 전제로, 그보다 먼 범위(다른 사람,
        배경)를 아예 검게 지워 MoveNet이 애초에 한 사람만 보게 만든다. 범위 밖(0 포함,
        무효값)은 전부 제외된다.
        """
        depth_frame = self._last_depth_frame
        if depth_frame is None:
            return None
        depth_m = self._cam.get_depth_image_m(depth_frame)
        return (depth_m >= min_depth_m) & (depth_m <= max_depth_m)

    def release(self) -> None:
        self._cam.stop()


# 자동 인식(캘리브레이션)에 쓸 keypoint: 코, 양 어깨, 양 엉덩이 — occlusion smoothing과
# 동일한 "상반신 핵심 지점"을 기준으로 삼는다.
_CALIBRATION_KEYPOINT_IDX = [0, 5, 6, 11, 12]
_LEFT_SHOULDER_IDX, _RIGHT_SHOULDER_IDX = 5, 6
# 어깨 너비 대비 가로 폭 배수 — 팔까지 여유 있게 포함되도록 넉넉하게 잡는다.
_SHOULDER_WIDTH_MARGIN = 2.2


def _estimate_subject_bounds(keypoints: list[Keypoint], frame_shape, depth_lookup, confidence_threshold: float):
    """현재 프레임의 keypoint로 '사용자가 있는 범위'(ROI·depth)를 추정한다.

    수동으로 사각형을 그리지 않아도, 화면에 확실하게 잡힌 사람의 코/어깨/엉덩이 위치와
    깊이로 그 사람 주변의 대략적인 범위를 계산해 자동 인식에 쓴다. 신뢰할 수 있는
    keypoint가 2개 미만이면(아직 사람이 제대로 안 잡혔거나 가림이 심함) None을 반환해
    호출부가 이번 프레임은 건너뛰게 한다.

    가로 폭은 양 어깨가 둘 다 잡히면 어깨 너비를 기준으로 정한다 — 순간적인 팔 벌림
    등에 덜 민감하고, 몸통의 실제 '폭'을 더 안정적으로 대표한다. 어깨가 둘 다 안
    잡히면(가려짐 등) 기존처럼 keypoint들의 bbox 기반으로 대체한다.
    """
    confident = [keypoints[i] for i in _CALIBRATION_KEYPOINT_IDX if keypoints[i].score >= confidence_threshold]
    if len(confident) < 2:
        return None

    xs = [kp.x for kp in confident]
    ys = [kp.y for kp in confident]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    h_norm = max(y_max - y_min, 0.05)

    left_sh, right_sh = keypoints[_LEFT_SHOULDER_IDX], keypoints[_RIGHT_SHOULDER_IDX]
    if left_sh.score >= confidence_threshold and right_sh.score >= confidence_threshold:
        shoulder_center_x = (left_sh.x + right_sh.x) / 2.0
        shoulder_width = max(abs(right_sh.x - left_sh.x), 0.05)
        half_w = shoulder_width * _SHOULDER_WIDTH_MARGIN / 2.0
        x_min = shoulder_center_x - half_w
        x_max = shoulder_center_x + half_w
    else:
        w_norm = max(x_max - x_min, 0.05)
        x_min -= w_norm * 0.9
        x_max += w_norm * 0.9
    x_min = max(0.0, x_min)
    x_max = min(1.0, x_max)

    # 코~엉덩이까지만 잡히므로, 머리 위·하반신까지 넉넉히 포함되게 세로로 여유를 둔다. 또한
    # calibration이 하필 구부정하거나 기울어진 자세일 때 이뤄져도 다른 자세(정자세 등)로
    # 돌아왔을 때 몸이 잘려 나가지 않도록, 한 자세의 순간 bbox보다 더 넉넉하게 잡는다.
    y_min = max(0.0, y_min - h_norm * 0.7)
    y_max = min(1.0, y_max + h_norm * 1.8)

    h_px, w_px = frame_shape[:2]
    x_px = int(x_min * (w_px - 1))
    y_px = int(y_min * (h_px - 1))
    w_box = max(1, int((x_max - x_min) * (w_px - 1)))
    h_box = max(1, int((y_max - y_min) * (h_px - 1)))

    depth_range = None
    if depth_lookup is not None:
        depths = []
        for kp in confident:
            d = depth_lookup(kp.x, kp.y)
            if d is not None:
                depths.append(d)
        if depths:
            depth_range = (max(0.05, min(depths) - 0.35), max(depths) + 0.35)

    return {"roi_rect": (x_px, y_px, w_box, h_box), "depth_range": depth_range}


class _RoiKalmanTracker:
    """관심영역(ROI)을 추적한다 — 위치는 Kalman 필터, 크기는 느린 EMA로 보정한다.

    기존 EMA(지수이동평균) 방식은 그 순간의 추정치로 서서히 수렴할 뿐이라, 사람이
    잠깐 가려지면 그 자리에 멈춰버린다. 위치는 Kalman 필터로 속도까지 같이 추정해서,
    keypoint를 놓친 프레임에도 그동안의 이동 방향·속도로 계속 예측하며 따라가다가,
    다시 측정값이 들어오면 그걸로 보정한다 — "target tracking"에서 흔히 쓰는 방식.

    가로 폭(w)은 어깨 너비 기준으로 계산된 측정값을 따라 느린 EMA로 서서히 맞춰간다 —
    사용자가 카메라에 가까워지거나 멀어지면 어깨가 화면에서 더 크거나 작게 보이므로
    자연스럽게 같이 커지거나 작아진다. 세로 길이(h)는 요청에 따라 최초 인식 때 값으로
    완전히 고정하고 이후 다시 바꾸지 않는다.
    """

    def __init__(self, size_adapt_alpha: float = 0.08):
        self._kf = cv2.KalmanFilter(4, 2)  # 상태: [cx, cy, vcx, vcy], 측정: [cx, cy]
        self._kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)
        self._kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32)
        self._kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
        self._kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 5e-1
        self._kf.errorCovPost = np.eye(4, dtype=np.float32)
        self._size_adapt_alpha = size_adapt_alpha
        self._w = 1.0
        self._h = 1.0

    def init(self, roi_rect) -> None:
        x, y, w, h = roi_rect
        self._w, self._h = float(w), float(h)
        cx, cy = x + w / 2.0, y + h / 2.0
        state = np.array([[cx], [cy], [0.0], [0.0]], dtype=np.float32)
        self._kf.statePost = state
        self._kf.statePre = state.copy()

    def predict(self):
        """측정값 없이 속도만으로 다음 위치를 예측한다 (가림 등으로 못 잡았을 때 사용).
        크기는 측정값이 없으므로 직전 크기를 그대로 유지한다."""
        state = self._kf.predict()
        return self._state_to_rect(state)

    def correct(self, roi_rect):
        """새 측정값(위치+가로폭)으로 보정한다. 위치는 Kalman, 가로폭은 느린 EMA, 세로
        길이는 최초 인식 값 그대로 고정."""
        x, y, w, h = roi_rect
        cx, cy = x + w / 2.0, y + h / 2.0
        measurement = np.array([[cx], [cy]], dtype=np.float32)
        state = self._kf.correct(measurement)
        # 사람이 가까워지거나(박스가 커져야 함) 멀어지면(작아져야 함) 매 프레임 바로
        # 반영하지 않고 천천히 따라가게 해서, 한두 프레임의 잘못된 측정에 박스가
        # 출렁이지 않게 한다. 세로 길이는 건드리지 않는다(요청에 따라 고정).
        self._w += self._size_adapt_alpha * (w - self._w)
        return self._state_to_rect(state)

    def _state_to_rect(self, state):
        cx, cy = float(state[0, 0]), float(state[1, 0])
        return (int(cx - self._w / 2.0), int(cy - self._h / 2.0), int(self._w), int(self._h))


def _clamp_rect_to_frame(rect, frame_shape):
    """ROI 사각형이 화면(프레임) 범위를 벗어나지 않게 자른다 — 추적 중 사람이 화면 가장자리나
    밖으로 나가 예측치가 범위를 벗어나도, 마스킹에 쓸 때 음수 인덱스 등으로 깨지지 않게 한다.
    """
    x, y, w, h = rect
    h_px, w_px = frame_shape[:2]
    w = max(1, min(w, w_px))
    h = max(1, min(h, h_px))
    x = max(0, min(x, w_px - w))
    y = max(0, min(y, h_px - h))
    return (x, y, w, h)


def _next_serial(save_dir: Path, participant: str, label: str) -> int:
    """이미 저장된 {참가자}_{라벨}_NNN.jpg 파일들을 훑어 다음 일련번호를 정한다."""
    prefix = f"{participant}_{label}_"
    existing = []
    for p in save_dir.glob(f"{prefix}*.jpg"):
        stem = p.stem[len(prefix):]
        if stem.isdigit():
            existing.append(int(stem))
    return (max(existing) + 1) if existing else 1


def main():
    parser = argparse.ArgumentParser(description="MoveNet 17키포인트 실시간 추출/기록")
    parser.add_argument("--model", required=True, help="MoveNet Lightning .tflite 모델 경로")
    parser.add_argument("--camera", type=int, default=0, help="cv2.VideoCapture 인덱스 (기본 0, --realsense와 함께 쓰지 않음)")
    parser.add_argument("--realsense", action="store_true",
                         help="일반 웹캠 대신 RealSense D455의 color 스트림을 사용 (pyrealsense2 필요)")
    parser.add_argument("--fps", type=int, default=30, help="--realsense 사용 시 요청 fps")
    parser.add_argument("--threshold", type=float, default=0.3, help="시각화용 confidence 임계값")
    parser.add_argument("--labels", action="store_true", help="각 keypoint 점 옆에 이름을 표시 (어떤 점이 어떤 keypoint인지 눈으로 확인용)")
    parser.add_argument("--log", type=str, default=None, help="keypoint별 confidence를 저장할 CSV 경로")
    parser.add_argument("--participant", type=str, default=None,
                         help="데이터 수집 캡처 모드 활성화 — 참가자 ID (예: p01)")
    parser.add_argument("--label", type=str, default=VALID_LABELS[0], choices=VALID_LABELS,
                         help=f"캡처 모드에서 시작할 자세 클래스, 기본 {VALID_LABELS[0]} ({', '.join(VALID_LABELS)})")
    parser.add_argument("--save-dir", type=str, default="data/raw",
                         help="캡처 모드에서 이미지를 저장할 루트 폴더 (기본 data/raw, 참가자별 하위 폴더 자동 생성)")
    parser.add_argument("--hip-warn-threshold", type=float, default=0.3,
                         help="저장 시 hip keypoint confidence가 이 값 미만이면 경고 표시 (팔로 가려짐 등 감지용)")
    parser.add_argument("--subject-min-depth", type=float, default=0.3,
                         help="--realsense에서 전경 분리 시 유효 거리 하한(m), 기본 0.3")
    parser.add_argument("--subject-max-depth", type=float, default=1.0,
                         help="--realsense에서 전경 분리 시 유효 거리 상한(m), 기본 1.0 "
                              "(책상 앞 촬영 대상의 가슴/허리 실측 깊이가 보통 0.6~0.85m인 점을 "
                              "감안한 값 — data/capture_features_log.csv 참고). 자동 인식이 "
                              "계산한 범위도 이 값을 넘지 않도록 제한된다 — 이보다 먼 것은 항상 지워짐")
    parser.add_argument("--no-subject-isolation", action="store_true",
                         help="--realsense여도 depth 기반 전경 분리(다른 사람 지우기)를 끈다 — "
                              "두 명이 같이 화면에 잡혀야 하는 디버깅 등 특수한 경우에만 사용")
    parser.add_argument("--no-auto-calibrate", action="store_true",
                         help="시작 시 자동으로 '컴퓨터 사용자' 위치를 인식하는 기능을 끈다 — "
                              "끄면 --subject-min-depth/--subject-max-depth 값을 처음부터 그대로 사용")
    parser.add_argument("--calibration-confidence-threshold", type=float, default=0.15,
                         help="자동 인식(위치 계산)에 쓰는 confidence 기준, 기본 0.15 — "
                              "--hip-warn-threshold(기본 0.3)보다 낮게 잡아서, 구부정하거나 기울어진 "
                              "자세처럼 confidence가 원래 낮게 나오는 자세로 시작해도 인식되게 한다 "
                              "(바른 자세로 앉아 있어야만 인식되는 문제 방지)")
    parser.add_argument("--no-track-subject", action="store_true",
                         help="최초 인식 후 사용자를 계속 따라가며 관심영역을 갱신하는 기능을 끈다 — "
                              "끄면 최초 인식 당시 위치에 영역이 고정된 채로 유지됨")
    parser.add_argument("--judge", action="store_true",
                         help="자세·거리 판정(정상/주의/경고 + 화면 근접)을 화면에 표시하고, 샘플 주기마다 알림 조건을 "
                              "검사해 터미널에 출력한다 (5주차 판정 로직 1차 통합, src/logic/decision.py)")
    parser.add_argument("--sample-interval", type=float, default=5.0,
                         help="--judge에서 알림 상태머신에 샘플을 넣는 주기(초), 기본 5 (project.md 5초 샘플링)")
    args = parser.parse_args()

    if not Path(args.model).exists():
        raise SystemExit(f"모델 파일을 찾을 수 없습니다: {args.model} (스크립트 상단 docstring 참고)")

    capture_mode = bool(args.participant)
    save_dir = None
    serial = None
    label_idx = VALID_LABELS.index(args.label)
    shots_taken = 0  # 현재 클래스에서 이번 실행 중에 찍은 장수 (화면 표시용 카운터)
    if capture_mode:
        save_dir = Path(args.save_dir) / args.participant
        save_dir.mkdir(parents=True, exist_ok=True)
        serial = _next_serial(save_dir, args.participant, args.label)
        instruction = LABEL_INSTRUCTIONS.get(args.label, args.label)
        print(f"캡처 모드: {args.participant} / {args.label} — 's' 키로 저장, 다음 번호부터 시작: {serial:03d}")
        print(f"자세 지시: \"{instruction}\"")
        print(f"'n' 키를 누르면 원할 때 바로 다음 클래스로 넘어갑니다. 순서: {' → '.join(VALID_LABELS)}")

    instruction_font = _load_korean_font(30) if capture_mode else None
    if capture_mode and instruction_font is None:
        print("(참고: 화면에 한글 안내 문구를 못 그렸습니다 — Pillow 미설치 또는 한글 폰트를 못 찾음. "
              "터미널에 찍힌 자세 지시 문구를 참고해주세요. 필요하면 `pip install Pillow`)")

    # 화면 표시용 hip confidence는 최근 몇 프레임 평균을 쓴다 — 매 프레임 순간값만 보면
    # threshold(0.3) 경계에서 0.1초 단위로 초록/빨강이 번갈아 깜빡이는 것처럼 보일 수
    # 있어서(자연스러운 프레임 간 잡음), 화면이 안정적으로 읽히게 스무딩한다.
    hip_score_history: deque[float] = deque(maxlen=8)

    # 머리를 앞으로 내밀 때 머리가 어깨/엉덩이를 순간적으로 가리는 자기 가림(self-occlusion)
    # 대응 — 짧은 순간의 가림은 마지막으로 확실했던 위치를 유지한다.
    occlusion_smoother = KeypointOcclusionSmoother(hold_frames=10, confidence_threshold=args.hip_warn_threshold)

    extractor = MoveNetExtractor(args.model)
    source = _RealSenseSource(args.fps) if args.realsense else _WebcamSource(args.camera)

    log_file = None
    log_writer = None
    if args.log:
        log_file = open(args.log, "w", newline="", encoding="utf-8")
        log_writer = csv.writer(log_file)
        header = ["timestamp"] + [f"{name}_score" for name in KEYPOINT_NAMES]
        if args.realsense:
            header += DEPTH_FEATURE_KEYS
        log_writer.writerow(header)

    # 저장된 사진 각각에 대응하는 depth 특징을 같이 남긴다 — RGB 사진(raw_frame)과 달리
    # depth 프레임 자체는 저장하지 않으므로, 지금 안 남기면 이 정보는 영구히 사라진다.
    # data/raw/ 는 통째로 .gitignore 돼 있어서(원본 이미지 git 제외 규칙), 이 로그는
    # 한 단계 위(예: data/)에 별도로 둬서 labels.csv처럼 git에 올라가게 한다.
    capture_feature_log_path = None
    capture_feature_log_file = None
    capture_feature_log_writer = None
    if capture_mode and args.realsense:
        capture_feature_log_path = Path(args.save_dir).parent / "capture_features_log.csv"
        is_new = not capture_feature_log_path.exists()
        capture_feature_log_file = open(capture_feature_log_path, "a", newline="", encoding="utf-8")
        capture_feature_log_writer = csv.writer(capture_feature_log_file)
        if is_new:
            capture_feature_log_writer.writerow(
                ["filename", "participant_id", "label", "timestamp",
                 "nose_score", "left_shoulder_score", "right_shoulder_score",
                 "left_hip_score", "right_hip_score"] + DEPTH_FEATURE_KEYS
            )
        print(f"촬영 사진별 depth 특징도 같이 기록합니다: {capture_feature_log_path}")
    elif capture_mode and not args.realsense:
        print("(참고: --realsense 없이는 depth 특징을 기록할 수 없습니다 — 사진만 저장됩니다.)")

    subject_isolation = args.realsense and not args.no_subject_isolation
    auto_calibrate = not args.no_auto_calibrate

    # 수동으로 사각형을 그리는 대신, 화면에 잡힌 사람의 keypoint 위치·깊이로 "컴퓨터 사용자가
    # 있는 범위"를 자동으로 추정한다 — 시작 직후(또는 'r' 키로 재인식 시) 몇 프레임 연속으로
    # 확실하게 잡힌 사람을 그 사용자로 보고 ROI·depth 범위를 고정한다. 그 전까지는 마스킹 없이
    # 전체 화면을 보고(그래서 계산 자체가 가능), 고정된 뒤부터는 그 범위 밖을 지워 다른 사람이
    # 끼어들어도 무시한다.
    calibrated = not auto_calibrate  # 자동 인식을 끄면 처음부터 CLI로 받은 거리 범위를 그대로 사용
    roi_rect = None  # (x, y, w, h), 픽셀 좌표
    subject_min_depth = args.subject_min_depth
    subject_max_depth = args.subject_max_depth
    calibration_buffer: list[dict] = []
    calibration_misses = 0
    CALIBRATION_HITS_NEEDED = 8
    CALIBRATION_MISS_TOLERANCE = 3  # 이 횟수 안의 실패는 봐주고 계속 모은다 (잠깐의 가림 등)

    # 최초 인식 뒤에도 계속 사용자 위치를 추적해서 관심영역이 따라가게 한다(의자를 당기거나
    # 몸을 많이 움직여도 영역 밖으로 잘리지 않게). 중심 위치는 Kalman 필터로 추적한다 —
    # 속도까지 추정해서 잠깐 keypoint를 놓쳐도(가림 등) 그 방향으로 계속 예측하며 따라가다가
    # 다시 잡히면 보정한다. 박스 크기는 사용자가 카메라에 가까워지거나 멀어지면 같이 커지거나
    # 작아지도록 느리게(EMA) 따라간다 — 처음 잡은 크기가 맞지 않아도 서서히 맞춰짐.
    track_subject = auto_calibrate and not args.no_track_subject
    roi_tracker = _RoiKalmanTracker() if track_subject else None
    DEPTH_TRACK_ALPHA = 0.15  # depth 범위는 위치만큼 민감할 필요 없어 기존처럼 EMA로 충분

    if auto_calibrate:
        print("자동 인식: 처음에 화면에 잡힌 사람을 '컴퓨터 사용자'로 고정합니다 — "
              "구부정하거나 기울어진 자세로 시작해도 괜찮습니다. 다시 인식하려면 'r' 키."
              + (" 이후 사용자가 움직이면 관심영역도 같이 따라갑니다." if track_subject else ""))
    elif subject_isolation:
        print(f"전경 분리 사용: {subject_min_depth:.2f}~{subject_max_depth:.2f}m 밖은 검게 지웁니다 "
              f"(여러 명이 잡혀도 그 범위 안 사람만 인식 — 끄려면 --no-subject-isolation)")

    alert_machine = AlertStateMachine() if args.judge else None
    last_sample_t = None

    print(f"실행 중 (소스: {'RealSense D455' if args.realsense else f'웹캠 index={args.camera}'})... 'q'를 누르면 종료합니다.")
    try:
        while True:
            frame = source.read()
            if frame is None:
                continue

            combined_mask = None
            if calibrated:
                if subject_isolation:
                    combined_mask = source.get_foreground_mask(subject_min_depth, subject_max_depth)
                if roi_rect is not None:
                    x, y, w, h = _clamp_rect_to_frame(roi_rect, frame.shape)
                    roi_mask = np.zeros(frame.shape[:2], dtype=bool)
                    roi_mask[y:y + h, x:x + w] = True
                    combined_mask = roi_mask if combined_mask is None else (combined_mask & roi_mask)
                if combined_mask is not None:
                    frame[~combined_mask] = 0  # 범위 밖(다른 사람/배경)은 검게 지워 MoveNet이 못 보게 함

            raw_frame = frame.copy()  # 저장용 원본 (스켈레톤 안 그려진 상태, 마스킹은 반영됨)
            keypoints_raw = extractor.infer(frame)
            # 화면 표시·특징 계산에는 자기 가림 보정을 적용한 값을 쓰고, confidence 로그에는
            # 원본 점수를 그대로 남긴다(보정은 위치만 바꾸고 score는 원본을 유지하긴 하지만,
            # 로그는 항상 extractor가 준 원본 리스트를 기준으로 쓴다).
            keypoints = occlusion_smoother.smooth(keypoints_raw)

            depth_lookup = source.get_depth_lookup(raw_frame.shape) if hasattr(source, "get_depth_lookup") else None

            if auto_calibrate and not calibrated:
                bounds = _estimate_subject_bounds(keypoints_raw, raw_frame.shape, depth_lookup,
                                                   args.calibration_confidence_threshold)
                if bounds is not None:
                    calibration_buffer.append(bounds)
                    calibration_misses = 0
                else:
                    # 구부정하거나 기울어진 자세는 원래 confidence가 낮게 나올 수 있어(실측으로 확인),
                    # 한두 프레임 놓쳤다고 바로 포기하지 않고 몇 번은 봐준다. 그래도 계속 못 잡으면
                    # 사람이 자리를 비웠거나 다른 사람으로 바뀐 걸로 보고 처음부터 다시 모은다.
                    calibration_misses += 1
                    if calibration_misses > CALIBRATION_MISS_TOLERANCE:
                        calibration_buffer.clear()
                        calibration_misses = 0
                if len(calibration_buffer) >= CALIBRATION_HITS_NEEDED:
                    xs = [b["roi_rect"][0] for b in calibration_buffer]
                    ys = [b["roi_rect"][1] for b in calibration_buffer]
                    ws = [b["roi_rect"][2] for b in calibration_buffer]
                    hs = [b["roi_rect"][3] for b in calibration_buffer]
                    roi_rect = (int(sum(xs) / len(xs)), int(sum(ys) / len(ys)),
                                int(sum(ws) / len(ws)), int(sum(hs) / len(hs)))
                    depth_ranges = [b["depth_range"] for b in calibration_buffer if b["depth_range"] is not None]
                    if depth_ranges:
                        # 자동 계산값이라도 --subject-min-depth/--subject-max-depth(기본 0.3~1.0m)
                        # 범위를 벗어나 더 느슨해지지는 않게 한다 — 예: 1m보다 먼 건 항상 지워짐.
                        subject_min_depth = max(args.subject_min_depth, min(d[0] for d in depth_ranges))
                        subject_max_depth = min(args.subject_max_depth, max(d[1] for d in depth_ranges))
                    calibrated = True
                    if roi_tracker is not None:
                        roi_tracker.init(roi_rect)  # 이때의 위치·크기를 시작값으로 추적 시작
                    print(f"자동 인식 완료 — ROI={roi_rect}" +
                          (f", depth={subject_min_depth:.2f}~{subject_max_depth:.2f}m" if depth_ranges else "") +
                          " (다시 인식하려면 'r')")
            elif track_subject and calibrated:
                # 매 프레임 predict()로 속도 기반 다음 위치를 구하고, keypoint가 이번 프레임에
                # 잡히면 correct()로 보정한다. 가려짐 등으로 못 잡아도 predict() 결과(그동안의
                # 이동 방향·속도로 계속 예측한 위치)를 쓰므로 제자리에 멈추지 않고 계속 따라간다.
                predicted_rect = roi_tracker.predict()
                bounds = _estimate_subject_bounds(keypoints_raw, raw_frame.shape, depth_lookup,
                                                   args.calibration_confidence_threshold)
                if bounds is not None:
                    roi_rect = roi_tracker.correct(bounds["roi_rect"])
                    if subject_isolation and bounds["depth_range"] is not None:
                        d_min = max(args.subject_min_depth, bounds["depth_range"][0])
                        d_max = min(args.subject_max_depth, bounds["depth_range"][1])
                        subject_min_depth += DEPTH_TRACK_ALPHA * (d_min - subject_min_depth)
                        subject_max_depth += DEPTH_TRACK_ALPHA * (d_max - subject_max_depth)
                else:
                    roi_rect = predicted_rect

            frame = draw_skeleton(frame, keypoints, args.threshold, show_labels=args.labels)
            if auto_calibrate and not calibrated:
                cv2.putText(frame, "자동 인식 중... (아무 자세나 괜찮음, 화면에 잠시 있어주세요)", (10, frame.shape[0] - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 1, cv2.LINE_AA)

            if capture_mode:
                instruction = LABEL_INSTRUCTIONS.get(args.label, args.label)
                frame = draw_text_kr(frame, f"[{args.label}] {instruction}", (10, 8),
                                      instruction_font, color_bgr=(0, 255, 255))

            features = compute_posture_features(keypoints)
            y0 = 60 if capture_mode else 24  # 캡처 모드는 위쪽에 한글 안내 문구가 있어 자리를 비켜준다
            if features:
                for k, v in features.items():
                    cv2.putText(frame, f"{k}: {v:.1f}", (10, y0), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    y0 += 20

            depth_features = None
            if depth_lookup is not None:
                depth_features = compute_depth_features(keypoints, depth_lookup)
                if depth_features:
                    for k in DEPTH_FEATURE_KEYS:
                        v = depth_features.get(k)
                        text = f"{k}: {v:.3f}" if v is not None else f"{k}: -"
                        cv2.putText(frame, text, (10, y0), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (170, 220, 255), 1, cv2.LINE_AA)
                        y0 += 20

            if alert_machine is not None and not (auto_calibrate and not calibrated):
                judgement = judge(keypoints, depth_features)
                color = {NORMAL: (0, 255, 170), CAUTION: (0, 200, 255), WARNING: (0, 0, 255)}[judgement.posture_level]
                kind = f" ({judgement.posture_kind})" if judgement.posture_kind else ""
                near = "  [화면 근접]" if judgement.proximity else ""
                cv2.putText(frame, f"{judgement.posture_level}{kind}{near}", (frame.shape[1] - 330, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
                now = time.time()
                if last_sample_t is None or now - last_sample_t >= args.sample_interval:
                    last_sample_t = now
                    for alert in alert_machine.update(now, judgement):
                        print(f"[알림] {alert} — {', '.join(judgement.reasons)}")

            if log_writer:
                row = [time.time()] + [f"{kp.score:.4f}" for kp in keypoints_raw]
                if args.realsense:
                    row += [
                        (f"{depth_features[k]:.4f}" if depth_features and depth_features.get(k) is not None else "")
                        for k in DEPTH_FEATURE_KEYS
                    ]
                log_writer.writerow(row)

            if capture_mode:
                hip_score = min(keypoints[11].score, keypoints[12].score)  # left_hip, right_hip (순간값)
                hip_score_history.append(hip_score)
                hip_score_smoothed = sum(hip_score_history) / len(hip_score_history)  # 화면 표시용 (최근 8프레임 평균)
                hip_ok = hip_score_smoothed >= args.hip_warn_threshold
                status_color = (0, 255, 170) if hip_ok else (0, 0, 255)
                cv2.putText(frame, f"[{args.label}] {shots_taken}장 찍음 (다음 번호: {serial:03d})",
                            (10, frame.shape[0] - 46), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, f"hip conf: {hip_score_smoothed:.2f}{'  (낮음 - 팔/가림 확인)' if not hip_ok else ''}",
                            (10, frame.shape[0] - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 1, cv2.LINE_AA)
                cv2.putText(frame, "'s' = 저장, 'n' = 다음 자세, 'r' = 다시 자동 인식, 'q' = 종료", (10, frame.shape[0] - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow("MoveNet 17 Keypoints (q to quit)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r") and auto_calibrate:
                calibrated = False
                roi_rect = None
                subject_min_depth = args.subject_min_depth
                subject_max_depth = args.subject_max_depth
                calibration_buffer.clear()
                calibration_misses = 0
                print("다시 자동 인식합니다 — 어떤 자세로 있어도 괜찮습니다.")
            if capture_mode and key == ord("s"):
                filename = f"{args.participant}_{args.label}_{serial:03d}.jpg"
                out_path = save_dir / filename
                cv2.imwrite(str(out_path), raw_frame)
                # 저장 시점 판단도 화면과 똑같이 스무딩된 값을 기준으로 삼는다 (순간 잡음 하나로
                # 잘못된 경고가 찍히지 않게).
                hip_score_smoothed = sum(hip_score_history) / len(hip_score_history) if hip_score_history else 0.0
                warn = "  ※ hip confidence 낮음 — 팔/몸에 가려졌을 수 있음, 확인 권장" if hip_score_smoothed < args.hip_warn_threshold else ""
                print(f"저장: {out_path} (hip conf: {hip_score_smoothed:.2f}){warn}")
                if capture_feature_log_writer is not None:
                    row = [filename, args.participant, args.label, f"{time.time():.3f}"]
                    row += [
                        f"{keypoints_raw[0].score:.4f}", f"{keypoints_raw[5].score:.4f}",
                        f"{keypoints_raw[6].score:.4f}", f"{keypoints_raw[11].score:.4f}",
                        f"{keypoints_raw[12].score:.4f}",
                    ]
                    row += [
                        (f"{depth_features[k]:.4f}" if depth_features and depth_features.get(k) is not None else "")
                        for k in DEPTH_FEATURE_KEYS
                    ]
                    capture_feature_log_writer.writerow(row)
                    capture_feature_log_file.flush()
                serial += 1
                shots_taken += 1

            if capture_mode and key == ord("n"):
                label_idx += 1
                if label_idx >= len(VALID_LABELS):
                    print("모든 클래스(5개) 촬영이 끝났습니다. 'q'로 종료하거나, 필요하면 더 찍어도 됩니다"
                          f"(마지막 클래스 '{args.label}' 계속 저장됨).")
                    label_idx = len(VALID_LABELS) - 1  # 마지막 클래스에 그대로 머문다 (원하면 계속 찍을 수 있게)
                else:
                    args.label = VALID_LABELS[label_idx]
                    serial = _next_serial(save_dir, args.participant, args.label)
                    shots_taken = 0
                    instruction = LABEL_INSTRUCTIONS.get(args.label, args.label)
                    print(f"\n다음 클래스: [{args.label}] 다음 번호부터 시작: {serial:03d}")
                    print(f"자세 지시: \"{instruction}\"")
    finally:
        source.release()
        cv2.destroyAllWindows()
        if log_file:
            log_file.close()
            print(f"confidence 로그 저장 완료: {args.log}")
        if capture_feature_log_file:
            capture_feature_log_file.close()
            print(f"촬영 depth 특징 로그 저장 완료: {capture_feature_log_path}")


if __name__ == "__main__":
    main()
