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
    # 스크립트를 한 번만 켜두면 클래스당 --shots-per-label(기본 5)장 찍을 때마다
    # normal → slouch_forward → slouch_back → tilt_left → tilt_right 순서로 자동으로
    # 넘어간다 ('n' 키로 언제든 수동으로도 넘길 수 있음). 재시작 없이 5클래스 전부 촬영 가능.
    python -m src.pose.movenet_keypoints --model models/movenet_lightning_int8.tflite \
        --realsense --participant p01 --shots-per-label 5
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

    def release(self) -> None:
        self._cam.stop()


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
    parser.add_argument("--shots-per-label", type=int, default=5,
                         help="캡처 모드에서 한 클래스당 이 장수를 찍으면 자동으로 다음 클래스로 넘어감 (기본 5). "
                              "'n' 키로 언제든 수동으로도 넘길 수 있음")
    parser.add_argument("--save-dir", type=str, default="data/raw",
                         help="캡처 모드에서 이미지를 저장할 루트 폴더 (기본 data/raw, 참가자별 하위 폴더 자동 생성)")
    parser.add_argument("--hip-warn-threshold", type=float, default=0.3,
                         help="저장 시 hip keypoint confidence가 이 값 미만이면 경고 표시 (팔로 가려짐 등 감지용)")
    args = parser.parse_args()

    if not Path(args.model).exists():
        raise SystemExit(f"모델 파일을 찾을 수 없습니다: {args.model} (스크립트 상단 docstring 참고)")

    capture_mode = bool(args.participant)
    save_dir = None
    serial = None
    label_idx = VALID_LABELS.index(args.label)
    shots_taken = 0  # 현재 클래스에서 이번 실행 중에 찍은 장수 (--shots-per-label 도달하면 자동 전환)
    if capture_mode:
        save_dir = Path(args.save_dir) / args.participant
        save_dir.mkdir(parents=True, exist_ok=True)
        serial = _next_serial(save_dir, args.participant, args.label)
        instruction = LABEL_INSTRUCTIONS.get(args.label, args.label)
        print(f"캡처 모드: {args.participant} / {args.label} — 's' 키로 저장, 다음 번호부터 시작: {serial:03d}")
        print(f"자세 지시: \"{instruction}\"")
        print(f"클래스당 {args.shots_per_label}장 찍으면 자동으로 다음 클래스로 넘어갑니다. "
              f"'n' 키로 언제든 바로 넘어갈 수도 있습니다. 순서: {' → '.join(VALID_LABELS)}")

    instruction_font = _load_korean_font(30) if capture_mode else None
    if capture_mode and instruction_font is None:
        print("(참고: 화면에 한글 안내 문구를 못 그렸습니다 — Pillow 미설치 또는 한글 폰트를 못 찾음. "
              "터미널에 찍힌 자세 지시 문구를 참고해주세요. 필요하면 `pip install Pillow`)")

    # 화면 표시용 hip confidence는 최근 몇 프레임 평균을 쓴다 — 매 프레임 순간값만 보면
    # threshold(0.3) 경계에서 0.1초 단위로 초록/빨강이 번갈아 깜빡이는 것처럼 보일 수
    # 있어서(자연스러운 프레임 간 잡음), 화면이 안정적으로 읽히게 스무딩한다.
    hip_score_history: deque[float] = deque(maxlen=8)

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

    print(f"실행 중 (소스: {'RealSense D455' if args.realsense else f'웹캠 index={args.camera}'})... 'q'를 누르면 종료합니다.")
    try:
        while True:
            frame = source.read()
            if frame is None:
                continue

            raw_frame = frame.copy()  # 저장용 원본 (스켈레톤 안 그려진 상태)
            keypoints = extractor.infer(frame)
            frame = draw_skeleton(frame, keypoints, args.threshold, show_labels=args.labels)

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
            depth_lookup = source.get_depth_lookup(raw_frame.shape) if hasattr(source, "get_depth_lookup") else None
            if depth_lookup is not None:
                depth_features = compute_depth_features(keypoints, depth_lookup)
                if depth_features:
                    for k in DEPTH_FEATURE_KEYS:
                        v = depth_features.get(k)
                        text = f"{k}: {v:.3f}" if v is not None else f"{k}: -"
                        cv2.putText(frame, text, (10, y0), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (170, 220, 255), 1, cv2.LINE_AA)
                        y0 += 20

            if log_writer:
                row = [time.time()] + [f"{kp.score:.4f}" for kp in keypoints]
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
                cv2.putText(frame, f"[{args.label}] {shots_taken}/{args.shots_per_label}장 (다음 번호: {serial:03d})",
                            (10, frame.shape[0] - 46), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, f"hip conf: {hip_score_smoothed:.2f}{'  (낮음 - 팔/가림 확인)' if not hip_ok else ''}",
                            (10, frame.shape[0] - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 1, cv2.LINE_AA)
                cv2.putText(frame, "'s' = 저장, 'n' = 다음 자세로 넘어가기, 'q' = 종료", (10, frame.shape[0] - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow("MoveNet 17 Keypoints (q to quit)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if capture_mode and key == ord("s"):
                filename = f"{args.participant}_{args.label}_{serial:03d}.jpg"
                out_path = save_dir / filename
                cv2.imwrite(str(out_path), raw_frame)
                # 저장 시점 판단도 화면과 똑같이 스무딩된 값을 기준으로 삼는다 (순간 잡음 하나로
                # 잘못된 경고가 찍히지 않게).
                hip_score_smoothed = sum(hip_score_history) / len(hip_score_history) if hip_score_history else 0.0
                warn = "  ※ hip confidence 낮음 — 팔/몸에 가려졌을 수 있음, 확인 권장" if hip_score_smoothed < args.hip_warn_threshold else ""
                print(f"저장: {out_path} (hip conf: {hip_score_smoothed:.2f}){warn}")
                serial += 1
                shots_taken += 1
                if shots_taken >= args.shots_per_label and label_idx < len(VALID_LABELS) - 1:
                    print(f"[{args.label}] {args.shots_per_label}장 촬영 완료 — 자동으로 다음 클래스로 넘어갑니다.")
                    key = ord("n")  # 아래 'n' 처리 분기를 그대로 재사용해서 넘어간다

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


if __name__ == "__main__":
    main()
