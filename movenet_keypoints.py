#!/usr/bin/env python3
"""
바른자세 - MoveNet 17키포인트 추출 모듈

용도
----
1) 웹캠(또는 추후 스테레오 카메라의 좌안 프레임)에서 MoveNet Lightning으로
   17개 keypoint를 실시간 추출하고 화면에 스켈레톤을 그려 보여준다.
2) 각 keypoint의 confidence를 CSV로 기록한다 — "상반신만 보이는 책상 구도에서
   어떤 keypoint를 신뢰할 수 있는가"를 실측으로 검증하기 위한 기초 데이터다.
3) project.md 5장 1단계에서 정의한 특징 추출(엉덩이 중심 정규화 + 어깨너비
   스케일링)을 구현해, RF 학습에 바로 넘길 수 있는 특징 벡터를 만든다.

모델 준비
--------
이 스크립트는 TFLite 형식의 MoveNet SinglePose Lightning 모델 파일이 필요하다.
TensorFlow Hub의 모델 배포처가 Kaggle Models로 통합되어, 지금은 아래 Kaggle
페이지가 공식 다운로드 경로다 (본 개발 환경은 네트워크 제한으로 직접 받을 수
없어 코드만 준비해두었다. 실제 노트북에서 받으면 된다):

    # https://www.kaggle.com/models/google/movenet/tfLite/singlepose-lightning-tflite-int8
    # 위 페이지에서 "Download" 받은 .tflite 파일을 models/ 아래에 둔다.
    # (float16 버전을 쓰고 싶으면 같은 페이지에서 singlepose-lightning-tflite-fp16 선택)
    #
    # 입출력 스펙(둘 다 입력은 uint8, 192x192, 출력은 [1,1,17,3]=y,x,score)은
    # TensorFlow 공식 튜토리얼에 정리되어 있다:
    # https://www.tensorflow.org/hub/tutorials/movenet

의존성
------
    pip install tflite-runtime opencv-python numpy
    # tflite-runtime이 설치가 안 되는 환경(일부 Windows)이면 대신
    #   pip install tensorflow
    # 을 설치하면 tensorflow.lite.Interpreter로 자동 대체된다.

실행 예시
--------
    # 웹캠으로 실시간 확인 (q로 종료)
    python movenet_keypoints.py --model models/movenet_lightning_int8.tflite

    # confidence 로그를 CSV로 저장 (카메라 프레이밍/keypoint 신뢰도 검증용)
    python movenet_keypoints.py --model models/movenet_lightning_int8.tflite \
        --log keypoint_confidence_log.csv --camera 0
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "opencv-python이 필요합니다: pip install opencv-python"
    ) from exc


# ---------------------------------------------------------------------------
# MoveNet(COCO) 17키포인트 순서 — 모델 출력이 항상 이 순서로 나온다.
# ---------------------------------------------------------------------------
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

# 책상 착석 구도에서 "일단 상반신 위주"로 다룰 때 참고할 인덱스 그룹.
# 실제 채택 여부는 확보한 confidence 로그로 검증한 뒤 확정한다(project.md 5장 참고).
UPPER_BODY_IDX = list(range(0, 13))  # nose ~ hip (Pawitra et al. 2026 과 동일 기준)
LOWER_BODY_IDX = list(range(13, 17))  # knee, ankle — 책상 구도에서는 대체로 신뢰 불가

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


def filter_by_confidence(keypoints: list[Keypoint], threshold: float) -> list[Optional[Keypoint]]:
    """threshold 미만인 keypoint는 None으로 바꿔 이후 계산에서 제외한다."""
    return [kp if kp.score >= threshold else None for kp in keypoints]


def normalize_keypoints(keypoints: list[Keypoint]) -> Optional[np.ndarray]:
    """
    project.md 5장 1단계: 엉덩이(hip) 중심으로 원점 이동 + 어깨너비로 스케일 정규화.

    반환값: (17, 2) 배열 (정규화된 x, y). 엉덩이가 안 보여 계산이 불가능하면 None.
    """
    left_hip, right_hip = keypoints[11], keypoints[12]
    left_shoulder, right_shoulder = keypoints[5], keypoints[6]

    if min(left_hip.score, right_hip.score, left_shoulder.score, right_shoulder.score) < 1e-6:
        # 이 함수는 순수 정규화용이라 confidence 필터링은 호출부에서 이미 했다고 가정하되,
        # 완전히 0점(=탐지 자체가 안 된 경우)만 최소 방어한다.
        pass

    hip_center = np.array([(left_hip.x + right_hip.x) / 2.0, (left_hip.y + right_hip.y) / 2.0])
    shoulder_width = float(np.hypot(left_shoulder.x - right_shoulder.x, left_shoulder.y - right_shoulder.y))

    if shoulder_width < 1e-6:
        return None  # 어깨가 겹쳐 보이거나 검출 실패 — 스케일 기준으로 못 씀

    coords = np.array([[kp.x, kp.y] for kp in keypoints])
    normalized = (coords - hip_center) / shoulder_width
    return normalized


def angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """세 점 a-b-c에서 b를 꼭짓점으로 하는 각도(도)를 계산한다."""
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def compute_posture_features(keypoints: list[Keypoint]) -> Optional[dict]:
    """
    정규화된 keypoint로부터 자세 판단에 쓸 각도/거리 특징 몇 가지를 계산한다.
    (RF 학습용 전체 특징 세트는 6주차에 실측 데이터로 확정 — 여기서는 뼈대만 제공)
    """
    normalized = normalize_keypoints(keypoints)
    if normalized is None:
        return None

    nose, l_sh, r_sh, l_hip, r_hip = (
        normalized[0], normalized[5], normalized[6], normalized[11], normalized[12]
    )
    neck = (l_sh + r_sh) / 2.0
    hip_center = (l_hip + r_hip) / 2.0

    return {
        "neck_tilt_deg": angle_deg(nose, neck, neck + np.array([1.0, 0.0])),  # 목 좌우 기울임
        "torso_lean_deg": angle_deg(neck, hip_center, hip_center + np.array([0.0, -1.0])),  # 상체 앞뒤 기울임
        "shoulder_slope_deg": angle_deg(l_sh, neck, r_sh),  # 어깨 좌우 비대칭
        "nose_to_hip_dist": float(np.linalg.norm(nose - hip_center)),  # 화면과의 거리 대용(가까워지면 값이 커짐)
    }


def draw_skeleton(frame: np.ndarray, keypoints: list[Keypoint], threshold: float) -> np.ndarray:
    h, w = frame.shape[:2]
    for kp in keypoints:
        if kp.score < threshold:
            continue
        cx, cy = int(kp.x * w), int(kp.y * h)
        cv2.circle(frame, (cx, cy), 4, (0, 255, 170), -1)

    for i, j in SKELETON_EDGES:
        a, b = keypoints[i], keypoints[j]
        if a.score < threshold or b.score < threshold:
            continue
        pa = (int(a.x * w), int(a.y * h))
        pb = (int(b.x * w), int(b.y * h))
        cv2.line(frame, pa, pb, (255, 200, 0), 2)

    return frame


def main():
    parser = argparse.ArgumentParser(description="MoveNet 17키포인트 실시간 추출/기록")
    parser.add_argument("--model", required=True, help="MoveNet Lightning .tflite 모델 경로")
    parser.add_argument("--camera", type=int, default=0, help="cv2.VideoCapture 인덱스 (기본 0)")
    parser.add_argument("--threshold", type=float, default=0.3, help="시각화용 confidence 임계값")
    parser.add_argument("--log", type=str, default=None, help="keypoint별 confidence를 저장할 CSV 경로")
    args = parser.parse_args()

    if not Path(args.model).exists():
        raise SystemExit(f"모델 파일을 찾을 수 없습니다: {args.model} (스크립트 상단 docstring 참고)")

    extractor = MoveNetExtractor(args.model)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"카메라(index={args.camera})를 열 수 없습니다.")

    log_file = None
    log_writer = None
    if args.log:
        log_file = open(args.log, "w", newline="", encoding="utf-8")
        log_writer = csv.writer(log_file)
        log_writer.writerow(["timestamp"] + [f"{name}_score" for name in KEYPOINT_NAMES])

    print("실행 중... 'q'를 누르면 종료합니다.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("프레임을 읽지 못했습니다.")
                break

            keypoints = extractor.infer(frame)
            frame = draw_skeleton(frame, keypoints, args.threshold)

            features = compute_posture_features(keypoints)
            if features:
                y0 = 24
                for k, v in features.items():
                    cv2.putText(frame, f"{k}: {v:.1f}", (10, y0), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    y0 += 20

            if log_writer:
                log_writer.writerow([time.time()] + [f"{kp.score:.4f}" for kp in keypoints])

            cv2.imshow("MoveNet 17 Keypoints (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if log_file:
            log_file.close()
            print(f"confidence 로그 저장 완료: {args.log}")


if __name__ == "__main__":
    main()
