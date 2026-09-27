#!/usr/bin/env python3
"""
바른자세 - MoveNet 17키포인트 추출 모듈

용도
----
1) 웹캠(또는 추후 스테레오 카메라의 좌안 프레임)에서 MoveNet Lightning으로
   17개 keypoint를 실시간 추출하고 화면에 스켈레톤을 그려 보여준다.
2) 각 keypoint의 confidence를 CSV로 기록한다 — "상반신만 보이는 책상 구도에서
   어떤 keypoint를 신뢰할 수 있는가"를 실측으로 검증하기 위한 기초 데이터다.

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

    # confidence 로그를 CSV로 저장 (카메라 프레이밍/keypoint 신뢰도 검증용)
    python -m src.pose.movenet_keypoints --model models/movenet_lightning_int8.tflite \
        --log keypoint_confidence_log.csv --camera 0
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "opencv-python이 필요합니다: pip install opencv-python"
    ) from exc

from src.features.posture_features import compute_posture_features


# ---------------------------------------------------------------------------
# MoveNet(COCO) 17키포인트 순서 — 모델 출력이 항상 이 순서로 나온다.
# 어떤 keypoint를 실제로 쓸지(상반신/전신)는 src/features/posture_features.py의
# UPPER_BODY_IDX/LOWER_BODY_IDX 참고.
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
