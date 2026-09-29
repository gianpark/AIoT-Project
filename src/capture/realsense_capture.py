#!/usr/bin/env python3
"""
바른자세 - Intel RealSense D455 캡처 모듈

용도
----
1) RealSense 파이프라인(depth + color 동기화 스트림)을 감싸는 얇은 래퍼(`RealSenseCamera`)를 제공한다.
2) `view` 모드 — 화면 중앙 십자선까지의 거리와 fps를 실시간으로 보여준다. 줄자로 실제
   거리를 재면서 SDK가 보고하는 값과 비교하는 용도(project.md 4주차 게이트의 "거리 정확도"
   검증). RealSense는 출고 시 스테레오 캘리브레이션이 끝나 있어 체스보드 캘리브레이션은
   생략하고, 이렇게 실측 검증만 한다.
3) `gate` 모드 — N초 동안 fps와 프레임 간격(타임스탬프 차이)을 자동으로 측정해 평균 fps,
   최대 프레임 간격("스파이크"), 스파이크 발생 횟수를 요약 출력한다(project.md 4주차 게이트의
   "fps·프레임타임 스파이크" 항목).

카메라 없이는 실행 불가
----------------------
이 모듈은 실제 RealSense D455 하드웨어가 연결되어 있어야 동작한다(이 저장소를 만든
개발 환경에는 카메라가 없어 `pyrealsense2` 설치·문법 검증까지만 하고, 실제 실행은
카메라가 연결된 팀원 노트북에서 해야 한다). 카메라 인식 자체가 안 되면 먼저
`realsense-viewer`(SDK 설치 시 같이 깔리는 GUI 도구)로 스트림이 뜨는지 확인할 것
(README 참고).

의존성
------
    pip install -r requirements.txt   # pyrealsense2, opencv-python 포함

실행 예시
--------
    # 실시간 화면 + 거리·fps 표시 (줄자 대고 거리 정확도 확인, q로 종료)
    python -m src.capture.realsense_capture view

    # 30초 동안 fps/프레임타임 스파이크 자동 측정
    python -m src.capture.realsense_capture gate --seconds 30
"""

from __future__ import annotations

import argparse
import time

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "opencv-python이 필요합니다: pip install opencv-python"
    ) from exc

try:
    import pyrealsense2 as rs
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "pyrealsense2가 필요합니다: pip install pyrealsense2\n"
        "(Apple Silicon Mac은 pip 설치가 안 될 수 있음 - README 참고)"
    ) from exc


DEFAULT_WIDTH = 848
DEFAULT_HEIGHT = 480
DEFAULT_FPS = 30


class RealSenseCamera:
    """RealSense 파이프라인을 감싸는 얇은 래퍼 - depth+color 동기화 스트림."""

    def __init__(self, width: int = DEFAULT_WIDTH, height: int = DEFAULT_HEIGHT,
                 fps: int = DEFAULT_FPS, use_filters: bool = True):
        self.width = width
        self.height = height
        self.fps = fps
        self.use_filters = use_filters

        self._pipeline = rs.pipeline()
        self._config = rs.config()
        self._config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self._config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

        # depth를 color 시점으로 정렬 - 같은 픽셀 좌표로 색상/깊이를 함께 참조하기 위함
        self._align = rs.align(rs.stream.color)

        # project.md 6장 "5주차: RealSense depth 프리셋·필터 튜닝"에서 다듬을 필터들.
        # 지금은 노이즈를 줄이는 기본값으로 켜두고, 5주차에 파라미터를 조정한다.
        self._spatial = rs.spatial_filter()
        self._temporal = rs.temporal_filter()
        self._hole_filling = rs.hole_filling_filter()

        self._profile = None

    def start(self) -> "RealSenseCamera":
        try:
            self._profile = self._pipeline.start(self._config)
        except RuntimeError as exc:
            raise SystemExit(
                f"RealSense 카메라를 열 수 없습니다: {exc}\n"
                "- USB 3.0/3.1 Type-C 포트에 꽂혀있는지 확인\n"
                "- realsense-viewer로 먼저 인식되는지 확인 (README 참고)"
            ) from exc
        return self

    def stop(self) -> None:
        if self._profile is not None:
            self._pipeline.stop()
            self._profile = None

    def __enter__(self) -> "RealSenseCamera":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def read(self):
        """
        동기화된 color+depth 프레임 한 쌍을 읽는다.

        반환: (color_image, depth_frame, timestamp_ms) 또는 프레임이 아직 안 왔으면 None.
        - color_image: BGR np.ndarray (OpenCV로 바로 표시/처리 가능)
        - depth_frame: 원본 rs.depth_frame 객체 그대로 반환 — get_distance(x, y) 같은
          메서드를 호출부(게이트 측정 등)에서 바로 쓸 수 있게 하기 위함
        - timestamp_ms: 프레임 캡처 시각(ms) — fps/프레임타임 스파이크 계산에 사용
        """
        frames = self._pipeline.wait_for_frames()
        frames = self._align.process(frames)

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            return None

        if self.use_filters:
            depth_frame = self._spatial.process(depth_frame)
            depth_frame = self._temporal.process(depth_frame)
            depth_frame = self._hole_filling.process(depth_frame)
            depth_frame = depth_frame.as_depth_frame()

        color_image = np.asanyarray(color_frame.get_data())
        timestamp_ms = color_frame.get_timestamp()  # RealSense 프레임 타임스탬프는 기본 ms 단위
        return color_image, depth_frame, timestamp_ms

    @staticmethod
    def get_distance_m(depth_frame, x: int, y: int) -> float:
        """(x, y) 픽셀의 추정 거리(미터)."""
        return float(depth_frame.get_distance(x, y))


def _view_mode(args: argparse.Namespace) -> None:
    """
    실시간으로 color 프레임 + 중앙 십자선 거리값 + fps를 화면에 띄운다.
    줄자로 실측하면서 눈으로 오차를 확인하는 용도(project.md 4주차 게이트 중 '거리 정확도').
    """
    with RealSenseCamera(fps=args.fps) as cam:
        print("실행 중... 'q'를 누르면 종료합니다. 화면 중앙 십자선까지의 거리가 표시됩니다.")
        prev_t = time.time()
        fps_smoothed = 0.0
        try:
            while True:
                result = cam.read()
                if result is None:
                    continue
                color_image, depth_frame, _ = result

                h, w = color_image.shape[:2]
                cx, cy = w // 2, h // 2
                distance = cam.get_distance_m(depth_frame, cx, cy)

                now = time.time()
                dt = now - prev_t
                prev_t = now
                if dt > 0:
                    fps_smoothed = 0.9 * fps_smoothed + 0.1 * (1.0 / dt)

                cv2.drawMarker(color_image, (cx, cy), (0, 255, 170), cv2.MARKER_CROSS, 20, 2)
                cv2.putText(color_image, f"distance: {distance:.3f} m", (10, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(color_image, f"fps: {fps_smoothed:.1f}", (10, 54),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

                cv2.imshow("RealSense D455 (q to quit)", color_image)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            cv2.destroyAllWindows()


def _gate_mode(args: argparse.Namespace) -> None:
    """
    project.md 4주차 게이트: fps·프레임타임 스파이크를 N초 동안 측정해서 요약 출력한다.
    거리 정확도는 이 모드가 아니라 `view` 모드에서 사람이 줄자를 대고 눈으로 확인한다
    (자동화하려면 정확히 알려진 거리에 고정 타겟을 두는 별도 셋업이 필요해서 4주차 범위를 넘어감).
    """
    frame_times_ms: list[float] = []
    print(f"{args.seconds}초 동안 게이트 측정 중... (화면은 안 뜨고 콘솔에만 진행상황 표시)")

    with RealSenseCamera(fps=args.fps) as cam:
        start = time.time()
        last_ts = None
        count = 0
        while time.time() - start < args.seconds:
            result = cam.read()
            if result is None:
                continue
            _, _, ts_ms = result
            if last_ts is not None:
                frame_times_ms.append(ts_ms - last_ts)
            last_ts = ts_ms
            count += 1

    if not frame_times_ms:
        raise SystemExit("프레임을 하나도 못 받았습니다 - 카메라 연결을 확인하세요.")

    elapsed = args.seconds
    avg_fps = count / elapsed
    avg_frame_time = sum(frame_times_ms) / len(frame_times_ms)
    max_frame_time = max(frame_times_ms)
    spike_threshold_ms = 1000.0 / args.fps * 2  # 목표 프레임 간격의 2배 넘으면 스파이크로 간주
    spikes = [t for t in frame_times_ms if t > spike_threshold_ms]

    print("\n=== 4주차 게이트 측정 결과 ===")
    print(f"총 프레임 수: {count}개 / {elapsed:.1f}초")
    print(f"평균 fps: {avg_fps:.2f}  (목표: 최소 5fps, 목표치 8~10fps — project.md 7장 참고)")
    print(f"평균 프레임 간격: {avg_frame_time:.1f} ms")
    print(f"최대 프레임 간격(가장 심한 스파이크): {max_frame_time:.1f} ms")
    print(f"스파이크(간격 > {spike_threshold_ms:.0f}ms) 발생 횟수: {len(spikes)}회")
    if avg_fps < 5:
        print("⚠ 목표 fps(5) 미달 - 해상도/fps 설정을 낮춰서(--fps 15 등) 재시도해볼 것")


def main() -> None:
    parser = argparse.ArgumentParser(description="RealSense D455 캡처 / 4주차 게이트 측정")
    sub = parser.add_subparsers(dest="mode", required=True)

    view_p = sub.add_parser("view", help="실시간 화면 + 거리·fps 표시 (줄자로 거리 정확도 확인용)")
    view_p.add_argument("--fps", type=int, default=DEFAULT_FPS)

    gate_p = sub.add_parser("gate", help="N초 동안 fps/프레임타임 스파이크 자동 측정")
    gate_p.add_argument("--seconds", type=float, default=30.0)
    gate_p.add_argument("--fps", type=int, default=DEFAULT_FPS)

    args = parser.parse_args()
    if args.mode == "view":
        _view_mode(args)
    elif args.mode == "gate":
        _gate_mode(args)


if __name__ == "__main__":
    main()
