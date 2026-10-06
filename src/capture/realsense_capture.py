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

    # 5주차: 여러 거리에서 실측 대조 기록 (40/60/80/100/150/200cm 등에 줄자로 표시해두고,
    # 그 지점에 십자선을 맞춘 뒤 's'를 눌러 실제 거리(cm)를 입력하면 오차가 CSV에 쌓인다)
    python -m src.capture.realsense_capture view --log data/distance_accuracy_log.csv

    # 5주차: 필터 파라미터를 바꿔가며 같은 방식으로 비교 (기본값 대비 스무딩을 더 강하게)
    python -m src.capture.realsense_capture view --log data/distance_accuracy_log.csv --spatial-alpha 0.7 --temporal-alpha 0.6

    # 5주차: depth 필터 프리셋(필터없음/SDK기본/강/약...) 자동 비교 — 평소 작업 거리에서 가만히 앉아서
    python -m src.capture.realsense_capture filters --out data/depth_filter_tuning.csv --note "60cm 정면"

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

# D455 depth 유효 측정 범위(스펙상 약 0.6~6m). 이 범위 밖 값은 스테레오 매칭 실패로
# 인한 무효값(0 또는 필터를 거치며 생기는 uint16 최대치 65.535m 등)이라 화면에
# "invalid"로 표시하고 숫자 그대로는 보여주지 않는다.
MIN_VALID_DEPTH_M = 0.3
MAX_VALID_DEPTH_M = 8.0


class RealSenseCamera:
    """RealSense 파이프라인을 감싸는 얇은 래퍼 - depth+color 동기화 스트림."""

    def __init__(self, width: int = DEFAULT_WIDTH, height: int = DEFAULT_HEIGHT,
                 fps: int = DEFAULT_FPS, use_filters: bool = True,
                 spatial_alpha: float | None = None, spatial_delta: float | None = None,
                 spatial_magnitude: float | None = None,
                 temporal_alpha: float | None = None, temporal_delta: float | None = None,
                 use_spatial: bool = True, use_temporal: bool = True, use_hole_filling: bool = True,
                 hole_filling_mode: int | None = None):
        """
        spatial_*/temporal_* — 5주차 필터 튜닝용 오버라이드. None이면 RealSense SDK
        기본값을 그대로 쓴다. 의미(공식 SDK 옵션):
        - spatial_alpha (0~1, 기본 0.5): 클수록 더 많이 평활화(노이즈↓, 디테일↓)
        - spatial_delta (1~50, 기본 20): 이 값보다 깊이차가 크면 edge로 보고 보존
        - spatial_magnitude (1~5, 기본 2): spatial filter 반복 횟수
        - temporal_alpha (0~1, 기본 0.4): 클수록 이전 프레임 영향을 더 받음(떨림↓, 지연↑)
        - temporal_delta (1~100, 기본 20): 프레임 간 이 값 넘는 변화는 급격한 실제 변화로 보고 그대로 반영
        """
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

        self.configure_filters(
            use_spatial=use_spatial, use_temporal=use_temporal, use_hole_filling=use_hole_filling,
            spatial_alpha=spatial_alpha, spatial_delta=spatial_delta, spatial_magnitude=spatial_magnitude,
            temporal_alpha=temporal_alpha, temporal_delta=temporal_delta, hole_filling_mode=hole_filling_mode,
        )

        self._profile = None

    def configure_filters(self, use_spatial: bool = True, use_temporal: bool = True,
                          use_hole_filling: bool = True,
                          spatial_alpha: float | None = None, spatial_delta: float | None = None,
                          spatial_magnitude: float | None = None,
                          temporal_alpha: float | None = None, temporal_delta: float | None = None,
                          hole_filling_mode: int | None = None) -> None:
        """필터를 (재)구성한다 — 새 필터 객체를 만들므로 temporal filter의 누적 상태도 초기화된다.
        `filters` 모드가 카메라를 다시 켜지 않고 프리셋을 갈아끼울 때 쓴다.
        hole_filling_mode: 0=왼쪽 값으로 채움, 1=최근접(far) 값, 2=최근접(near) 값 (SDK 기본 1).
        """
        self.use_spatial, self.use_temporal, self.use_hole_filling = use_spatial, use_temporal, use_hole_filling
        self._spatial = rs.spatial_filter()
        if spatial_alpha is not None:
            self._spatial.set_option(rs.option.filter_smooth_alpha, spatial_alpha)
        if spatial_delta is not None:
            self._spatial.set_option(rs.option.filter_smooth_delta, spatial_delta)
        if spatial_magnitude is not None:
            self._spatial.set_option(rs.option.filter_magnitude, spatial_magnitude)

        self._temporal = rs.temporal_filter()
        if temporal_alpha is not None:
            self._temporal.set_option(rs.option.filter_smooth_alpha, temporal_alpha)
        if temporal_delta is not None:
            self._temporal.set_option(rs.option.filter_smooth_delta, temporal_delta)

        self._hole_filling = rs.hole_filling_filter()
        if hole_filling_mode is not None:
            self._hole_filling.set_option(rs.option.holes_fill, hole_filling_mode)

    def start(self) -> "RealSenseCamera":
        try:
            self._profile = self._pipeline.start(self._config)
        except RuntimeError as exc:
            raise SystemExit(
                f"RealSense 카메라를 열 수 없습니다: {exc}\n"
                "- USB 3.0/3.1 Type-C 포트에 꽂혀있는지 확인\n"
                "- realsense-viewer로 먼저 인식되는지 확인 (README 참고)"
            ) from exc
        # depth 원시값(uint16)을 미터로 바꾸는 배율 — 전경/배경 분리(get_depth_image_m)에 필요
        self.depth_scale = self._profile.get_device().first_depth_sensor().get_depth_scale()
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
            if self.use_spatial:
                depth_frame = self._spatial.process(depth_frame)
            if self.use_temporal:
                depth_frame = self._temporal.process(depth_frame)
            if self.use_hole_filling:
                depth_frame = self._hole_filling.process(depth_frame)
            depth_frame = depth_frame.as_depth_frame()

        color_image = np.asanyarray(color_frame.get_data())
        timestamp_ms = color_frame.get_timestamp()  # RealSense 프레임 타임스탬프는 기본 ms 단위
        return color_image, depth_frame, timestamp_ms

    @staticmethod
    def get_distance_m(depth_frame, x: int, y: int) -> float:
        """(x, y) 픽셀의 추정 거리(미터)."""
        return float(depth_frame.get_distance(x, y))

    def get_depth_image_m(self, depth_frame) -> np.ndarray:
        """depth_frame 전체를 (color와 같은 해상도의) 미터 단위 2D 배열로 변환한다.
        픽셀 단위로 get_distance()를 매번 호출하면 느려서(848x480=약 40만 번/프레임),
        원시 uint16 배열을 한 번에 numpy로 변환한 뒤 depth_scale을 곱한다. 사람이 둘 이상
        잡힐 때 "카메라에서 가까운 범위만 남기고 나머지는 지우는" 전경 분리에 쓴다
        (src/pose/movenet_keypoints.py의 _RealSenseSource.get_foreground_mask 참고).
        값이 0인 픽셀은 무효(거리 측정 실패) — 호출부에서 min_depth_m > 0으로 두면
        자연히 걸러진다.
        """
        return np.asanyarray(depth_frame.get_data()).astype(np.float32) * self.depth_scale


def _view_mode(args: argparse.Namespace) -> None:
    """
    실시간으로 color 프레임 + 중앙 십자선 거리값 + fps를 화면에 띄운다.
    줄자로 실측하면서 눈으로 오차를 확인하는 용도(project.md 4주차 게이트 중 '거리 정확도').

    --log가 주어지면 's' 키를 눌러 현재 SDK 거리값을 기록하고, 그 순간 줄자로 잰 실제
    거리(cm)를 터미널에 입력하면 오차를 계산해 CSV에 한 줄씩 누적한다(5주차 "거리 정확도
    정밀 재검증" 용도 — project.md 6장 5주차 항목 참고).
    """
    log_path = args.log
    log_file = None
    log_writer = None
    if log_path:
        import csv
        from pathlib import Path
        is_new = not Path(log_path).exists()
        log_file = open(log_path, "a", newline="", encoding="utf-8")
        log_writer = csv.writer(log_file)
        if is_new:
            log_writer.writerow(["timestamp", "sdk_distance_m", "actual_distance_cm", "actual_distance_m", "error_m", "error_pct", "note"])

    cam_kwargs = dict(
        fps=args.fps,
        spatial_alpha=args.spatial_alpha, spatial_delta=args.spatial_delta, spatial_magnitude=args.spatial_magnitude,
        temporal_alpha=args.temporal_alpha, temporal_delta=args.temporal_delta,
    )
    with RealSenseCamera(**cam_kwargs) as cam:
        print("실행 중... 'q'를 누르면 종료합니다. 화면 중앙 십자선까지의 거리가 표시됩니다.")
        if log_path:
            print(f"'s'를 누르면 현재 거리값을 기록합니다 → {log_path}에 저장 (터미널 창에서 실제 줄자 거리(cm)를 입력하라는 안내가 뜹니다).")
        prev_t = time.time()
        fps_smoothed = 0.0
        last_distance = 0.0
        try:
            while True:
                result = cam.read()
                if result is None:
                    continue
                color_image, depth_frame, _ = result

                h, w = color_image.shape[:2]
                cx, cy = w // 2, h // 2
                distance = cam.get_distance_m(depth_frame, cx, cy)
                last_distance = distance

                now = time.time()
                dt = now - prev_t
                prev_t = now
                if dt > 0:
                    fps_smoothed = 0.9 * fps_smoothed + 0.1 * (1.0 / dt)

                is_valid = MIN_VALID_DEPTH_M <= distance <= MAX_VALID_DEPTH_M
                distance_text = f"distance: {distance:.3f} m" if is_valid else "distance: invalid (범위 밖, 0.6~6m 이내로 다시 측정)"
                text_color = (255, 255, 255) if is_valid else (0, 0, 255)

                cv2.drawMarker(color_image, (cx, cy), (0, 255, 170), cv2.MARKER_CROSS, 20, 2)
                cv2.putText(color_image, distance_text, (10, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2, cv2.LINE_AA)
                cv2.putText(color_image, f"fps: {fps_smoothed:.1f}", (10, 54),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                if log_path:
                    cv2.putText(color_image, "'s' = 기록", (10, 84),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 170), 2, cv2.LINE_AA)

                cv2.imshow("RealSense D455 (q to quit)", color_image)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("s") and log_writer is not None:
                    print(f"\n[기록] 현재 SDK 거리값: {last_distance:.3f} m")
                    raw = input("  줄자로 잰 실제 거리(cm)를 입력하고 Enter (건너뛰려면 그냥 Enter): ").strip()
                    if raw:
                        try:
                            actual_cm = float(raw)
                            actual_m = actual_cm / 100.0
                            error_m = last_distance - actual_m
                            error_pct = (error_m / actual_m * 100.0) if actual_m else float("nan")
                            log_writer.writerow([time.time(), f"{last_distance:.4f}", actual_cm, f"{actual_m:.4f}", f"{error_m:.4f}", f"{error_pct:.2f}", ""])
                            log_file.flush()
                            print(f"  저장됨 — 오차 {error_m*100:+.1f}cm ({error_pct:+.1f}%)\n")
                        except ValueError:
                            print("  숫자로 입력해주세요 — 이번 기록은 건너뜁니다.\n")
                    else:
                        print("  건너뜀\n")
        finally:
            cv2.destroyAllWindows()
            if log_file:
                log_file.close()
                print(f"거리 정확도 로그 저장 완료: {log_path}")


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


# `filters` 모드 프리셋: 이름 → configure_filters 인자 (None 값은 SDK 기본)
FILTER_PRESETS: dict[str, dict] = {
    "필터없음": dict(use_spatial=False, use_temporal=False, use_hole_filling=False),
    "SDK기본": dict(),
    "구멍메우기없음": dict(use_hole_filling=False),
    "공간강": dict(spatial_alpha=0.7, spatial_delta=30, spatial_magnitude=3),
    "시간강": dict(temporal_alpha=0.2, temporal_delta=40),
    "시간약": dict(temporal_alpha=0.7, temporal_delta=15),
    "강+구멍없음": dict(use_hole_filling=False, spatial_alpha=0.7, spatial_delta=30,
                    spatial_magnitude=3, temporal_alpha=0.2, temporal_delta=40),
}


def _filters_mode(args: argparse.Namespace) -> None:
    """
    5주차 depth 필터 튜닝: 프리셋별로 같은 자리를 N초씩 찍어 떨림·구멍·튐·fps를 비교한다.
    사용자는 평소 작업 거리에서 가만히 앉아 화면 중앙(십자선)에 가슴이 오도록 맞춘다.
    (지표 정의는 src/capture/depth_filter_metrics.py 참고. 결과는 --out CSV로도 저장.)
    """
    import csv
    from pathlib import Path

    from src.capture.depth_filter_metrics import format_table, frame_stats, summarize

    names = args.presets.split(",") if args.presets else list(FILTER_PRESETS)
    unknown = [n for n in names if n not in FILTER_PRESETS]
    if unknown:
        raise SystemExit(f"알 수 없는 프리셋: {unknown} (가능: {list(FILTER_PRESETS)})")

    rows: list[tuple[str, dict]] = []
    with RealSenseCamera(fps=args.fps) as cam:
        for name in names:
            cam.configure_filters(**FILTER_PRESETS[name])
            print(f"\n[{name}] 준비 — 화면 중앙 십자선에 가슴을 맞추고 가만히 있어주세요. "
                  f"{args.warmup:.0f}초 워밍업 후 {args.seconds:.0f}초 측정 (q로 중단)")
            stats = []
            t0 = time.time()
            measure_start = None
            while True:
                now = time.time()
                if measure_start is None and now - t0 >= args.warmup:
                    measure_start = now
                if measure_start is not None and now - measure_start >= args.seconds:
                    break
                result = cam.read()
                if result is None:
                    continue
                color_image, depth_frame, _ = result
                depth_m = cam.get_depth_image_m(depth_frame)
                if measure_start is not None:
                    stats.append(frame_stats(depth_m, args.patch))
                if args.show:
                    h, w = color_image.shape[:2]
                    half = args.patch // 2
                    cv2.rectangle(color_image, (w // 2 - half, h // 2 - half), (w // 2 + half, h // 2 + half), (0, 255, 170), 2)
                    label = f"{name} {'워밍업' if measure_start is None else '측정중'}"
                    cv2.putText(color_image, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.imshow("depth filter tuning (q to abort)", color_image)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        raise SystemExit("중단했습니다.")
            rows.append((name, summarize(stats, time.time() - measure_start)))
    cv2.destroyAllWindows()

    print("\n=== depth 필터 비교 결과 ===")
    print(format_table(rows))
    print("\n읽는 법: 시간떨림·표면노이즈·튐%는 낮을수록, 패치유효%는 높을수록, fps는 목표(8~10) 이상이면 OK.")
    print("구멍메우기는 유효%를 올리지만 값을 지어내는 것이므로 떨림/튐과 같이 볼 것.")

    if args.out:
        out = Path(args.out)
        is_new = not out.exists()
        with open(out, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if is_new:
                w.writerow(["timestamp", "preset", "note", "frames", "fps", "patch_valid_pct", "frame_valid_pct",
                            "median_depth_m", "temporal_std_mm", "spatial_std_mm", "spike_pct"])
            for name, m in rows:
                if m:
                    w.writerow([time.time(), name, args.note, m["frames"], f"{m['fps']:.2f}",
                                f"{m['patch_valid_pct']:.2f}", f"{m['frame_valid_pct']:.2f}",
                                "" if m["median_depth_m"] is None else f"{m['median_depth_m']:.4f}",
                                "" if m["temporal_std_mm"] is None else f"{m['temporal_std_mm']:.3f}",
                                "" if m["spatial_std_mm"] is None else f"{m['spatial_std_mm']:.3f}",
                                "" if m["spike_pct"] is None else f"{m['spike_pct']:.2f}"])
        print(f"저장: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="RealSense D455 캡처 / 4주차 게이트 측정")
    sub = parser.add_subparsers(dest="mode", required=True)

    view_p = sub.add_parser("view", help="실시간 화면 + 거리·fps 표시 (줄자로 거리 정확도 확인용)")
    view_p.add_argument("--fps", type=int, default=DEFAULT_FPS)
    view_p.add_argument("--log", type=str, default=None, help="거리 정확도 실측 로그 CSV 경로 ('s' 키로 기록, 5주차 정밀 재검증용)")
    view_p.add_argument("--spatial-alpha", type=float, default=None, help="spatial filter smooth_alpha 오버라이드 (0~1, SDK 기본 0.5)")
    view_p.add_argument("--spatial-delta", type=float, default=None, help="spatial filter smooth_delta 오버라이드 (1~50, SDK 기본 20)")
    view_p.add_argument("--spatial-magnitude", type=float, default=None, help="spatial filter 반복 횟수 오버라이드 (1~5, SDK 기본 2)")
    view_p.add_argument("--temporal-alpha", type=float, default=None, help="temporal filter smooth_alpha 오버라이드 (0~1, SDK 기본 0.4)")
    view_p.add_argument("--temporal-delta", type=float, default=None, help="temporal filter smooth_delta 오버라이드 (1~100, SDK 기본 20)")

    gate_p = sub.add_parser("gate", help="N초 동안 fps/프레임타임 스파이크 자동 측정")
    gate_p.add_argument("--seconds", type=float, default=30.0)
    gate_p.add_argument("--fps", type=int, default=DEFAULT_FPS)

    filt_p = sub.add_parser("filters", help="depth 필터 프리셋별 떨림·구멍·fps 비교 (5주차 필터 튜닝)")
    filt_p.add_argument("--fps", type=int, default=DEFAULT_FPS)
    filt_p.add_argument("--seconds", type=float, default=10.0, help="프리셋당 측정 시간(초)")
    filt_p.add_argument("--warmup", type=float, default=2.0, help="프리셋 전환 후 temporal filter 안정화 대기(초)")
    filt_p.add_argument("--patch", type=int, default=41, help="화면 중앙 관심 패치 한 변(px)")
    filt_p.add_argument("--presets", type=str, default=None, help=f"쉼표로 구분한 프리셋 이름 (기본: 전부) — {', '.join(FILTER_PRESETS)}")
    filt_p.add_argument("--out", type=str, default=None, help="결과를 누적할 CSV 경로 (예: data/depth_filter_tuning.csv)")
    filt_p.add_argument("--note", type=str, default="", help="CSV에 남길 조건 메모 (예: '60cm 정면 형광등')")
    filt_p.add_argument("--no-show", dest="show", action="store_false", help="화면 표시 없이 측정만")

    args = parser.parse_args()
    if args.mode == "view":
        _view_mode(args)
    elif args.mode == "gate":
        _gate_mode(args)
    elif args.mode == "filters":
        _filters_mode(args)


if __name__ == "__main__":
    main()
