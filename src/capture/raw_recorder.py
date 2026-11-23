"""RGB-D 원본 프레임 저장기 (IRB 계획서: 승인 후 촬영분의 RGB·Depth 원본을 타 팀에 제공하기 위함).

저장 형식 (한 자세 = 한 폴더):
    <out_dir>/frames.csv          프레임별 대응표 (아래 열)
    <out_dir>/color/000001.png    BGR, 무손실 PNG (color_format="jpg"이면 JPEG 품질 95)
    <out_dir>/depth/000001.png    uint16 PNG(1단위 = depth_scale 미터), color에 정렬된 원본 depth(필터 적용 전)
    <out_dir>/meta.json           카메라 정보(해상도·fps·depth_scale·내부/외부 파라미터·왜곡계수),
                                  실제 저장 fps, 프레임 번호 누락·중복, 무효값 규격, 처리 이력

frames.csv 열:
    index               저장 순번(1부터)
    timestamp_ms        color 타임스탬프 (이전 버전과 호환용, color_timestamp_ms와 같음)
    color_frame_number  카메라가 붙인 color 프레임 번호 — 불연속이면 그 사이 프레임이 저장되지 않은 것
    depth_frame_number  카메라가 붙인 depth 프레임 번호
    color_timestamp_ms  color 프레임 타임스탬프(ms)
    depth_timestamp_ms  depth 프레임 타임스탬프(ms)
    timestamp_domain    타임스탬프 기준 (예: global_time, hardware_clock, system_time)
    host_time_ms        노트북이 프레임을 받은 시각(유닉스 ms)
    color_file, depth_file

depth는 공간/시간/홀필링 필터를 거치기 전의 값을 저장한다 — 받는 팀이 자기 방식으로 처리할 수 있게 하기 위함.
쓰기는 여러 스레드가 하고, 대기열이 가득 차면 촬영 루프를 잠깐 멈춘다(이때 카메라 쪽에서 빠진 프레임은
color_frame_number 불연속으로 meta.json에 기록된다).
`active`가 False인 동안(예: 카운트다운)에는 저장하지 않는다.
"""
from __future__ import annotations

import csv
import json
import queue
import threading
import time
from pathlib import Path

import cv2
import numpy as np

FRAME_COLUMNS = ["index", "timestamp_ms", "color_frame_number", "depth_frame_number",
                 "color_timestamp_ms", "depth_timestamp_ms", "timestamp_domain", "host_time_ms",
                 "color_file", "depth_file"]


def frame_number_stats(numbers: list[int]) -> dict:
    """카메라 프레임 번호 목록에서 누락(건너뛴 번호 수)·중복·역행을 센다."""
    nums = [n for n in numbers if n is not None]
    if len(nums) < 2:
        return {"checked": len(nums), "missing": 0, "duplicates": 0, "backwards": 0, "gaps": []}
    missing = dup = back = 0
    gaps = []
    for prev, cur in zip(nums, nums[1:]):
        d = cur - prev
        if d == 0:
            dup += 1
        elif d < 0:
            back += 1
        elif d > 1:
            missing += d - 1
            if len(gaps) < 50:
                gaps.append([prev, cur])
    return {"checked": len(nums), "missing": missing, "duplicates": dup, "backwards": back,
            "gaps": gaps, "expected_if_none_missing": nums[-1] - nums[0] + 1}


class RawRgbdRecorder:
    JPEG_QUALITY = 95
    PNG_COMPRESSION = 1  # 0~9, 낮을수록 빠름(무손실은 동일)

    def __init__(self, out_dir: str | Path, cam_info: dict | None = None, max_queue: int = 240,
                 color_format: str = "png", writers: int = 4):
        if color_format not in ("png", "jpg"):
            raise ValueError("color_format은 png 또는 jpg")
        self.out_dir = Path(out_dir)
        if (self.out_dir / "frames.csv").exists():
            raise SystemExit(f"{self.out_dir} 에 이미 저장된 원본이 있습니다 — 다른 폴더를 쓰세요(덮어쓰지 않음).")
        (self.out_dir / "color").mkdir(parents=True, exist_ok=True)
        (self.out_dir / "depth").mkdir(parents=True, exist_ok=True)
        self.cam_info = dict(cam_info or {})
        self.session_info: dict = {}
        self.color_format = color_format
        self.active = False
        self.count = 0
        self._q: queue.Queue = queue.Queue(maxsize=max_queue)
        self._rows: list[tuple] = []
        self._rows_lock = threading.Lock()
        self._errors: list[str] = []
        self._threads = [threading.Thread(target=self._worker, daemon=True) for _ in range(max(1, writers))]
        for t in self._threads:
            t.start()
        self._t0 = time.time()
        self._blocked_puts = 0

    def put(self, timestamp_ms: float, color_bgr: np.ndarray, depth_u16: np.ndarray,
            extra: dict | None = None) -> None:
        """extra: color_frame_number, depth_frame_number, depth_timestamp_ms, timestamp_domain (없으면 빈 칸)."""
        if not self.active:
            return
        self.count += 1
        e = dict(extra or {})
        e.setdefault("host_time_ms", time.time() * 1000.0)
        item = (self.count, timestamp_ms, color_bgr.copy(), depth_u16.copy(), e)  # 카메라 버퍼 재사용 대비 복사
        try:
            self._q.put_nowait(item)
        except queue.Full:
            self._blocked_puts += 1
            self._q.put(item)

    def _worker(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                self._q.task_done()
                return
            idx, ts, color, depth, e = item
            try:
                if self.color_format == "png":
                    cname = f"color/{idx:06d}.png"
                    ok_c = cv2.imwrite(str(self.out_dir / cname), color, [cv2.IMWRITE_PNG_COMPRESSION, self.PNG_COMPRESSION])
                else:
                    cname = f"color/{idx:06d}.jpg"
                    ok_c = cv2.imwrite(str(self.out_dir / cname), color, [cv2.IMWRITE_JPEG_QUALITY, self.JPEG_QUALITY])
                dname = f"depth/{idx:06d}.png"
                ok_d = cv2.imwrite(str(self.out_dir / dname), depth.astype(np.uint16),
                                   [cv2.IMWRITE_PNG_COMPRESSION, self.PNG_COMPRESSION])
                if not (ok_c and ok_d):
                    raise OSError("imwrite 실패")
                row = (idx, f"{ts:.3f}", e.get("color_frame_number", ""), e.get("depth_frame_number", ""),
                       f"{ts:.3f}", _fmt(e.get("depth_timestamp_ms")), e.get("timestamp_domain", ""),
                       _fmt(e.get("host_time_ms")), cname, dname)
                with self._rows_lock:
                    self._rows.append(row)
            except Exception as exc:  # noqa: BLE001 - 한 프레임 실패로 촬영 전체를 멈추지 않는다
                with self._rows_lock:
                    self._errors.append(f"{idx}: {exc}")
            finally:
                self._q.task_done()

    def close(self) -> int:
        self.active = False
        for _ in self._threads:
            self._q.put(None)
        for t in self._threads:
            t.join()
        rows = sorted(self._rows)
        with open(self.out_dir / "frames.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(FRAME_COLUMNS)
            w.writerows(rows)

        color_ts = [float(r[4]) for r in rows]
        saved_fps = None
        duration_s = None
        if len(color_ts) >= 2 and color_ts[-1] > color_ts[0]:
            duration_s = (color_ts[-1] - color_ts[0]) / 1000.0
            saved_fps = (len(color_ts) - 1) / duration_s
        sync = [abs(float(r[4]) - float(r[5])) for r in rows if r[5] != ""]
        cnums = [int(r[2]) for r in rows if r[2] != ""]

        meta = dict(self.cam_info)
        meta.update(self.session_info)
        meta.update({
            "frames": len(rows),
            "write_errors": self._errors[:50],
            "duration_s": duration_s,
            "saved_fps": saved_fps,
            "color_frame_number_check": frame_number_stats(cnums),
            "rgb_depth_timestamp_diff_ms": ({"mean": float(np.mean(sync)), "max": float(np.max(sync)),
                                             "first": sync[0]} if sync else None),
            "writer_queue_full_count": self._blocked_puts,
            "segment": ({"start_index": rows[0][0], "end_index": rows[-1][0],
                         "start_color_frame_number": rows[0][2], "end_color_frame_number": rows[-1][2],
                         "start_color_timestamp_ms": float(rows[0][4]), "end_color_timestamp_ms": float(rows[-1][4]),
                         "start_host_time_ms": rows[0][7], "end_host_time_ms": rows[-1][7],
                         "note": "폴더 전체가 한 자세의 지시 구간(카운트다운 제외). 안정 구간은 영상 검토로 따로 정한다."}
                        if rows else None),
            "color_file": ("BGR 8bit PNG, lossless" if self.color_format == "png"
                           else f"BGR 8bit JPEG, quality {self.JPEG_QUALITY} (lossy)"),
            "depth_png": "uint16, value * depth_scale = meters (Z distance along the camera axis), "
                         "aligned to color (color intrinsics apply), before spatial/temporal/hole-filling filters",
            "depth_invalid_value": 0,
            "depth_invalid_note": "0 = 스테레오 매칭 실패 또는 범위 밖(무효). 필터를 쓰지 않으므로 65535 같은 채움값은 생기지 않는다.",
            "depth_valid_range_m_used_in_our_app": [0.3, 8.0],
            "depth_valid_range_note": "D455 권장 범위는 약 0.6~6 m. 우리 앱은 0.3~8 m 밖을 무효로 취급했다.",
            "processing": "리사이즈·크롭 없음, depth→color 정렬(rs.align)만 적용, depth 필터 없음, 프레임 제거 없음",
            "recorded_at_unix": self._t0,
            "recorded_at_local": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self._t0)),
        })
        with open(self.out_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        return len(rows)


def _fmt(v) -> str:
    return "" if v is None else f"{float(v):.3f}"
