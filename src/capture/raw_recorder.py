"""RGB-D 원본 프레임 저장기 (IRB 계획서: 승인 후 촬영분의 RGB·Depth 원본을 타 팀에 제공하기 위함).

저장 형식 (한 자세 = 한 폴더):
    <out_dir>/frames.csv          index, timestamp_ms, color_file, depth_file
    <out_dir>/color/000001.jpg    BGR, JPEG 품질 95
    <out_dir>/depth/000001.png    uint16 PNG(1단위 = depth_scale 미터), color에 정렬된 원본 depth(필터 적용 전)
    <out_dir>/meta.json           해상도·fps·depth_scale·color 내부 파라미터·프레임 수

depth는 공간/시간/홀필링 필터를 거치기 전의 값을 저장한다 — 받는 팀이 자기 방식으로 처리할 수 있게 하기 위함.
쓰기는 별도 스레드가 하고, 대기열이 가득 차면 촬영 루프를 잠깐 멈춰서라도 프레임을 버리지 않는다.
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


class RawRgbdRecorder:
    JPEG_QUALITY = 95

    def __init__(self, out_dir: str | Path, cam_info: dict | None = None, max_queue: int = 240):
        self.out_dir = Path(out_dir)
        if (self.out_dir / "frames.csv").exists():
            raise SystemExit(f"{self.out_dir} 에 이미 저장된 원본이 있습니다 — 다른 폴더를 쓰세요(덮어쓰지 않음).")
        (self.out_dir / "color").mkdir(parents=True, exist_ok=True)
        (self.out_dir / "depth").mkdir(parents=True, exist_ok=True)
        self.cam_info = dict(cam_info or {})
        self.active = False
        self.count = 0
        self._q: queue.Queue = queue.Queue(maxsize=max_queue)
        self._rows: list[tuple] = []
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        self._t0 = time.time()

    def put(self, timestamp_ms: float, color_bgr: np.ndarray, depth_u16: np.ndarray) -> None:
        if not self.active:
            return
        self.count += 1
        # 카메라 버퍼가 재사용될 수 있으므로 복사해서 넘긴다
        self._q.put((self.count, timestamp_ms, color_bgr.copy(), depth_u16.copy()))

    def _worker(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                self._q.task_done()
                return
            idx, ts, color, depth = item
            cname, dname = f"color/{idx:06d}.jpg", f"depth/{idx:06d}.png"
            cv2.imwrite(str(self.out_dir / cname), color, [cv2.IMWRITE_JPEG_QUALITY, self.JPEG_QUALITY])
            cv2.imwrite(str(self.out_dir / dname), depth.astype(np.uint16))
            self._rows.append((idx, f"{ts:.3f}", cname, dname))
            self._q.task_done()

    def close(self) -> int:
        self.active = False
        self._q.put(None)
        self._thread.join()
        with open(self.out_dir / "frames.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["index", "timestamp_ms", "color_file", "depth_file"])
            w.writerows(sorted(self._rows))
        meta = dict(self.cam_info)
        meta.update({
            "frames": len(self._rows),
            "depth_png": "uint16, value * depth_scale = meters, aligned to color, before spatial/temporal/hole-filling filters",
            "color_jpeg_quality": self.JPEG_QUALITY,
            "recorded_at_unix": self._t0,
        })
        with open(self.out_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        return len(self._rows)
