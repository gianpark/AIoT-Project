"""공유용 RGB-D 원본 촬영 (MoveNet 없이 저장만 해서 카메라 fps를 그대로 유지한다).

앱(movenet_keypoints --save-raw)은 자세 추정 루프가 프레임을 읽을 때만 저장해서 30 fps 중 일부만 남는다.
이 스크립트는 추정을 하지 않고 color+depth만 저장하므로 다른 팀이 요구한 "전체 프레임"에 가깝게 남긴다.
그래도 빠진 프레임이 있으면 meta.json의 color_frame_number_check에 개수가 기록된다.

사용 예 (저장소 루트, 가상환경, 성능 모드 + 충전기):
    python -m scripts.capture_raw --participant C01 --pose P0 --label normal --take 1
    python -m scripts.capture_raw --participant C01 --pose P1 --label neck_hard --take 1 --duration 20 --countdown 5
    python -m scripts.capture_raw --participant C01 --pose BAD --label chin       (공유 대상 아님)

저장 위치: data/raw_rgbd/<participant>_<pose>_<label>_t<take>_<YYYYMMDD-HHMMSS>/
매 촬영마다 data/raw_rgbd/sessions.csv에 한 줄이 쌓인다(share_map.csv 작성용).
참가자 코드는 공유용 가명(C01 등)을 쓴다. 실명은 어디에도 적지 않는다.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import time
from pathlib import Path

import cv2

from src.capture.raw_recorder import RawRgbdRecorder
from src.capture.realsense_capture import RealSenseCamera

POSES = {
    "P0": "정상",
    "P1": "거북목",
    "P2": "상체 후방",
    "P3": "좌측 기울임 (목 내밀지 않음)",
    "P4": "우측 기울임 (목 내밀지 않음)",
    "P5": "전방",
}
SESSION_COLS = ["source_dir", "participant", "pose", "pose_name", "label", "take", "capture_date", "start_local",
                "frames", "saved_fps", "missing_frames", "lr_convention", "note"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--participant", required=True, help="공유용 가명 코드 (예: C01)")
    ap.add_argument("--pose", required=True, help="P0~P5, 또는 우리 조 전용 자세 이름(공유 대상 아님)")
    ap.add_argument("--label", default="", help="원래 촬영 라벨 (예: neck_light, neck_hard, chin)")
    ap.add_argument("--take", type=int, default=1)
    ap.add_argument("--duration", type=float, default=20.0, help="저장 시간(초)")
    ap.add_argument("--countdown", type=float, default=5.0, help="저장 전 대기(초), 이 동안은 저장 안 함")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=848)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--color-format", choices=["png", "jpg"], default="png")
    ap.add_argument("--out-root", default="data/raw_rgbd")
    ap.add_argument("--lr-convention", default="참가자 본인 기준",
                    help="P3/P4의 왼쪽·오른쪽 기준 (다른 팀과 합의한 대로)")
    ap.add_argument("--note", default="", help="자유 메모 (실명 금지)")
    ap.add_argument("--no-preview", action="store_true", help="미리보기 창 끄기")
    args = ap.parse_args()

    pose = args.pose.strip()
    pose_name = POSES.get(pose.upper(), "")
    if pose.upper() in POSES:
        pose = pose.upper()
    else:
        print(f"주의: '{pose}'는 공통 P0~P5가 아니므로 공유 패키지에서 자동으로 빠집니다.")

    start = dt.datetime.now()
    label = args.label.strip()
    name = f"{args.participant}_{pose}{'_' + label if label else ''}_t{args.take}_{start:%Y%m%d-%H%M%S}"
    out_root = Path(args.out_root)
    out_dir = out_root / name

    cam = RealSenseCamera(width=args.width, height=args.height, fps=args.fps, use_filters=False).start()
    rec = RawRgbdRecorder(out_dir, color_format=args.color_format)
    rec.cam_info.update(cam.cam_info())
    rec.session_info.update({"participant": args.participant, "pose": pose, "pose_name": pose_name, "label": label,
                             "take": args.take, "lr_convention": args.lr_convention, "note": args.note,
                             "countdown_s": args.countdown, "duration_s_requested": args.duration,
                             "capture_script": "scripts/capture_raw.py (MoveNet 없음)"})
    cam.raw_sink = rec
    preview = not args.no_preview
    print(f"[{name}] {pose} {pose_name} — {args.countdown:.0f}초 뒤 {args.duration:.0f}초 저장")

    t_start = time.monotonic()
    shown = 0
    aborted = False
    try:
        while True:
            el = time.monotonic() - t_start
            in_countdown = el < args.countdown
            rec.active = not in_countdown
            if el >= args.countdown + args.duration:
                break
            result = cam.read()
            if result is None:
                continue
            if preview:
                shown += 1
                if shown % 3 == 0:  # 미리보기는 3프레임에 1번만 그려 저장을 방해하지 않는다
                    img = result[0].copy()
                    msg = (f"COUNTDOWN {args.countdown - el:.0f}" if in_countdown
                           else f"REC {el - args.countdown:4.1f}/{args.duration:.0f}s  {pose}")
                    cv2.putText(img, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (0, 0, 255) if not in_countdown else (0, 200, 255), 2)
                    cv2.imshow("capture_raw (q = stop)", img)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        aborted = True
                        break
    finally:
        rec.active = False
        cam.stop()
        if preview:
            cv2.destroyAllWindows()
        n = rec.close()

    import json
    meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
    miss = meta["color_frame_number_check"]["missing"]
    fps = meta["saved_fps"]
    print(f"저장 완료: {out_dir}  프레임 {n}개, 실제 저장 fps {fps:.2f}" if fps else f"저장 완료: {out_dir}  프레임 {n}개")
    print(f"카메라 프레임 번호 기준 누락 {miss}개, 쓰기 대기열 가득 참 {meta['writer_queue_full_count']}회")
    if aborted:
        print("q로 중단됨 — 이 촬영은 다시 찍는 것이 좋습니다.")

    log = out_root / "sessions.csv"
    new = not log.exists()
    with log.open("a", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(SESSION_COLS)
        w.writerow([out_dir.as_posix(), args.participant, pose, pose_name, label, args.take, start.date().isoformat(),
                    start.strftime("%H:%M:%S"), n, f"{fps:.2f}" if fps else "", miss, args.lr_convention,
                    ("중단됨 " if aborted else "") + args.note])
    return 3 if aborted else 0


if __name__ == "__main__":
    raise SystemExit(main())
