"""한 참가자의 IRB 촬영 자세 11종을 순서대로 찍는다 (자세마다 scripts.capture_raw를 실행).

순서는 IRB 계획서 기준: 정상(P0)을 맨 앞에 찍어 개인 기준값(약 150프레임)으로 쓰고, 마지막에 정상을 한 번 더 찍는다.
자세마다 Enter를 누르면 5초 준비 후 20초 녹화한다.
    Enter = 촬영 / s = 이 자세 건너뛰기 / r = 방금 자세 다시 찍기(회차+1) / q = 종료

사용 (저장소 루트, 가상환경, 성능 모드 + 충전기):
    python -m scripts.capture_session --participant C01
    python -m scripts.capture_session --participant C01 --only P0 P1     # 일부만
    python -m scripts.capture_session --participant C01 --sample          # 5초 샘플 1개(P0)만
"""
from __future__ import annotations

import argparse
import subprocess
import sys

# (공통 코드, 라벨, 참가자에게 읽어 줄 안내)
POSES = [
    ("P0", "normal", "정상: 등받이에 등을 대고 바르게 앉아 모니터를 봅니다. (개인 기준값으로 쓰는 구간)"),
    ("P1", "neck_light", "거북목 약: 목을 앞으로 살짝 내밉니다. 몸통은 그대로."),
    ("P1", "neck_hard", "거북목 강: 목을 앞으로 확실히 내밉니다. 무리하지 않는 범위."),
    ("P2", "back_light", "상체 후방 약: 등받이 쪽으로 살짝 기댑니다."),
    ("P2", "back_hard", "상체 후방 강: 등받이 쪽으로 확실히 기댑니다."),
    ("P5", "forward", "전방: 상체 전체를 모니터 쪽으로 숙입니다."),
    ("P3", "tilt_left", "좌측 기울임: 목은 내밀지 말고 상체를 본인 왼쪽으로 기울입니다."),
    ("P4", "tilt_right", "우측 기울임: 목은 내밀지 말고 상체를 본인 오른쪽으로 기울입니다."),
    ("BAD", "chin", "턱 괴기: 한 손으로 턱을 받칩니다. (공유 대상 아님)"),
    ("BAD", "slouch", "허리 구부정: 엉덩이는 그대로 두고 등을 말아 허리를 구부립니다. (공유 대상 아님)"),
    ("BAD", "back_neck", "뒤로 기대며 고개 내밈: 등받이에 기대면서 고개를 앞으로 내밉니다. (공유 대상 아님)"),
    ("P0", "normal", "정상(마무리): 처음처럼 바르게 앉습니다."),
]


def run(participant, code, label, take, args) -> int:
    cmd = [sys.executable, "-m", "scripts.capture_raw", "--participant", participant, "--pose", code,
           "--label", label, "--take", str(take), "--duration", str(args.duration),
           "--countdown", str(args.countdown)]
    if args.no_preview:
        cmd.append("--no-preview")
    if args.color_format != "png":
        cmd += ["--color-format", args.color_format]
    return subprocess.call(cmd)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--participant", required=True, help="공유용 가명 코드 (예: C01)")
    ap.add_argument("--only", nargs="*", help="이 코드나 라벨만 찍기 (예: P0 P1 chin)")
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--countdown", type=float, default=5.0)
    ap.add_argument("--color-format", choices=["png", "jpg"], default="png")
    ap.add_argument("--no-preview", action="store_true")
    ap.add_argument("--sample", action="store_true", help="A팀 확인용 5초 샘플(P0) 1개만 찍기")
    args = ap.parse_args()

    if args.sample:
        args.duration = 5.0
        return run(args.participant, "P0", "sample", 1, args)

    poses = POSES
    if args.only:
        want = {w.lower() for w in args.only}
        poses = [p for p in POSES if p[0].lower() in want or p[1].lower() in want]
    takes: dict = {}
    done, skipped, failed = [], [], []
    i = 0
    last = None
    while i < len(poses):
        code, label, guide = poses[i]
        print(f"\n[{i + 1}/{len(poses)}] {code} {label}\n  {guide}")
        ans = input("  Enter=촬영, s=건너뛰기, r=방금 자세 다시, q=종료 > ").strip().lower()
        if ans == "q":
            break
        if ans == "s":
            skipped.append(f"{code}/{label}")
            i += 1
            continue
        if ans == "r":
            if last is None:
                print("  다시 찍을 자세가 없습니다.")
                continue
            code, label = last
        key = (code, label)
        takes[key] = takes.get(key, 0) + 1
        rc = run(args.participant, code, label, takes[key], args)
        tag = f"{code}/{label}/t{takes[key]}"
        (done if rc == 0 else failed).append(tag)
        if rc != 0:
            print("  이 촬영은 중단되거나 실패했습니다. r로 다시 찍을 수 있습니다.")
        last = key
        if ans != "r":
            i += 1
    print("\n완료:", ", ".join(done) or "없음")
    if skipped:
        print("건너뜀:", ", ".join(skipped))
    if failed:
        print("중단·실패:", ", ".join(failed))
    print("각 촬영의 저장 fps와 누락 수는 data/raw_rgbd/sessions.csv에 있습니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
