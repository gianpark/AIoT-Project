#!/usr/bin/env python3
"""
5주차 "라벨링 교차검증" 준비 스크립트 (data_collection_protocol.md 5번).

촬영자가 찍으면서 붙인 파일명의 클래스명이 "1차 라벨"이다(예: p01_normal_003.jpg의
"normal"). 이 스크립트는 전체 사진 중 일부(기본 15%, 프로토콜 기준 10~20%)를 무작위로
뽑아, 파일명(=1차 라벨)을 알 수 없는 임의 이름(rl_001.jpg, rl_002.jpg, ...)으로 복사해
"블라인드 폴더"를 만든다. 촬영에 참여하지 않은 팀원이 이 폴더의 사진만 보고(진짜
파일명도, 라벨도 모르는 상태로) 독립적으로 다시 라벨링하게 하기 위한 것이다.

사용법 (프로젝트 루트에서)
--------------------------
    python scripts/prepare_blind_relabel_sample.py

출력
----
- data/relabel_blind/                         블라인드 사진들 (rl_001.jpg, rl_002.jpg, ...)
- data/relabel_blind/relabel_template.csv     2차 라벨링 담당자가 채워 넣을 양식
                                               (image_filename, guess_label, notes)
- data/relabel_mapping.csv                    정답 매핑표 (random_filename -> 원본 파일명,
                                               1차 라벨) — *2차 라벨링이 끝나기 전까지는
                                               절대 열어보지 않는다*, 끝난 뒤
                                               compare_relabels.py가 채점에 사용

data/relabel_blind/ 안의 사진은 data/raw/와 마찬가지로 개인정보라 .gitignore에 등록돼
있어 git에는 올라가지 않는다. CSV 두 개(매핑·결과)는 메타데이터만 담으므로 git에 올라간다.
"""
from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path

VALID_LABELS = ["normal", "slouch_forward", "slouch_back", "tilt_left", "tilt_right"]


def _parse_label_from_filename(filename: str) -> str | None:
    """{참가자ID}_{클래스}_{일련번호}.jpg 형식에서 클래스명을 뽑는다."""
    stem = filename.rsplit(".", 1)[0]
    for label in sorted(VALID_LABELS, key=len, reverse=True):
        if f"_{label}_" in stem:
            return label
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="라벨링 교차검증용 블라인드 샘플 준비")
    parser.add_argument("--raw-dir", type=str, default="data/raw",
                         help="원본 사진 루트 (참가자별 하위 폴더), 기본 data/raw")
    parser.add_argument("--out-dir", type=str, default="data/relabel_blind",
                         help="블라인드 사진을 복사할 폴더")
    parser.add_argument("--mapping-out", type=str, default="data/relabel_mapping.csv",
                         help="정답 매핑표 저장 경로")
    parser.add_argument("--fraction", type=float, default=0.15,
                         help="전체 중 뽑을 비율, 기본 0.15=15%% (프로토콜 기준 10~20%%)")
    parser.add_argument("--no-stratify", dest="stratify", action="store_false", default=True,
                         help="클래스 비율 맞춰 뽑는 대신 전체에서 순수 무작위로 뽑는다 "
                              "(기본은 클래스별 비율 유지 — 5클래스 전부 샘플에 포함되게 함)")
    parser.add_argument("--seed", type=int, default=None,
                         help="재현 가능한 샘플링을 원하면 지정 (기본은 매번 다르게 뽑힘)")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    all_photos = sorted(raw_dir.rglob("*.jpg"))
    if not all_photos:
        raise SystemExit(f"{raw_dir} 아래에서 사진을 못 찾았습니다.")

    by_label: dict[str, list[Path]] = {}
    unlabeled: list[Path] = []
    for p in all_photos:
        label = _parse_label_from_filename(p.name)
        if label is None:
            unlabeled.append(p)
        else:
            by_label.setdefault(label, []).append(p)
    if unlabeled:
        preview = [p.name for p in unlabeled[:5]]
        print(f"(참고: 파일명에서 클래스를 못 읽은 {len(unlabeled)}장은 샘플링에서 제외됩니다: "
              f"{preview}{'...' if len(unlabeled) > 5 else ''})")

    rng = random.Random(args.seed)
    selected: list[tuple[Path, str]] = []
    if args.stratify:
        for label, photos in by_label.items():
            n = max(1, round(len(photos) * args.fraction))
            n = min(n, len(photos))
            selected.extend((p, label) for p in rng.sample(photos, n))
    else:
        pool = [(p, label) for label, photos in by_label.items() for p in photos]
        n = max(1, round(len(pool) * args.fraction))
        selected = rng.sample(pool, min(n, len(pool)))

    rng.shuffle(selected)  # 클래스 순서로 몰리지 않게 섞는다 (블라인드 효과 보강)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mapping_rows = []
    template_rows = []
    digits = max(3, len(str(len(selected))))
    for i, (src_path, true_label) in enumerate(selected, start=1):
        random_name = f"rl_{i:0{digits}d}.jpg"
        shutil.copy2(src_path, out_dir / random_name)
        mapping_rows.append([random_name, src_path.name, true_label])
        template_rows.append([random_name, "", ""])

    with open(args.mapping_out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["random_filename", "original_filename", "true_label_1st"])
        w.writerows(mapping_rows)

    template_path = out_dir / "relabel_template.csv"
    with open(template_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_filename", "guess_label", "notes"])
        w.writerows(template_rows)

    print(f"총 {len(all_photos)}장 중 {len(selected)}장({len(selected) / len(all_photos) * 100:.1f}%) 샘플링 완료")
    for label in VALID_LABELS:
        cnt = sum(1 for _, l in selected if l == label)
        print(f"  - {label}: {cnt}장")
    print(f"블라인드 사진: {out_dir}/ (rl_001.jpg ~ rl_{len(selected):0{digits}d}.jpg)")
    print(f"채울 양식: {template_path}")
    print(f"정답 매핑(2차 라벨링 끝나기 전까지 절대 열어보지 말 것!): {args.mapping_out}")
    print(f"\n다음 단계: '{out_dir}' 폴더 전체를 촬영에 참여하지 않은 팀원에게 전달 → "
          f"relabel_template.csv의 guess_label 칸에 {VALID_LABELS} 중 하나씩 채우게 함 "
          f"(사진 파일명만 보고 판단, 원본 라벨은 모르는 상태) → "
          f"다 채운 CSV를 scripts/compare_relabels.py로 비교")


if __name__ == "__main__":
    main()
