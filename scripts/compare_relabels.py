#!/usr/bin/env python3
"""
5주차 "라벨링 교차검증" 비교 스크립트 (data_collection_protocol.md 5번).

prepare_blind_relabel_sample.py로 만든 블라인드 샘플을, 촬영에 참여하지 않은 팀원이
독립적으로 재라벨링해서 채운 CSV(relabel_template.csv의 guess_label 칸을 채운 것)와
정답 매핑표(data/relabel_mapping.csv)를 비교해 1차/2차 라벨 일치율을 계산한다.

사용법 (프로젝트 루트에서)
--------------------------
    python scripts/compare_relabels.py --filled data/relabel_blind/relabel_template.csv

출력
----
- 터미널에 전체 일치율, 클래스별 불일치 패턴(혼동 행렬 형태) 출력
- data/relabel_results.csv 에 전체 비교 결과 저장 (원본 파일명, 1차/2차 라벨, 일치 여부, 비고)

프로토콜 기준(data_collection_protocol.md 5번): 불일치율이 체감상 20% 이상이면 클래스
정의나 촬영 지시 문구 자체를 재논의해야 한다 — 사람이 봐도 헷갈리면 분류기도 못 배운다.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter


def main() -> None:
    parser = argparse.ArgumentParser(description="라벨링 교차검증 결과 비교")
    parser.add_argument("--mapping", type=str, default="data/relabel_mapping.csv",
                         help="prepare_blind_relabel_sample.py가 만든 정답 매핑표")
    parser.add_argument("--filled", type=str, required=True,
                         help="팀원이 guess_label을 채운 CSV (relabel_template.csv)")
    parser.add_argument("--out", type=str, default="data/relabel_results.csv",
                         help="상세 비교 결과를 저장할 경로")
    args = parser.parse_args()

    mapping: dict[str, tuple[str, str]] = {}
    with open(args.mapping, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            mapping[row["random_filename"]] = (row["original_filename"], row["true_label_1st"])

    results = []
    missing = []
    with open(args.filled, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fname = row["image_filename"]
            guess = (row.get("guess_label") or "").strip()
            if fname not in mapping:
                continue
            original_filename, first_label = mapping[fname]
            if not guess:
                missing.append(fname)
                continue
            results.append({
                "random_filename": fname,
                "original_filename": original_filename,
                "label_1st": first_label,
                "label_2nd": guess,
                "agree": first_label == guess,
                "note": row.get("notes", ""),
            })

    if missing:
        print(f"(참고: guess_label이 비어 있는 {len(missing)}장은 집계에서 제외됨 — 아직 라벨링 안 됨)")

    total = len(results)
    if total == 0:
        raise SystemExit("비교할 라벨이 없습니다 — 먼저 guess_label 칸을 채워주세요.")

    agree_count = sum(1 for r in results if r["agree"])
    agree_rate = agree_count / total * 100
    disagree_rate = 100 - agree_rate

    print(f"\n전체 {total}장 중 일치 {agree_count}장 — 일치율 {agree_rate:.1f}% (불일치 {disagree_rate:.1f}%)")
    if disagree_rate >= 20:
        print("※ 프로토콜 기준(불일치 20% 이상)에 걸립니다 — 클래스 정의나 촬영 지시 문구 재논의 필요")
    else:
        print("※ 프로토콜 기준(불일치 20% 미만) 통과")

    confusion = Counter((r["label_1st"], r["label_2nd"]) for r in results if not r["agree"])
    if confusion:
        print("\n불일치 패턴 (1차 라벨 -> 2차 라벨 : 건수):")
        for (a, b), cnt in confusion.most_common():
            print(f"  {a} -> {b} : {cnt}건")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["random_filename", "original_filename", "label_1st", "label_2nd", "agree", "note"])
        for r in results:
            w.writerow([r["random_filename"], r["original_filename"], r["label_1st"],
                        r["label_2nd"], r["agree"], r["note"]])
    print(f"\n상세 결과 저장: {args.out} (불일치 건 포함 전체 — 팀 논의 때 참고)")


if __name__ == "__main__":
    main()
