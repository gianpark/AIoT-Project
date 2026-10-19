"""사람 간 차이 대 같은 사람의 회차 간 차이 (서론 한계 3 문장 검증용).

각 (참가자, 자세, 회차) 로그의 특징 중앙값을 한 점으로 보고,
- 회차 간 SD  : 같은 참가자·같은 자세의 회차(_r2, _r3 포함) 중앙값들의 SD를 참가자 평균한 값
- 참가자 간 SD: 참가자별 회차 평균들의 SD
비율 = 참가자 간 SD / 회차 간 SD. 자세별로 계산해 중앙값·범위를 보고한다.

사용: python -m scripts.between_within_variability
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

LOG_DIR = Path("data/side_logs")
FEATURES = ["neck_forward_offset_m", "torso_recline_offset_m", "head_forward_deg",
            "wrist_face_sw", "head_up_m", "chest_depth_m"]
NAME = re.compile(r"side_(p\d+)_(.+?)(?:_r(\d+))?\.csv$")
MIN_ROWS = 21


def load() -> pd.DataFrame:
    rows = []
    for f in sorted(LOG_DIR.glob("side_*.csv")):
        m = NAME.match(f.name)
        if not m:
            continue
        d = pd.read_csv(f)
        if len(d) < MIN_ROWS:
            continue
        rec = {"p": m.group(1), "pose": m.group(2), "round": int(m.group(3) or 1)}
        for k in FEATURES:
            rec[k] = float(pd.to_numeric(d[k], errors="coerce").median()) if k in d else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def main() -> None:
    df = load()
    out = ["# 사람 간 차이 대 회차 간 차이", "",
           f"- 파일 {len(df)}개(참가자 {df.p.nunique()}명, 자세 {df.pose.nunique()}종). 각 파일의 특징 중앙값을 한 점으로 쓴다.",
           "- 회차 간 SD: 같은 참가자·자세의 회차 중앙값 SD를 참가자 평균. 참가자 간 SD: 참가자 평균들의 SD. 비율 = 참가자 간 / 회차 간.",
           "- 자세 11종이 각각 참가자 3명 x 회차 3개이므로 자세별 비율이 11개 나온다. 요약은 그 중앙값과 범위, 비율>1인 자세 수.", ""]
    out += ["| 특징 | 비율 중앙값 | 비율 범위 | 비율>1 자세 수 | 참가자 간 SD 평균 | 회차 간 SD 평균 |", "|---|---|---|---|---|---|"]
    per_pose_tables = []
    for k in FEATURES:
        recs = []
        for pose, g in df.groupby("pose"):
            piv = g.pivot_table(index="p", columns="round", values=k)
            if piv.shape[0] < 2 or piv.shape[1] < 2:
                continue
            within = piv.std(axis=1, ddof=1).mean()
            between = piv.mean(axis=1).std(ddof=1)
            if within and np.isfinite(within) and np.isfinite(between):
                recs.append((pose, between, within, between / within))
        if not recs:
            continue
        r = pd.DataFrame(recs, columns=["pose", "between", "within", "ratio"])
        out.append(f"| {k} | {r.ratio.median():.2f} | {r.ratio.min():.2f}–{r.ratio.max():.2f} | "
                   f"{int((r.ratio > 1).sum())}/{len(r)} | {r.between.mean():.4f} | {r.within.mean():.4f} |")
        per_pose_tables.append((k, r))
    out += ["", "## 자세별 비율", ""]
    for k, r in per_pose_tables:
        out += [f"**{k}**", "", "| 자세 | 참가자 간 SD | 회차 간 SD | 비율 |", "|---|---|---|---|"]
        out += [f"| {x.pose} | {x.between:.4f} | {x.within:.4f} | {x.ratio:.2f} |" for x in r.itertuples()]
        out.append("")
    Path("experiments").mkdir(exist_ok=True)
    Path("experiments/between_within_variability.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out[:16]))


if __name__ == "__main__":
    main()
