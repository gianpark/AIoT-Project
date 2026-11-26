"""11/26 새 참가자(p04~p07)를 더한 경고 단위 평가. eval_alert_level.evaluate를 그대로 쓴다.

- 데이터: 환경 1 p01~p03(data/side_logs, 자세당 3회, 좌우 lean_neck 포함) + p04~p07(data/side_1126, 정상 3회·나머지 8자세 1회).
- 제외: 파일 이름에 _bad(자세를 잘못 취해 다시 찍음), framing, framecheck가 들어간 녹화.
- 새 참가자에게 좌우 lean_neck이 없으므로 명확한 자세 검출은 좌우 lean_neck을 뺀 값으로 비교한다.
- 집계: (a) 녹화 단위로 합친 값(기존 표와 같은 방식, 자세별 평균의 평균), (b) 사람별 값을 낸 뒤 사람 평균.

사용: python -m scripts.eval_1126 [--out experiments/eval_1126.md]
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

from scripts.eval_alert_level import BORDER, LEAN, evaluate, load

OLD, NEW = ["p01", "p02", "p03"], ["p04", "p05", "p06", "p07"]


def load_all() -> pd.DataFrame:
    a = load("data/side_logs", OLD)
    b = load("data/side_1126", NEW)
    b = b[~b.pose.str.contains("_bad|framing|framecheck")]
    return pd.concat([a, b], ignore_index=True)


def summ(R: pd.DataFrame) -> dict:
    nm = R[R.pose == "normal"]
    bad = R[(R.pose != "normal") & ~R.pose.isin(LEAN)]
    clear, border = bad[~bad.pose.isin(BORDER)], bad[bad.pose.isin(BORDER)]
    f = lambda x: x.groupby("pose").posture.mean().mean() * 100 if len(x) else float("nan")
    return dict(fa=nm.posture.mean() * 100, clear=f(clear), border=f(border),
                correct=clear[clear.pose != "back_neck"].groupby("pose").correct.mean().mean() * 100)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/eval_1126.md")
    a = ap.parse_args()
    D = load_all()
    L = ["# 경고 단위 평가: p01~p07 (환경 1 + 11/26 촬영)", "",
         "- 5초 간격 샘플, 자세 경고 연속 3회, 위상 10가지 평균. 녹화 하나(20초)에서 자세 경고가 한 번이라도 울렸는지를 센다.",
         "- 보정: 사람별 가장 이른 normal 녹화의 처음 150프레임으로 neck_forward_offset을 REF_NF로 이동(eval_alert_level과 같음). 그 150프레임은 평가에서 뺀다.",
         "- 명확한 자세 = neck_hard, forward, slouch, back_hard, back_neck, chin. 경계 자세 = neck_light, back_light. 좌우 lean_neck은 새 참가자에게 없어 제외.",
         "- 종류 일치 = 명확한 자세 중 기대 종류가 있는 자세(back_neck 제외)에서 기대 종류로 울린 비율.",
         "- _bad·framing·framecheck 녹화는 제외.", "",
         "녹화 수: " + ", ".join(f"{p} {g.groupby(['pose','rep']).ngroups}" for p, g in D.groupby("p")), ""]
    groups = [("p01~p03 (기존)", OLD), ("p04~p07 (새 참가자)", NEW), ("p01~p07 전체", OLD + NEW),
              ("p01~p07 중 p06 제외", [p for p in OLD + NEW if p != "p06"])]
    for cal in (False, True):
        R = evaluate(D, cal)
        lab = "보정" if cal else "보정 없음"
        L += [f"## {lab}", "", "### 묶음별 (녹화 단위로 합침)", "",
              "| 묶음 | 정상 오경고(%) | 명확한 자세 검출(%) | 경계 자세 검출(%) | 종류 일치(%) |", "|---|---|---|---|---|"]
        for name, ps in groups:
            s = summ(R[R.p.isin(ps)])
            L.append(f"| {name} | {s['fa']:.0f} | {s['clear']:.0f} | {s['border']:.0f} | {s['correct']:.0f} |")
        L += ["", "### 사람별", "", "| 사람 | 정상 오경고(%) | 명확한 자세 검출(%) | 경계 자세 검출(%) | 종류 일치(%) |", "|---|---|---|---|---|"]
        per = {}
        for p in OLD + NEW:
            per[p] = s = summ(R[R.p == p])
            L.append(f"| {p} | {s['fa']:.0f} | {s['clear']:.0f} | {s['border']:.0f} | {s['correct']:.0f} |")
        P = pd.DataFrame(per).T
        L += ["", f"- 사람 평균(7명): 오경고 {P.fa.mean():.0f}%, 명확한 자세 {P.clear.mean():.0f}% (최소 {P.clear.min():.0f}, 최대 {P.clear.max():.0f}), 경계 자세 {P.border.mean():.0f}%", ""]
        L += ["### 자세별 경고 발생률(%), 사람별", ""]
        T = R[~R.pose.isin(LEAN)].groupby(["pose", "p"]).posture.mean().unstack() * 100
        L += ["| 자세 | " + " | ".join(T.columns) + " |", "|---" * (len(T.columns) + 1) + "|"]
        for pose, row in T.iterrows():
            L.append(f"| {pose} | " + " | ".join("-" if pd.isna(v) else f"{v:.0f}" for v in row) + " |")
        L.append("")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
