"""좌우 특징 3개와 세션 시작 보정이 분류 성능에 주는 영향을 보는 예비 실험(11/5~11/6).

모드(--mode):
  lopo       같은 환경 3명 교차검증. 기준 9개 / +좌우 3개 / +좌우 보정 / +전체 보정
  cross_env  환경 1 학습→환경 2 평가, 환경 2 학습→환경 1 평가(평가 대상 사람은 학습에서 제외)
  binary     정상/나쁨 이진 분류(같은 환경 교차, 환경 1→환경 2), seed 3개
  split      7-class에서 좌우 두 클래스를 뺀 평균과 앞뒤 5개를 다시 학습한 값 비교, bad_posture 세부 자세 분포,
             라벨별 neck_forward_offset 중앙값

공통 설정:
  - 특징: rf_lopo_7class.SETS["no_hip"] 9개. 좌우 특징 3개 = nose_rx, ear_mid_rx, shoulder_tilt_deg(±180° 뒤집힘 제거).
  - 보정: 사람(환경)별 normal 중 가장 이른 150프레임(약 5초)의 중앙값을 모든 특징에서 뺀다.
    그 150프레임은 모든 조건에서 학습·평가에서 제외한다(평가할 프레임으로 기준을 만들지 않기 위해).
  - RF 300그루, class_weight=balanced, 결측은 학습 fold 중앙값 대치+결측 플래그(Mode A).
  - 환경 1의 p01은 구도를 새로 맞춰 다시 찍은 --p01-dir 로그(좌표 포함)를 쓴다. 이 로그는 p02, p03과 구도가 달라
    선우의 7-class 수치와 직접 비교하면 안 된다(몸통 길이 비 0.71 대 1.05~1.14).
  - 참가자 3명(환경 2는 2명), 자세당 1~3회 녹화의 예비 결과다.

사용: python -m scripts.lateral_study --mode lopo [--seeds 5] [--p01-dir data/side_p01_v2]
출력: experiments/lateral_study_<mode>.md
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from scripts.replay_judge_on_logs import load
from scripts.rf_2d_baseline_7class import CLASS_ORDER, LABEL_MAPPING, prepare_mode_a
from scripts.rf_lopo_7class import SETS

CALIB_FRAMES = 150
LAT = ["nose_rx", "ear_mid_rx", "shoulder_tilt_deg"]
BASE = list(SETS["no_hip"])


def build(p01_dir: str, with_env2: bool) -> pd.DataFrame:
    """환경 1(p02, p03 기존 로그 + p01 새 로그), 필요하면 환경 2를 합쳐 좌우 특징을 계산한다."""
    from src.features.features_2d import features_2d

    a = load("data/side_logs")
    a = a[a["p"].astype(str).isin(["p02", "p03"])]
    b = load(p01_dir)
    b = b[b["p"].astype(str) == "p01"]
    a["env"], b["env"] = 1, 1
    parts = [a, b]
    if with_env2:
        e = load("data/side_env2")
        e["env"] = 2
        parts.append(e)
    d = pd.concat(parts, ignore_index=True)
    d["pid"] = d["p"].astype(str) + "_e" + d["env"].astype(str)
    d["raw"] = d["pose"]
    d["pose"] = d["pose"].map(LABEL_MAPPING)
    d = d[d["nose_x"].notna() & d["pose"].notna()].reset_index(drop=True)
    d = pd.concat([d, features_2d(d)], axis=1)
    d["ear_mid_rx"] = d[["left_ear_rx", "right_ear_rx"]].mean(axis=1)
    d["shoulder_tilt_deg"] = ((d["shoulder_slope_2d"] + 90) % 180) - 90
    return d


def baseline_frames(d: pd.DataFrame) -> list[int]:
    bl: list[int] = []
    for k in d["pid"].unique():
        bl += list(d[(d.pid == k) & (d.pose == "normal")].sort_values("timestamp").index[:CALIB_FRAMES])
    return bl


def calibrate(d: pd.DataFrame, bl: list[int], cols: list[str]) -> pd.DataFrame:
    d = d.copy()
    for k in d["pid"].unique():
        idx = [i for i in bl if d.loc[i, "pid"] == k]
        d.loc[d.pid == k, cols] = d.loc[d.pid == k, cols] - d.loc[idx, cols].median()
    return d


def fit_predict(tr, te, cols, seed, label_col="pose"):
    xtr, xte = prepare_mode_a(tr, te, cols)
    m = RandomForestClassifier(300, class_weight="balanced", random_state=seed, n_jobs=-1).fit(xtr, tr[label_col])
    return m.predict(xte)


def prepare_sets(d, bl):
    drop = lambda x: x.drop(index=bl).reset_index(drop=True)
    d0 = drop(d)
    d_lat = drop(calibrate(d, bl, LAT))
    d_all = drop(calibrate(d, bl, BASE + LAT))
    return d0, d_lat, d_all


def md_table(df: pd.DataFrame, fmt="{:.3f}") -> str:
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "---|" * len(df.columns)
    rows = ["| " + " | ".join(fmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r) + " |"
            for r in df.itertuples(index=False)]
    return "\n".join([head, sep] + rows)


def mean_std(R: pd.DataFrame, by: str, cols: list[str]) -> pd.DataFrame:
    g = R.groupby(by, sort=False)[cols].agg(["mean", "std"])
    out = pd.DataFrame({by: g.index})
    for c in cols:
        out[c] = [f"{m:.3f} ({s:.3f})" for m, s in zip(g[c]["mean"], g[c]["std"])]
    return out


def mode_lopo(a):
    d = build(a.p01_dir, False)
    bl = baseline_frames(d)
    d0, d_lat, d_all = prepare_sets(d, bl)
    sets = [("기준 9개", d0, BASE), ("+좌우 3개", d0, BASE + LAT),
            ("+좌우 3개, 좌우만 보정", d_lat, BASE + LAT), ("+좌우 3개, 전체 보정", d_all, BASE + LAT)]
    rows = []
    for name, D, cols in sets:
        for seed in range(a.seeds):
            pr, tr_, fm = [], [], []
            for ho in sorted(D.p.unique()):
                tr, te = D[D.p != ho], D[D.p == ho]
                p = fit_predict(tr, te, cols, seed)
                pr += list(p)
                tr_ += list(te.pose)
                fm.append(f1_score(te.pose, p, labels=CLASS_ORDER, average="macro", zero_division=0))
            _, _, F, _ = precision_recall_fscore_support(tr_, pr, labels=CLASS_ORDER, zero_division=0)
            r = dict(set=name, seed=seed, macro_fold=np.mean(fm), macro_pooled=F.mean(), acc=accuracy_score(tr_, pr))
            r.update({"F1_" + c: f for c, f in zip(CLASS_ORDER, F)})
            rows.append(r)
    R = pd.DataFrame(rows)
    cols = ["macro_fold", "macro_pooled", "acc", "F1_left_lean", "F1_right_lean", "F1_normal", "F1_forward"]
    return ["# 같은 환경 3명 교차검증: 좌우 특징과 보정", "",
            "값은 seed 평균(표준편차). 보정 프레임(사람별 normal 처음 150프레임)은 모든 조건에서 제외.", "",
            md_table(mean_std(R, "set", cols))]


def mode_cross_env(a):
    d = build(a.p01_dir, True)
    bl = baseline_frames(d)
    d0, _, d_all = prepare_sets(d, bl)
    sets = [("기준 9개", d0, BASE), ("+좌우 3개", d0, BASE + LAT), ("+좌우 3개, 보정", d_all, BASE + LAT)]
    L = ["# 환경 교차 실험", "", "평가 대상 사람의 데이터는 학습에서 제외(사람과 환경이 모두 다른 조건).", ""]
    for title, tr_env, te_env in (("환경 1 학습, 환경 2 평가", 1, 2), ("환경 2 학습, 환경 1 평가", 2, 1)):
        rows = []
        for name, D, cols in sets:
            for seed in range(a.seeds):
                per = []
                for tp in sorted(D[D.env == te_env].p.unique()):
                    te = D[(D.env == te_env) & (D.p == tp)]
                    tr = D[(D.env == tr_env) & (D.p != tp)]
                    p = fit_predict(tr, te, cols, seed)
                    _, _, F, _ = precision_recall_fscore_support(te.pose, p, labels=CLASS_ORDER, zero_division=0)
                    r = dict(macro=F.mean(), acc=accuracy_score(te.pose, p))
                    r.update({"F1_" + c: f for c, f in zip(CLASS_ORDER, F)})
                    per.append(r)
                m = pd.DataFrame(per).mean()
                rows.append(dict(set=name, seed=seed, **m.to_dict()))
        R = pd.DataFrame(rows)
        cols = ["macro", "acc", "F1_left_lean", "F1_right_lean", "F1_normal", "F1_forward"]
        L += [f"## {title}", "", "평가 대상 사람별 값의 평균(seed 평균, 표준편차).", "", md_table(mean_std(R, "set", cols)), ""]
    return L


def mode_binary(a):
    d = build(a.p01_dir, True)
    bl = baseline_frames(d)
    d0, _, d_all = prepare_sets(d, bl)
    sets = [("기준 9개", d0, BASE), ("+좌우 3개", d0, BASE + LAT), ("+좌우 3개, 보정", d_all, BASE + LAT)]
    rows = []
    for name, D, cols in sets:
        for seed in range(3):
            for tag, tr_env, te_env in (("같은 환경 교차", 1, 1), ("환경1→환경2", 1, 2)):
                for tp in sorted(D[D.env == te_env].p.unique()):
                    te = D[(D.env == te_env) & (D.p == tp)]
                    tr = D[(D.env == tr_env) & (D.p != tp)]
                    tr = tr.assign(y=(tr.pose != "normal").astype(int))
                    yte = (te.pose != "normal").astype(int).to_numpy()
                    p = fit_predict(tr, te, cols, seed, "y")
                    tpr = ((p == 1) & (yte == 1)).sum() / (yte == 1).sum()
                    tnr = ((p == 0) & (yte == 0)).sum() / (yte == 0).sum()
                    rows.append(dict(setting=tag, set=name, seed=seed, test=tp, 나쁜자세_검출률=tpr, 정상_정답률=tnr,
                                     균형정확도=(tpr + tnr) / 2))
    R = pd.DataFrame(rows)
    m = R.groupby(["setting", "set", "seed"], sort=False)[["나쁜자세_검출률", "정상_정답률", "균형정확도"]].mean().reset_index()
    g = m.groupby(["setting", "set"], sort=False)[["나쁜자세_검출률", "정상_정답률", "균형정확도"]].mean().reset_index()
    return ["# 정상/나쁨 이진 분류", "", "프레임 단위, 평가 대상 사람은 학습에서 제외, seed 3개 평균.", "", md_table(g)]


def mode_split(a):
    d = load("data/side_logs")
    d["raw"] = d["pose"]
    d["pose"] = d["pose"].map(LABEL_MAPPING)
    five = [c for c in CLASS_ORDER if c not in ("left_lean", "right_lean")]

    def lopo(D, seed, labels):
        pr, tr_, idx = [], [], []
        for ho in sorted(D.p.unique()):
            tr, te = D[D.p != ho], D[D.p == ho]
            pr += list(fit_predict(tr, te, BASE, seed))
            tr_ += list(te.pose)
            idx += list(te.index)
        return np.array(tr_), np.array(pr), idx

    rows, sub = [], None
    for seed in range(a.seeds):
        y, p, idx = lopo(d, seed, CLASS_ORDER)
        _, _, F, _ = precision_recall_fscore_support(y, p, labels=CLASS_ORDER, zero_division=0)
        d5 = d[~d.pose.isin(["left_lean", "right_lean"])]
        y5, p5, _ = lopo(d5, seed, five)
        _, _, F5, _ = precision_recall_fscore_support(y5, p5, labels=five, zero_division=0)
        rows.append(dict(seed=seed, **{"7-class 전체": F.mean(),
                                       "7-class 결과에서 좌우 제외 평균": np.mean([F[CLASS_ORDER.index(c)] for c in five]),
                                       "앞뒤 5개만 다시 학습": F5.mean()}))
        if seed == 0:
            raw = d.loc[idx, "raw"].to_numpy()
            sub = {r: pd.Series(p[raw == r]).value_counts(normalize=True).round(3).to_dict()
                   for r in ("back_neck", "chin", "slouch")}
    R = pd.DataFrame(rows)
    nf = (d[d.raw.isin(["normal", "neck_hard", "neck_light", "left_lean_neck", "right_lean_neck"])]
          .groupby(["p", "raw"])["neck_forward_offset_m"].median().unstack().round(3).reset_index())
    L = ["# 7-class 분할 분석", "",
         "no_hip 9개 특징, 기존 3명 로그(좌표 불필요). 값은 Macro-F1 pooled, seed 평균(표준편차).", "",
         md_table(pd.DataFrame({"방식": R.columns[1:],
                                "Macro-F1": [f"{R[c].mean():.3f} ({R[c].std():.3f})" for c in R.columns[1:]]})), "",
         "## bad_posture 세부 자세가 7-class 모델에서 어디로 예측되는가 (seed 0, 비율)", ""]
    for r, v in sub.items():
        L.append(f"- {r}: {v}")
    L += ["", "## 라벨별 neck_forward_offset_m 중앙값 (좌우 라벨의 고개 내민 정도가 사람마다 다름)", "", md_table(nf)]
    return L


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["lopo", "cross_env", "binary", "split"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--p01-dir", default="data/side_p01_v2")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    L = {"lopo": mode_lopo, "cross_env": mode_cross_env, "binary": mode_binary, "split": mode_split}[a.mode](a)
    out = a.out or f"experiments/lateral_study_{a.mode}.md"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    open(out, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
