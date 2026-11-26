"""규칙 임계값 3개를 사람 단위 교차(한 명 빼고 정하기)로 다시 정해 경고 단위로 평가한다(오프라인, decision.py는 바꾸지 않음).

1단계(grid): 사람 × 임계값 조합마다 경고 단위 결과(자세별 경고 발생률, 기대 종류 일치)를 계산해 experiments/lopo_grid.csv에 저장.
2단계(select): 한 명을 뺀 6명에서 정상 오경고 1% 이하 조합 중 나쁜 자세 검출(사람별 평균)이 가장 높은 조합을 고르고
              (동점이면 현재 임계값에 가까운 조합), 뺀 사람에게 그 조합을 적용한 값을 보고한다.
- 대상 임계값: NECK_OUT_WARN_M(고개 내밈), NECK_BACK_WARN_M(뒤로 기댐), CHIN_WRIST_WARN_SW(턱 괴기).
- 보정 켬(eval_alert_level과 같은 세션 보정). 좌우 lean_neck 제외, _bad·framing 제외.

사용: python -m scripts.lopo_thresholds grid  /  python -m scripts.lopo_thresholds select
"""
from __future__ import annotations

import itertools
import sys

import numpy as np
import pandas as pd

import src.logic.decision as dec
from scripts.eval_1126 import load_all
from scripts.eval_alert_level import BORDER, CALIB_FRAMES, EXPECT, LEAN, PHASES, REF_NF, replay
from scripts.replay_judge_on_logs import DEPTH_COLS, num, real_keypoints, synth_keypoints

GRID = {"NECK_OUT_WARN_M": np.round(np.arange(0.160, 0.2001, 0.005), 3),
        "NECK_BACK_WARN_M": np.round(np.arange(0.080, 0.1201, 0.005), 3),
        "CHIN_WRIST_WARN_SW": np.round(np.arange(0.55, 1.001, 0.05), 2)}
CURRENT = {"NECK_OUT_WARN_M": 0.18, "NECK_BACK_WARN_M": 0.10, "CHIN_WRIST_WARN_SW": 0.65}
GRID_CSV, OUT_MD = "experiments/lopo_grid.csv", "experiments/lopo_thresholds.md"


def prepare(D):
    """사람별로 보정 이동량을 구하고 녹화마다 (keypoints, depth, aspect) 입력과 시각을 미리 만든다."""
    recs = []
    for p, gp in D.groupby("p"):
        nn = gp[gp.pose == "normal"]
        rep0 = sorted(nn.rep.unique())[0]
        first = nn[nn.rep == rep0].sort_values("timestamp").head(CALIB_FRAMES)
        shift = REF_NF - first["neck_forward_offset_m"].median()
        for (pose, rep), g in gp[~gp.pose.isin(LEAN)].groupby(["pose", "rep"]):
            g = g.sort_values("timestamp")
            g = g[~g.index.isin(first.index)]
            if len(g) < 300:
                continue
            inp = []
            for row in g.to_dict("records"):
                d = {c: num(row.get(c)) for c in DEPTH_COLS}
                if d["neck_forward_offset_m"] is not None:
                    d["neck_forward_offset_m"] += shift
                real = real_keypoints(row)
                inp.append((real, d, 640 / 480) if real else (synth_keypoints(row), d, 1.0))
            t = g["timestamp"].to_numpy(float)
            recs.append((p, pose, rep, inp, t - t[0]))
    return recs


def run_config(recs, cfg):
    for k, v in cfg.items():
        setattr(dec, k, v)
    out = []
    for p, pose, rep, inp, t in recs:
        js = [dec.judge(kp, d, frame_aspect=a) for kp, d, a in inp]
        exp = EXPECT.get(pose)
        post = cor = 0
        for ph in PHASES:
            al = [x for x in replay(js, t, ph) if x.startswith("posture")]
            post += bool(al)
            cor += bool(exp) and f"posture:{exp}" in al
        out.append(dict(p=p, pose=pose, rep=rep, posture=post / len(PHASES), correct=cor / len(PHASES)))
    return pd.DataFrame(out).groupby(["p", "pose"])[["posture", "correct"]].mean().reset_index()


def grid(chunk=None):
    """chunk(0~8)를 주면 NECK_OUT_WARN_M 한 값만 계산해 experiments/lopo_grid_part{chunk}.csv에 저장(호출 시간 제한 때문)."""
    D = load_all()
    recs = prepare(D)
    keys = list(GRID)
    rows = []
    space = dict(GRID)
    if chunk is not None:
        space["NECK_OUT_WARN_M"] = [GRID["NECK_OUT_WARN_M"][int(chunk)]]
    for vals in itertools.product(*space.values()):
        cfg = dict(zip(keys, vals))
        r = run_config(recs, cfg)
        for k, v in cfg.items():
            r[k] = v
        rows.append(r)
    for k, v in CURRENT.items():
        setattr(dec, k, v)
    path = GRID_CSV if chunk is None else GRID_CSV.replace(".csv", f"_part{chunk}.csv")
    pd.concat(rows).to_csv(path, index=False)
    print("saved", path, len(rows), "configs")


def person_metrics(G):
    """조합·사람별: 정상 오경고, 나쁜 자세(명확+경계) 검출, 명확, 경계, 종류 일치(자세별 평균)."""
    keys = list(GRID)
    bad = G[G.pose != "normal"]
    m = pd.DataFrame({
        "fa": G[G.pose == "normal"].groupby(keys + ["p"]).posture.mean(),
        "bad": bad.groupby(keys + ["p"]).posture.mean(),
        "clear": bad[~bad.pose.isin(BORDER)].groupby(keys + ["p"]).posture.mean(),
        "border": bad[bad.pose.isin(BORDER)].groupby(keys + ["p"]).posture.mean(),
        "correct": bad[~bad.pose.isin(BORDER) & (bad.pose != "back_neck")].groupby(keys + ["p"]).correct.mean(),
        "chin": bad[bad.pose == "chin"].groupby(keys + ["p"]).posture.mean(),
    }) * 100
    return m.reset_index()


def select():
    import glob
    parts = sorted(glob.glob(GRID_CSV.replace(".csv", "_part*.csv")))
    G = pd.concat([pd.read_csv(f) for f in parts]) if parts else pd.read_csv(GRID_CSV)
    keys = list(GRID)
    M = person_metrics(G)
    people = sorted(M.p.unique())
    dist = sum((M[k] - CURRENT[k]).abs() / (GRID[k].max() - GRID[k].min()) for k in keys)
    M["dist"] = dist
    cur = M[np.all([np.isclose(M[k], CURRENT[k]) for k in keys], axis=0)].set_index("p")
    L = ["# 임계값 사람 단위 교차 재설정 (LOPO, 보정 켬)", "",
         "- 6명으로 고른 조합을 뺀 1명에게 적용한 값. 선택 규칙: 6명 정상 오경고(녹화 단위 합산 아닌 사람 평균) 1% 이하 중 나쁜 자세 검출(사람 평균) 최대, 동점이면 현재 값에 가까운 조합.",
         "- 격자: " + "; ".join(f"{k} {GRID[k].min()}~{GRID[k].max()}" for k in keys), "",
         "| 뺀 사람 | 고른 조합 (고개 내밈 / 뒤로 기댐 / 턱 괴기) | 정상 오경고 현재→재설정 | 명확한 자세 | 경계 자세 | 턱 괴기 | 종류 일치 |",
         "|---|---|---|---|---|---|---|"]
    res = []
    for hold in people:
        tr = M[M.p != hold].groupby(keys)[["fa", "bad", "dist"]].mean().reset_index()
        ok = tr[tr.fa <= 1.0].sort_values(["bad", "dist"], ascending=[False, True])
        best = ok.iloc[0]
        cfg = {k: best[k] for k in keys}
        h = M[(M.p == hold) & np.all([np.isclose(M[k], cfg[k]) for k in keys], axis=0)].iloc[0]
        c = cur.loc[hold]
        res.append(dict(p=hold, **cfg, **{f"new_{x}": h[x] for x in ["fa", "clear", "border", "chin", "correct"]},
                        **{f"cur_{x}": c[x] for x in ["fa", "clear", "border", "chin", "correct"]}))
        L.append(f"| {hold} | {cfg['NECK_OUT_WARN_M']:.3f} / {cfg['NECK_BACK_WARN_M']:.3f} / {cfg['CHIN_WRIST_WARN_SW']:.2f} | "
                 f"{c.fa:.0f} → {h.fa:.0f} | {c.clear:.0f} → {h.clear:.0f} | {c.border:.0f} → {h.border:.0f} | "
                 f"{c.chin:.0f} → {h.chin:.0f} | {c.correct:.0f} → {h.correct:.0f} |")
    R = pd.DataFrame(res)
    L += ["", "사람 평균(7명):", ""]
    for x, lab in [("fa", "정상 오경고"), ("clear", "명확한 자세"), ("border", "경계 자세"), ("chin", "턱 괴기"), ("correct", "종류 일치")]:
        L.append(f"- {lab}: 현재 {R['cur_'+x].mean():.0f}% → 재설정 {R['new_'+x].mean():.0f}%")
    for grp, ps in [("p01~p03", ["p01", "p02", "p03"]), ("p04~p07", ["p04", "p05", "p06", "p07"])]:
        r = R[R.p.isin(ps)]
        L.append(f"- {grp}: 명확한 자세 {r.cur_clear.mean():.0f}→{r.new_clear.mean():.0f}%, 경계 {r.cur_border.mean():.0f}→{r.new_border.mean():.0f}%, 턱 괴기 {r.cur_chin.mean():.0f}→{r.new_chin.mean():.0f}%, 오경고 {r.cur_fa.mean():.0f}→{r.new_fa.mean():.0f}%")
    tr = M.groupby(keys)[["fa", "bad", "dist"]].mean().reset_index()
    b = tr[tr.fa <= 1.0].sort_values(["bad", "dist"], ascending=[False, True]).iloc[0]
    L += ["", f"7명 전체로 고른 조합(앱 적용 후보, 이 값 자체는 같은 데이터로 평가하면 낙관적): "
          f"고개 내밈 {b.NECK_OUT_WARN_M:.3f}, 뒤로 기댐 {b.NECK_BACK_WARN_M:.3f}, 턱 괴기 {b.CHIN_WRIST_WARN_SW:.2f}", ""]
    open(OUT_MD, "w", encoding="utf-8").write("\n".join(L) + "\n")
    R.to_csv("experiments/lopo_thresholds_folds.csv", index=False)
    print("\n".join(L))


if __name__ == "__main__":
    {"grid": grid, "select": select}[sys.argv[1]](*sys.argv[2:])
