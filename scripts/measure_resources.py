"""파이프라인 전체 CPU/RAM/fps 측정 (실험 4.7).

조건(base, roi, refine, both)마다 movenet_keypoints를 서브프로세스로 돌리고,
그동안 psutil로 그 프로세스의 CPU 사용률과 메모리(RSS)를 1초마다 기록한다.
- CPU %는 논리 코어 수로 나눈 값(전체 CPU 중 비율)과 코어 1개 기준 값을 둘 다 적는다.
- 처음 WARMUP초(자동 인식·모델 로딩)는 통계에서 뺀다.
- 실효 fps는 --log 파일의 행 수 / (마지막-처음 timestamp)로 계산한다.
- 화면 표시(cv2.imshow)가 켜진 상태의 값이다(실제 사용 형태). 

사용 (레포 루트, venv):  pip install psutil  후
    python -m scripts.measure_resources --seconds 60
    python -m scripts.measure_resources --seconds 60 --conditions base both
카메라 앞에 앉아 정상 자세로 작업하는 듯 있으면 된다. 전원 연결·다른 프로그램 종료 권장.
"""
from __future__ import annotations

import argparse
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import psutil

OPTS = {"base": [], "roi": ["--roi-crop"], "refine": ["--depth-refine"], "both": ["--roi-crop", "--depth-refine"],
        "thunder": [], "dual": []}
DEFAULT_CONDITIONS = ["base", "roi", "refine", "both"]   # 기존 4조건. thunder/dual은 --conditions로 지정할 때만
THUNDER_MODEL = "models/movenet_thunder_int8.tflite"
# thunder: 주 모델을 Thunder로 바꾼 전체 파이프라인 / dual: Lightning 주 모델 + 같은 프레임에 Thunder 동시 추론(teacher 기록)
WARMUP = 10.0


def run_condition(name: str, model: str, seconds: float, out_dir: Path) -> dict:
    log = out_dir / f"res_{name}.csv"
    extra = list(OPTS[name])
    if name == "thunder":
        model = THUNDER_MODEL
    elif name == "dual":
        extra += ["--teacher-model", THUNDER_MODEL, "--teacher-log", str(out_dir / "res_dual_teacher.csv"), "--overwrite"]
    cmd = [sys.executable, "-m", "src.pose.movenet_keypoints", "--model", model, "--realsense", "--judge",
           "--log", str(log), "--duration", str(seconds + WARMUP), *extra]
    ncpu = psutil.cpu_count(logical=True)
    t0 = time.time()
    errf = out_dir / f"stderr_{name}.txt"
    errh = open(errf, "w", encoding="utf-8", errors="replace")
    # 출력을 파일로 돌리면 Windows 기본(cp949)으로 인코딩되어 "—" 같은 문자에서 죽는다 -> UTF-8 강제
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    proc = subprocess.Popen(cmd, stdout=errh, stderr=subprocess.STDOUT, env=env)
    root = psutil.Process(proc.pid)
    cache: dict[int, psutil.Process] = {}   # pid -> Process (cpu_percent 상태 유지)
    cpu, rss = [], []

    def sample() -> tuple[float, float]:
        """런처(venv python.exe)와 그 자식 프로세스 전체의 CPU %(코어1개 기준)·RSS MB 합."""
        procs = [root]
        try:
            procs += root.children(recursive=True)
        except psutil.Error:
            pass
        c_sum = m_sum = 0.0
        for p in procs:
            q = cache.setdefault(p.pid, p)
            try:
                c_sum += q.cpu_percent(None)
                m_sum += q.memory_info().rss / 1e6
            except psutil.Error:
                continue
        return c_sum, m_sum

    sample()  # cpu_percent 기준점
    while proc.poll() is None:
        time.sleep(1.0)
        c, m = sample()
        if time.time() - t0 > WARMUP:
            cpu.append(c)
            rss.append(m)
    proc.wait()
    errh.close()
    elapsed = time.time() - t0
    if not cpu:
        print(f"  ! 샘플 0개: 프로세스가 {elapsed:.1f}초 만에 종료(코드 {proc.returncode}). 마지막 출력({errf}):")
        try:
            tail = errf.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-12:]
            print("\n".join("    " + t for t in tail))
        except OSError:
            pass
    fps = float("nan")
    try:
        d = pd.read_csv(log)
        ts = pd.to_numeric(d["timestamp"], errors="coerce").dropna()
        if len(ts) > 2 and ts.iloc[-1] > ts.iloc[0]:
            fps = (len(ts) - 1) / (ts.iloc[-1] - ts.iloc[0])
    except Exception as e:  # noqa: BLE001
        print(f"  로그 읽기 실패: {e}")
    return {
        "조건": name, "샘플수": len(cpu),
        "CPU %(전체 대비) 평균": statistics.fmean(cpu) / ncpu if cpu else float("nan"),
        "CPU %(코어1개 기준) 평균": statistics.fmean(cpu) if cpu else float("nan"),
        "CPU %(코어1개) 최대": max(cpu) if cpu else float("nan"),
        "RSS MB 평균": statistics.fmean(rss) if rss else float("nan"),
        "RSS MB 최대": max(rss) if rss else float("nan"),
        "실효 fps": fps,
    }


def _table(df: pd.DataFrame) -> str:
    """tabulate가 없어도 되는 마크다운 표."""
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/movenet_lightning_int8.tflite")
    ap.add_argument("--seconds", type=float, default=60.0, help="측정 구간(초). 앞에 WARMUP초가 추가로 돈다")
    ap.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS, choices=list(OPTS))
    ap.add_argument("--out", default="experiments/resource_usage.md")
    args = ap.parse_args()
    out_dir = Path("data/resource_logs")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in args.conditions:
        print(f"[{name}] {args.seconds + WARMUP:.0f}초 측정 중... (카메라 앞에서 정상 자세 유지)")
        r = run_condition(name, args.model, args.seconds, out_dir)
        print("  ", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()})
        rows.append(r)
    df = pd.DataFrame(rows)
    mem = psutil.virtual_memory().total / 1e9
    lines = ["# 파이프라인 CPU/RAM/fps 측정", "",
             f"- 하드웨어: {platform.processor()} / 논리 코어 {psutil.cpu_count(logical=True)}개(물리 {psutil.cpu_count(logical=False)}개) / RAM {mem:.1f} GB / {platform.platform()}",
             f"- 조건당 {args.seconds:.0f}초(앞 {WARMUP:.0f}초 제외), 화면 표시 켜진 상태, --judge 포함, 카메라: RealSense D455",
             "- 서브프로세스 하나의 값이라 시스템 전체 부하와 다르다. 전원 연결·다른 프로그램 종료 상태.", "",
             _table(df.round(2))]
    try:
        Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        print("저장 실패:", e)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
