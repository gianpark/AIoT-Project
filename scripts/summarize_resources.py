"""measure_resources 반복 측정 결과(resource_usage*.md)를 조건별 평균 ± SD로 요약한다.
사용: python -m scripts.summarize_resources experiments/resource_usage.md experiments/resource_usage_run2.md ... [--out 요약파일.md]
--out이 없으면 experiments/resource_usage_summary.md에 쓴다(기존 파일을 덮어쓴다)."""
from __future__ import annotations

import statistics
import sys
from pathlib import Path

COLS = {"fps": "실효 fps", "cpu1": "CPU %(코어1개 기준) 평균", "cpu_all": "CPU %(전체 대비) 평균", "rss": "RSS MB 평균"}


def parse(path: Path) -> dict[str, dict[str, float]]:
    rows, header = {}, None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|") or set(line) <= set("|- "):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if header is None:
            header = cells
        else:
            rows[cells[0]] = dict(zip(header, cells))
    return rows


def main(argv: list[str]) -> None:
    out_path = "experiments/resource_usage_summary.md"
    paths = list(argv)
    if "--out" in paths:
        i = paths.index("--out")
        out_path = paths[i + 1]
        del paths[i:i + 2]
    runs = [parse(Path(p)) for p in paths]
    conds = [c for c in ("base", "roi", "refine", "both", "thunder", "dual") if all(c in r for r in runs)]
    out = [f"# 파이프라인 리소스 반복 측정 요약 ({len(runs)}회, 평균 ± SD)", "",
           "| 조건 | 실효 fps | CPU %(코어 1개) | CPU %(전체 대비) | RAM MB | fps 회차별 |", "|---|---|---|---|---|---|"]
    for c in conds:
        cell = []
        for k in ("fps", "cpu1", "cpu_all", "rss"):
            v = [float(r[c][COLS[k]]) for r in runs]
            cell.append(f"{statistics.fmean(v):.2f} ± {statistics.stdev(v):.2f}" if len(v) > 1 else f"{v[0]:.2f}")
        per = " / ".join(f"{float(r[c][COLS['fps']]):.1f}" for r in runs)
        out.append(f"| {c} | " + " | ".join(cell) + f" | {per} |")
    text = "\n".join(out)
    Path(out_path).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main(sys.argv[1:])
