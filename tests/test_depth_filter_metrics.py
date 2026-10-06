import numpy as np

from src.capture.depth_filter_metrics import center_patch, format_table, frame_stats, summarize


def _flat(value, shape=(100, 120)):
    return np.full(shape, value, dtype=np.float32)


def test_frame_stats_counts_invalid_pixels():
    d = _flat(0.6)
    d[:, :60] = 0  # 왼쪽 절반 무효
    s = frame_stats(d, patch_size=21)
    assert abs(s.frame_valid - 0.5) < 1e-6
    assert 0 < s.patch_valid <= 1.0
    assert abs(s.patch_median_m - 0.6) < 1e-6


def test_all_invalid_patch_gives_none():
    s = frame_stats(_flat(0.0))
    assert s.patch_median_m is None and s.patch_valid == 0


def test_summarize_temporal_std_and_spikes():
    rng = np.random.default_rng(0)
    stats = [frame_stats(_flat(0.6 + rng.normal(0, 0.001))) for _ in range(100)]
    calm = summarize(stats, elapsed_s=10)
    assert calm["fps"] == 10
    assert calm["temporal_std_mm"] < 3 and calm["spike_pct"] == 0

    noisy = [frame_stats(_flat(0.6 + (0.05 if i % 10 == 0 else 0))) for i in range(100)]
    m = summarize(noisy, elapsed_s=10)
    assert m["spike_pct"] > 10 and m["temporal_std_mm"] > calm["temporal_std_mm"]


def test_summarize_empty_and_table():
    assert summarize([], 1.0) == {}
    table = format_table([("a", summarize([frame_stats(_flat(0.5))] * 3, 1.0)), ("b", {})])
    assert "a" in table and "(프레임 없음)" in table


def test_center_patch_shape():
    assert center_patch(_flat(1.0), 41).shape == (41, 41)
