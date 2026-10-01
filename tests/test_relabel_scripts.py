"""scripts/prepare_blind_relabel_sample.py의 파일명 파싱 로직 테스트."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.prepare_blind_relabel_sample import _parse_label_from_filename


def test_parses_known_label_from_standard_filename():
    assert _parse_label_from_filename("p01_normal_003.jpg") == "normal"
    assert _parse_label_from_filename("p02_slouch_forward_011.jpg") == "slouch_forward"
    assert _parse_label_from_filename("p01_tilt_right_007.jpg") == "tilt_right"


def test_returns_none_for_unrecognized_filename():
    assert _parse_label_from_filename("random_photo.jpg") is None
    assert _parse_label_from_filename("p01_unknownpose_001.jpg") is None
