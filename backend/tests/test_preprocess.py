import numpy as np

from app.pipeline import preprocess, quality
from app.pipeline.preprocess import load_grayscale


def test_load_grayscale_and_normalize(sample_png):
    img = load_grayscale(sample_png)
    assert img.ndim == 2
    assert img.dtype == np.uint8


def test_preprocess_preserves_shape(synthetic_pipe):
    out, params = preprocess.preprocess(synthetic_pipe, {}, preset="light")
    assert out.shape == synthetic_pipe.shape
    assert any(s.startswith("normalize") for s in params["steps"])
    # raw is untouched
    assert synthetic_pipe.max() <= 255


def test_presets_run_all_steps(synthetic_pipe):
    for preset in ("light", "standard", "aggressive"):
        out, params = preprocess.preprocess(synthetic_pipe, {"range_m": 60.0, "altitude_m": 15.0}, preset=preset)
        assert out.shape == synthetic_pipe.shape
        assert params["preset"] == preset


def test_quality_flags_degenerate_image():
    blank = np.zeros((100, 100), np.uint8)
    rep = quality.assess(blank, {})
    assert rep.score < 40
    assert "all_dark" in rep.flags


def test_quality_metadata_completeness():
    rep = quality.assess(np.full((64, 64), 128, np.uint8), {"lat": 1.0, "lon": 2.0, "heading_deg": 90.0})
    assert rep.metadata_completeness["gps"] is True
    assert rep.metadata_completeness["range"] is False
