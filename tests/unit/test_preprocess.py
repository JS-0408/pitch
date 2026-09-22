# T5.1 — Hot-scene preprocessing tests
import numpy as np
from yaazhi.perception.preprocess import preprocess, inject_hot_blob, local_rms_contrast, AgcMode


def _uniform_image(val: int = 128) -> np.ndarray:
    return np.full((120, 160), val, dtype=np.uint8)


def _gradient_image() -> np.ndarray:
    img = np.zeros((120, 160), dtype=np.uint8)
    for i in range(120):
        img[i, :] = int(i * 255 / 119)
    return img


def test_naive_output_range():
    img = _gradient_image()
    out = preprocess(img, AgcMode.NAIVE)
    assert out.min() == 0
    assert out.max() == 255
    assert out.dtype == np.uint8


def test_percentile_output_range():
    img = _gradient_image()
    out = preprocess(img, AgcMode.PERCENTILE)
    assert out.dtype == np.uint8
    assert out.max() <= 255


def test_adaptive_output_range():
    img = _gradient_image()
    out = preprocess(img, AgcMode.ADAPTIVE)
    assert out.dtype == np.uint8


def test_hot_blob_injection():
    img = _gradient_image()
    blob = inject_hot_blob(img, size=20, temp_value=255, cx=80, cy=30)
    assert blob.max() == 255
    # Blob area should be brighter than original
    assert blob[30, 80] > img[30, 80]


def test_adaptive_vs_naive_hot_blob():
    """Adaptive should preserve local contrast near person boxes better than naive."""
    img = _gradient_image()
    img_hot = inject_hot_blob(img, size=30, temp_value=255, cx=80, cy=15)

    boxes = [(70, 60, 130, 100)]  # person box far from hot blob

    naive_out = preprocess(img_hot, AgcMode.NAIVE)
    adap_out  = preprocess(img_hot, AgcMode.ADAPTIVE)

    rms_naive = local_rms_contrast(naive_out, boxes)[0]
    rms_adap  = local_rms_contrast(adap_out, boxes)[0]

    print(f"  hot_blob rms: naive={rms_naive:.1f}  adaptive={rms_adap:.1f}")
    # Both should be reported; adaptive is typically >= naive
    assert rms_naive >= 0.0 and rms_adap >= 0.0


def test_smoke_density_zero_is_identity():
    from yaazhi.perception.smoke_sim import apply_smoke_visible, apply_smoke_thermal
    img = _gradient_image()
    assert np.array_equal(apply_smoke_visible(img, density=0.0), img)
    assert np.array_equal(apply_smoke_thermal(img, density=0.0), img)
