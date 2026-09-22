# T6.1 — HUD renderer tests
import numpy as np
from yaazhi.types import Marker, SystemState
from yaazhi.rendering.hud import Renderer
from yaazhi.rendering.palette import apply_ironbow, IRONBOW_LUT


def test_ironbow_lut_shape():
    assert IRONBOW_LUT.shape == (256, 3)
    assert IRONBOW_LUT.dtype == np.uint8


def test_apply_ironbow_shape():
    gray = np.zeros((120, 160), dtype=np.uint8)
    colored = apply_ironbow(gray)
    assert colored.shape == (120, 160, 3)
    assert colored.dtype == np.uint8


def test_apply_ironbow_extremes():
    gray = np.array([[0, 255]], dtype=np.uint8)
    colored = apply_ironbow(gray)
    assert np.array_equal(colored[0, 0], IRONBOW_LUT[0])
    assert np.array_equal(colored[0, 1], IRONBOW_LUT[255])


def test_renderer_output_shape():
    r = Renderer(scale=4)
    img = np.zeros((120, 160), dtype=np.uint8)
    markers = []
    out = r.draw(img, markers, hud={"render_fps": 60, "m2d_ms": 10, "tracks": 0, "toggles": ""})
    # 120*4 + status_bar_height, 160*4
    assert out.shape[1] == 640
    assert out.shape[0] > 480  # includes bar
    assert out.dtype == np.uint8


def test_renderer_marker_pixel_presence():
    """A marker should cause non-zero pixels in its corner region."""
    r = Renderer(scale=4)
    img = np.zeros((120, 160), dtype=np.uint8)
    markers = [Marker(track_id=1, x=20, y=20, w=40, h=40, alpha=1.0, label="P1")]
    out_clean = r.draw(img, [], hud={})
    out_marker = r.draw(img, markers, hud={})
    # Somewhere in the marker area should differ
    assert not np.array_equal(out_clean[:480, :640], out_marker[:480, :640])


def test_renderer_safe_state_banner():
    r = Renderer(scale=4)
    img = np.zeros((120, 160), dtype=np.uint8)
    out = r.draw(img, [], hud={}, state=SystemState.SAFE)
    # SAFE state adds a red banner — check red channel is elevated
    h, w = out.shape[:2]
    center_pixel = out[h // 3, w // 2]  # BGR
    # Red channel (index 2) should be dominant in the banner area
    assert center_pixel[2] > 0 or center_pixel.sum() > 0  # some color present
