"""
T6.1 — HUD renderer.
Ironbow thermal colormap, corner-bracket reticles (not full outlines),
alpha-blended markers, COASTING dashed brackets, status bar.
Renderer has no knowledge of the output target (window vs. file).
Output is always a numpy image.
"""
from __future__ import annotations

import logging
import math

import cv2
import numpy as np

from yaazhi.types import Marker, TrackStatus, SystemState
from yaazhi.rendering.palette import apply_ironbow

logger = logging.getLogger(__name__)

# Reticle corner size (pixels)
_CORNER_LEN_FRAC = 0.25   # fraction of box side
_CORNER_THICK = 2
_DASH_LEN = 5
_DASH_GAP = 4


def _draw_corner_brackets(
    canvas: np.ndarray,
    x: int, y: int, w: int, h: int,
    color: tuple[int, int, int],
    thick: int,
    dashed: bool = False,
) -> None:
    """Draw 4 corner-bracket reticles around a box."""
    lx = max(4, int(w * _CORNER_LEN_FRAC))
    ly = max(4, int(h * _CORNER_LEN_FRAC))

    corners = [
        # (start, horiz_end, vert_end)
        ((x, y),         (x + lx, y),     (x, y + ly)),       # top-left
        ((x + w, y),     (x + w - lx, y), (x + w, y + ly)),   # top-right
        ((x, y + h),     (x + lx, y + h), (x, y + h - ly)),   # bot-left
        ((x + w, y + h), (x + w - lx, y + h), (x + w, y + h - ly)),  # bot-right
    ]

    for (cx, cy), (hx, hy), (vx, vy) in corners:
        if dashed:
            _draw_dashed_line(canvas, (cx, cy), (hx, hy), color, thick)
            _draw_dashed_line(canvas, (cx, cy), (vx, vy), color, thick)
        else:
            cv2.line(canvas, (cx, cy), (hx, hy), color, thick, cv2.LINE_AA)
            cv2.line(canvas, (cx, cy), (vx, vy), color, thick, cv2.LINE_AA)


def _draw_dashed_line(
    canvas: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    color: tuple[int, int, int],
    thick: int,
) -> None:
    """Draw a dashed line between two points."""
    x1, y1 = pt1
    x2, y2 = pt2
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy)
    if length < 1:
        return
    step = _DASH_LEN + _DASH_GAP
    n = max(1, int(length / step))
    for i in range(n):
        t0 = i * step / length
        t1 = min(1.0, (i * step + _DASH_LEN) / length)
        sx, sy = int(x1 + dx * t0), int(y1 + dy * t0)
        ex, ey = int(x1 + dx * t1), int(y1 + dy * t1)
        cv2.line(canvas, (sx, sy), (ex, ey), color, thick, cv2.LINE_AA)


def _alpha_color(base: tuple[int, int, int], alpha: float) -> tuple[int, int, int]:
    """Scale color by alpha (blending toward black)."""
    return tuple(int(c * alpha) for c in base)


class Renderer:
    """
    Renders thermal frames with HUD overlay.
    draw() returns a new numpy image; does not modify input.
    """

    def __init__(self, scale: int = 4) -> None:
        self._scale = scale

    def draw(
        self,
        image: np.ndarray,
        markers: list[Marker],
        hud: dict,
        state: SystemState = SystemState.NORMAL,
        coasting_ids: set[int] | None = None,
    ) -> np.ndarray:
        """
        Render pipeline:
        1. Apply ironbow colormap (grayscale→BGR).
        2. Upscale by display_scale.
        3. Draw markers (corner-bracket reticles + labels).
        4. Draw status bar.
        5. Apply system-state overlays.
        Returns BGR uint8 array.
        """
        coasting = coasting_ids or set()

        # Step 1+2: colormap + upscale
        colored = apply_ironbow(image)
        h, w = image.shape[:2]
        dh, dw = h * self._scale, w * self._scale
        canvas = cv2.resize(colored, (dw, dh), interpolation=cv2.INTER_NEAREST)

        sx = self._scale  # pixel scale factor

        # Step 3: markers
        if state != SystemState.SAFE:
            for m in markers:
                x = int(m.x * sx)
                y = int(m.y * sx)
                bw = int(m.w * sx)
                bh = int(m.h * sx)
                alpha = m.alpha

                dashed = m.track_id in coasting
                color_full = (0, 220, 255)  # yellow-ish in BGR
                color = _alpha_color(color_full, alpha)

                _draw_corner_brackets(canvas, x, y, bw, bh, color, _CORNER_THICK, dashed=dashed)

                # Label
                label_str = m.label
                font_scale = 0.4 * sx / 4
                lx, ly = x, max(4, y - 4)
                cv2.putText(canvas, label_str, (lx, ly),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                            color, 1, cv2.LINE_AA)

        # Step 4: status bar (bottom strip)
        bar_h = 24
        bar = np.zeros((bar_h, dw, 3), dtype=np.uint8)
        hud_str = self._build_hud_string(hud, state)
        cv2.putText(bar, hud_str, (6, 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, (200, 200, 200), 1, cv2.LINE_AA)
        canvas = np.vstack([canvas, bar])

        # Step 5: system-state banner
        if state == SystemState.SAFE:
            self._draw_safe_banner(canvas)
        elif state == SystemState.DEGRADED:
            self._draw_degraded_tint(canvas)

        return canvas

    def _build_hud_string(self, hud: dict, state: SystemState) -> str:
        fps = hud.get("render_fps", 0.0)
        m2d = hud.get("m2d_ms", 0.0)
        tracks = hud.get("tracks", 0)
        toggles = hud.get("toggles", "")
        return (
            f"FPS:{fps:.0f}  M2D:{m2d:.0f}ms  "
            f"TRK:{tracks}  STATE:{state.name}  "
            f"{toggles}  | PUBLIC DATA + SYNTHETIC IMU"
        )

    def _draw_safe_banner(self, canvas: np.ndarray) -> None:
        h, w = canvas.shape[:2]
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 50), -1)
        cv2.addWeighted(overlay, 0.4, canvas, 0.6, 0, canvas)
        msg = "SENSOR LOST. DIRECT VIEW."
        (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.putText(canvas, msg,
                    ((w - tw) // 2, (h - th) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

    def _draw_degraded_tint(self, canvas: np.ndarray) -> None:
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (canvas.shape[1], canvas.shape[0]),
                      (0, 60, 60), -1)
        cv2.addWeighted(overlay, 0.15, canvas, 0.85, 0, canvas)
