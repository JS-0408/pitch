"""
T4.2 — Hungarian IoU association for the multi-target tracker.
Pure functions — no GPU, no side effects.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from yaazhi.types import Detection, Track


def iou(box_a: tuple[float, float, float, float],
        box_b: tuple[float, float, float, float]) -> float:
    """Compute IoU between two boxes in xyxy format."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _track_to_xyxy(t: Track) -> tuple[float, float, float, float]:
    hw, hh = t.w / 2, t.h / 2
    return (t.cx - hw, t.cy - hh, t.cx + hw, t.cy + hh)


def associate(
    tracks: list[Track],
    detections: list[Detection],
    gate_iou: float = 0.2,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """
    Match tracks to detections using Hungarian algorithm on IoU cost matrix.

    Returns:
        matches:          list of (track_idx, det_idx)
        unmatched_tracks: list of track indices with no match
        unmatched_dets:   list of detection indices with no match
    """
    if not tracks or not detections:
        return [], list(range(len(tracks))), list(range(len(detections)))

    n_t, n_d = len(tracks), len(detections)
    iou_matrix = np.zeros((n_t, n_d), dtype=np.float32)
    for i, t in enumerate(tracks):
        for j, d in enumerate(detections):
            iou_matrix[i, j] = iou(_track_to_xyxy(t), d.xyxy)

    cost = 1.0 - iou_matrix
    row_ind, col_ind = linear_sum_assignment(cost)

    matches: list[tuple[int, int]] = []
    unmatched_tracks = list(range(n_t))
    unmatched_dets = list(range(n_d))

    for r, c in zip(row_ind, col_ind):
        if iou_matrix[r, c] >= gate_iou:
            matches.append((r, c))
            unmatched_tracks.remove(r)
            unmatched_dets.remove(c)

    return matches, unmatched_tracks, unmatched_dets
