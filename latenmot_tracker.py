#!/usr/bin/env python3
"""
LatenMOT: lightweight detector + motion + two-stage association + lost-track
reactivation + conditional appearance ReID.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm


class TrackState(str, Enum):
    TRACKED = "tracked"
    LOST = "lost"
    REMOVED = "removed"


@dataclass
class LatenMOTConfig:
    track_high_thresh: float = 0.45
    track_low_thresh: float = 0.05
    new_track_thresh: float = 0.55
    stage1_min_iou: float = 0.18
    stage2_min_iou: float = 0.08
    reactivation_min_iou: float = 0.08
    appearance_thresh: float = 0.42
    active_appearance_thresh: float = 0.48
    reactivation_cost_thresh: float = 0.72
    track_buffer: int = 60
    draw_lost_frames: int = 12
    motion_gate: float = 18.0
    lost_motion_gate: float = 35.0
    motion_lambda: float = 0.15
    occlusion_coverage_thresh: float = 0.45
    occlusion_velocity_damping: float = 0.55
    occlusion_reset_alpha: float = 0.08
    occlusion_box_enlarge: float = 1.25
    visibility_momentum: float = 0.75
    lost_visibility_decay: float = 0.92
    occluded_visibility_decay: float = 0.96
    output_visibility_thresh: float = 0.16
    lost_output_visibility_thresh: float = 0.22
    use_deferred_birth: bool = True
    pending_confirm_hits: int = 3
    pending_max_misses: int = 2
    pending_min_iou: float = 0.18
    use_deferred_reactivation: bool = True
    reactivation_evidence_thresh: float = 0.95
    reactivation_strong_quality: float = 0.72
    reactivation_evidence_decay: float = 0.65
    min_box_area: float = 12.0
    use_reid: bool = True
    use_active_reid: bool = True
    reid_after_frames: int = 2
    feature_alpha: float = 0.9


@dataclass
class Detection:
    xyxy: np.ndarray
    tlwh: np.ndarray
    score: float
    cls: int
    feature: Optional[np.ndarray] = None


def xyxy_to_tlwh(xyxy: Sequence[float]) -> np.ndarray:
    x1, y1, x2, y2 = map(float, xyxy)
    return np.array([x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)], dtype=np.float32)


def tlwh_to_xyxy(tlwh: Sequence[float]) -> np.ndarray:
    x, y, w, h = map(float, tlwh)
    return np.array([x, y, x + w, y + h], dtype=np.float32)


def tlwh_to_xyah(tlwh: Sequence[float]) -> np.ndarray:
    x, y, w, h = map(float, tlwh)
    h = max(h, 1e-6)
    return np.array([x + w / 2.0, y + h / 2.0, w / h, h], dtype=np.float32)


def xyah_to_tlwh(xyah: Sequence[float]) -> np.ndarray:
    cx, cy, a, h = map(float, xyah)
    w = max(0.0, a * h)
    return np.array([cx - w / 2.0, cy - h / 2.0, w, h], dtype=np.float32)


def bbox_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)

    a = a.astype(np.float32)
    b = b.astype(np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0.0, None)
    inter = wh[:, :, 0] * wh[:, :, 1]

    area_a = np.clip(a[:, 2] - a[:, 0], 0.0, None) * np.clip(a[:, 3] - a[:, 1], 0.0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0.0, None) * np.clip(b[:, 3] - b[:, 1], 0.0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.clip(union, 1e-6, None)


def iou_cost(tracks: Sequence["Track"], detections: Sequence[Detection]) -> np.ndarray:
    track_boxes = np.array([t.xyxy for t in tracks], dtype=np.float32)
    det_boxes = np.array([d.xyxy for d in detections], dtype=np.float32)
    return 1.0 - bbox_iou(track_boxes, det_boxes)


def track_iou_cost(
    tracks: Sequence["Track"], detections: Sequence[Detection], use_search_box: bool = False
) -> np.ndarray:
    track_boxes = np.array(
        [t.search_xyxy if use_search_box else t.xyxy for t in tracks], dtype=np.float32
    )
    det_boxes = np.array([d.xyxy for d in detections], dtype=np.float32)
    return 1.0 - bbox_iou(track_boxes, det_boxes)


def coverage_ratio(target: np.ndarray, occluders: np.ndarray) -> float:
    if len(occluders) == 0:
        return 0.0
    target = target.astype(np.float32)
    occluders = occluders.astype(np.float32)
    target_area = max(1e-6, float((target[2] - target[0]) * (target[3] - target[1])))
    lt = np.maximum(target[None, :2], occluders[:, :2])
    rb = np.minimum(target[None, 2:], occluders[:, 2:])
    wh = np.clip(rb - lt, 0.0, None)
    inter = wh[:, 0] * wh[:, 1]
    return float(np.max(inter / target_area))


def detection_coverages(detections: Sequence[Detection]) -> dict[int, float]:
    boxes = np.array([d.xyxy for d in detections], dtype=np.float32)
    coverages: dict[int, float] = {}
    for idx, det in enumerate(detections):
        if len(detections) <= 1:
            coverages[id(det)] = 0.0
            continue
        others = np.delete(boxes, idx, axis=0)
        coverages[id(det)] = coverage_ratio(det.xyxy, others)
    return coverages


def motion_uncertainty(mean: np.ndarray, covariance: np.ndarray) -> float:
    h = max(1.0, float(mean[3]))
    pos_sigma = float(np.sqrt(max(0.0, np.trace(covariance[:2, :2]))))
    return float(np.clip(pos_sigma / h, 0.0, 2.0) / 2.0)


def visibility_from_evidence(score: float, coverage: float, uncertainty: float) -> float:
    score_term = float(np.clip(score, 0.0, 1.0))
    coverage_term = 1.0 - 0.65 * float(np.clip(coverage, 0.0, 1.0))
    uncertainty_term = 1.0 - 0.45 * float(np.clip(uncertainty, 0.0, 1.0))
    return float(np.clip(score_term * coverage_term * uncertainty_term, 0.0, 1.0))


def motion_gated_iou_cost(
    tracks: Sequence["Track"],
    detections: Sequence[Detection],
    kf: "KalmanFilter",
    gate_limit: float,
    motion_lambda: float,
) -> np.ndarray:
    if not tracks or not detections:
        return np.zeros((len(tracks), len(detections)), dtype=np.float32)

    cost = iou_cost(tracks, detections)
    measurements = np.array([tlwh_to_xyah(det.tlwh) for det in detections], dtype=np.float32)
    for r, track in enumerate(tracks):
        gate = kf.gating_distance(track.mean, track.covariance, measurements)
        cost[r, gate > gate_limit] = np.inf
        cost[r] = cost[r] + motion_lambda * np.clip(gate / gate_limit, 0.0, 2.0)
    return cost


def active_association_cost(
    tracks: Sequence["Track"], detections: Sequence[Detection], cfg: LatenMOTConfig, kf: "KalmanFilter"
) -> np.ndarray:
    if not tracks or not detections:
        return np.zeros((len(tracks), len(detections)), dtype=np.float32)

    iou_dist = iou_cost(tracks, detections)
    measurements = np.array([tlwh_to_xyah(det.tlwh) for det in detections], dtype=np.float32)
    if not cfg.use_reid or not cfg.use_active_reid:
        return motion_gated_iou_cost(tracks, detections, kf, cfg.motion_gate, cfg.motion_lambda)

    cost = np.full_like(iou_dist, fill_value=np.inf, dtype=np.float32)
    for r, track in enumerate(tracks):
        gate = kf.gating_distance(track.mean, track.covariance, measurements)
        for c, det in enumerate(detections):
            if gate[c] > cfg.motion_gate:
                continue
            iou = 1.0 - float(iou_dist[r, c])
            app_dist = cosine_distance(track.smooth_feature, det.feature)
            can_use_motion = iou >= cfg.stage1_min_iou
            can_use_appearance = (
                track.smooth_feature is not None
                and det.feature is not None
                and app_dist <= cfg.active_appearance_thresh
            )
            if can_use_motion or can_use_appearance:
                cost[r, c] = (
                    0.65 * iou_dist[r, c] + 0.35 * app_dist
                    if can_use_appearance
                    else iou_dist[r, c]
                )
                cost[r, c] += cfg.motion_lambda * min(float(gate[c]) / cfg.motion_gate, 2.0)
    return cost


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 1.0
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom <= 1e-12:
        return 1.0
    return float(1.0 - np.dot(a, b) / denom)


def linear_assignment_with_threshold(
    cost: np.ndarray, max_cost: float
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    if cost.size == 0:
        return [], list(range(cost.shape[0])), list(range(cost.shape[1]))

    safe_cost = cost.copy()
    safe_cost[~np.isfinite(safe_cost)] = max_cost + 1e5
    row_ind, col_ind = linear_sum_assignment(safe_cost)

    matches: List[Tuple[int, int]] = []
    unmatched_rows = set(range(cost.shape[0]))
    unmatched_cols = set(range(cost.shape[1]))
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] <= max_cost:
            matches.append((int(r), int(c)))
            unmatched_rows.discard(int(r))
            unmatched_cols.discard(int(c))

    return matches, sorted(unmatched_rows), sorted(unmatched_cols)


class KalmanFilter:
    """Constant velocity Kalman filter over [cx, cy, aspect, h]."""

    ndim = 4
    dt = 1.0

    def __init__(self) -> None:
        self.motion_mat = np.eye(2 * self.ndim, dtype=np.float32)
        for i in range(self.ndim):
            self.motion_mat[i, self.ndim + i] = self.dt
        self.update_mat = np.eye(self.ndim, 2 * self.ndim, dtype=np.float32)
        self.std_weight_position = 1.0 / 20
        self.std_weight_velocity = 1.0 / 160

    def initiate(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mean = np.r_[measurement, np.zeros_like(measurement)].astype(np.float32)
        std = np.array(
            [
                2 * self.std_weight_position * measurement[3],
                2 * self.std_weight_position * measurement[3],
                1e-2,
                2 * self.std_weight_position * measurement[3],
                10 * self.std_weight_velocity * measurement[3],
                10 * self.std_weight_velocity * measurement[3],
                1e-5,
                10 * self.std_weight_velocity * measurement[3],
            ],
            dtype=np.float32,
        )
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(
        self, mean: np.ndarray, covariance: np.ndarray, lost_frames: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        h = max(1.0, float(mean[3]))
        uncertainty = min(1.0 + 0.25 * max(0, lost_frames), 4.0)
        std_pos = np.array(
            [
                self.std_weight_position * h * uncertainty,
                self.std_weight_position * h * uncertainty,
                1e-2,
                self.std_weight_position * h * uncertainty,
            ],
            dtype=np.float32,
        )
        std_vel = np.array(
            [
                self.std_weight_velocity * h * uncertainty,
                self.std_weight_velocity * h * uncertainty,
                1e-5,
                self.std_weight_velocity * h * uncertainty,
            ],
            dtype=np.float32,
        )
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        mean = self.motion_mat @ mean
        covariance = self.motion_mat @ covariance @ self.motion_mat.T + motion_cov
        return mean.astype(np.float32), covariance.astype(np.float32)

    def project(
        self, mean: np.ndarray, covariance: np.ndarray, score: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        h = max(1.0, float(mean[3]))
        score = float(np.clip(score, 0.01, 1.0))
        measurement_scale = 1.0 + 2.5 * (1.0 - score)
        std = np.array(
            [
                self.std_weight_position * h * measurement_scale,
                self.std_weight_position * h * measurement_scale,
                1e-1 * measurement_scale,
                self.std_weight_position * h * measurement_scale,
            ],
            dtype=np.float32,
        )
        innovation_cov = np.diag(np.square(std))
        projected_mean = self.update_mat @ mean
        projected_cov = self.update_mat @ covariance @ self.update_mat.T + innovation_cov
        return projected_mean.astype(np.float32), projected_cov.astype(np.float32)

    def update(
        self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray, score: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        projected_mean, projected_cov = self.project(mean, covariance, score)
        kalman_gain = covariance @ self.update_mat.T @ np.linalg.inv(projected_cov)
        innovation = measurement - projected_mean
        new_mean = mean + kalman_gain @ innovation
        new_covariance = covariance - kalman_gain @ projected_cov @ kalman_gain.T
        return new_mean.astype(np.float32), new_covariance.astype(np.float32)

    def apply_occlusion_damping(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        last_observed_mean: Optional[np.ndarray],
        velocity_damping: float,
        reset_alpha: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        mean = mean.copy()
        covariance = covariance.copy()
        mean[4] *= velocity_damping
        mean[5] *= velocity_damping
        mean[6] *= velocity_damping
        mean[7] *= velocity_damping
        if last_observed_mean is not None:
            mean[:2] = (1.0 - reset_alpha) * mean[:2] + reset_alpha * last_observed_mean[:2]
            mean[2:4] = 0.95 * mean[2:4] + 0.05 * last_observed_mean[2:4]
        covariance[:4, :4] *= 1.1
        return mean.astype(np.float32), covariance.astype(np.float32)

    def gating_distance(
        self, mean: np.ndarray, covariance: np.ndarray, measurements: np.ndarray
    ) -> np.ndarray:
        if len(measurements) == 0:
            return np.zeros((0,), dtype=np.float32)
        projected_mean, projected_cov = self.project(mean, covariance)
        eye = np.eye(self.ndim, dtype=np.float32)
        try:
            chol = np.linalg.cholesky(projected_cov + eye * 1e-6)
        except np.linalg.LinAlgError:
            chol = np.linalg.cholesky(projected_cov + eye * 1e-3)
        diff = (measurements - projected_mean).T
        z = np.linalg.solve(chol, diff)
        return np.sum(z * z, axis=0).astype(np.float32)


class ColorHistReID:
    """Small appearance embedding used only when the tracker needs ReID help."""

    def __init__(self, bins: Tuple[int, int, int] = (16, 8, 8)) -> None:
        self.bins = bins

    def extract(self, frame: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = xyxy.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        if x2 <= x1 or y2 <= y1:
            return np.zeros(sum(self.bins), dtype=np.float32)

        crop = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([hsv], [0], None, [self.bins[0]], [0, 180]).flatten()
        hist_s = cv2.calcHist([hsv], [1], None, [self.bins[1]], [0, 256]).flatten()
        hist_v = cv2.calcHist([hsv], [2], None, [self.bins[2]], [0, 256]).flatten()
        feat = np.concatenate([hist_h, hist_s, hist_v]).astype(np.float32)
        norm = np.linalg.norm(feat)
        return feat / norm if norm > 1e-12 else feat


class Track:
    _next_id = 1

    def __init__(self, detection: Detection, frame_id: int, kf: KalmanFilter) -> None:
        self.track_id = Track._next_id
        Track._next_id += 1
        self.mean, self.covariance = kf.initiate(tlwh_to_xyah(detection.tlwh))
        self.score = detection.score
        self.cls = detection.cls
        self.state = TrackState.TRACKED
        self.start_frame = frame_id
        self.frame_id = frame_id
        self.last_seen = frame_id
        self.lost_since: Optional[int] = None
        self.is_occluded = False
        self.occlusion_age = 0
        self.search_enlarge = 1.0
        self.last_observed_mean = self.mean.copy()
        self.visibility = float(detection.score)
        self.reactivation_evidence = 0.0
        self.smooth_feature = detection.feature

    @property
    def tlwh(self) -> np.ndarray:
        return xyah_to_tlwh(self.mean[:4])

    @property
    def xyxy(self) -> np.ndarray:
        return tlwh_to_xyxy(self.tlwh)

    @property
    def search_xyxy(self) -> np.ndarray:
        tlwh = self.tlwh.copy()
        if self.search_enlarge > 1.0:
            cx = tlwh[0] + tlwh[2] / 2.0
            cy = tlwh[1] + tlwh[3] / 2.0
            tlwh[2] *= self.search_enlarge
            tlwh[3] *= self.search_enlarge
            tlwh[0] = cx - tlwh[2] / 2.0
            tlwh[1] = cy - tlwh[3] / 2.0
        return tlwh_to_xyxy(tlwh)

    def predict(self, kf: KalmanFilter, frame_id: int, cfg: LatenMOTConfig) -> None:
        if self.state != TrackState.TRACKED:
            self.mean[7] = 0
        lost_frames = max(0, frame_id - self.last_seen) if self.state == TrackState.LOST else 0
        self.mean, self.covariance = kf.predict(self.mean, self.covariance, lost_frames)
        if self.is_occluded:
            self.occlusion_age += 1
            self.search_enlarge = cfg.occlusion_box_enlarge
            self.visibility *= cfg.occluded_visibility_decay
            self.mean, self.covariance = kf.apply_occlusion_damping(
                self.mean,
                self.covariance,
                self.last_observed_mean,
                cfg.occlusion_velocity_damping,
                cfg.occlusion_reset_alpha,
            )
        elif self.state == TrackState.LOST:
            self.visibility *= cfg.lost_visibility_decay

    def update(
        self,
        detection: Detection,
        frame_id: int,
        kf: KalmanFilter,
        alpha: float,
        cfg: LatenMOTConfig,
        coverage: float = 0.0,
    ) -> None:
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, tlwh_to_xyah(detection.tlwh), detection.score
        )
        self.score = detection.score
        self.cls = detection.cls
        self.state = TrackState.TRACKED
        self.frame_id = frame_id
        self.last_seen = frame_id
        self.lost_since = None
        self.is_occluded = False
        self.occlusion_age = 0
        self.search_enlarge = 1.0
        self.last_observed_mean = self.mean.copy()
        uncertainty = motion_uncertainty(self.mean, self.covariance)
        instant_visibility = visibility_from_evidence(detection.score, coverage, uncertainty)
        self.visibility = (
            cfg.visibility_momentum * self.visibility
            + (1.0 - cfg.visibility_momentum) * instant_visibility
        )
        self.reactivation_evidence = 0.0
        if detection.feature is not None:
            if self.smooth_feature is None:
                self.smooth_feature = detection.feature
            else:
                self.smooth_feature = alpha * self.smooth_feature + (1.0 - alpha) * detection.feature
                norm = np.linalg.norm(self.smooth_feature)
                if norm > 1e-12:
                    self.smooth_feature = self.smooth_feature / norm

    def mark_lost(self, frame_id: int, cfg: LatenMOTConfig) -> None:
        self.state = TrackState.LOST
        self.lost_since = frame_id if self.lost_since is None else self.lost_since
        self.is_occluded = False
        self.search_enlarge = 1.0
        self.visibility *= cfg.lost_visibility_decay

    def mark_occluded(self, frame_id: int, kf: KalmanFilter, cfg: LatenMOTConfig) -> None:
        self.state = TrackState.LOST
        self.lost_since = frame_id if self.lost_since is None else self.lost_since
        if not self.is_occluded:
            self.occlusion_age = 0
        self.is_occluded = True
        self.search_enlarge = cfg.occlusion_box_enlarge
        self.visibility *= cfg.occluded_visibility_decay
        self.mean, self.covariance = kf.apply_occlusion_damping(
            self.mean,
            self.covariance,
            self.last_observed_mean,
            cfg.occlusion_velocity_damping,
            cfg.occlusion_reset_alpha,
        )

    def mark_removed(self) -> None:
        self.state = TrackState.REMOVED


class PendingCandidate:
    def __init__(self, detection: Detection, frame_id: int, cfg: LatenMOTConfig) -> None:
        self.tlwh = detection.tlwh.copy()
        self.xyxy = detection.xyxy.copy()
        self.score_sum = float(detection.score)
        self.score = float(detection.score)
        self.cls = detection.cls
        self.first_frame = frame_id
        self.last_frame = frame_id
        self.hits = 1
        self.misses = 0
        self.feature = detection.feature
        uncertainty = 0.0
        self.visibility = visibility_from_evidence(detection.score, 0.0, uncertainty)

    def update(self, detection: Detection, frame_id: int, cfg: LatenMOTConfig, coverage: float) -> None:
        self.tlwh = detection.tlwh.copy()
        self.xyxy = detection.xyxy.copy()
        self.score_sum += float(detection.score)
        self.score = float(detection.score)
        self.cls = detection.cls
        self.last_frame = frame_id
        self.hits += 1
        self.misses = 0
        instant_visibility = visibility_from_evidence(detection.score, coverage, 0.0)
        self.visibility = (
            cfg.visibility_momentum * self.visibility
            + (1.0 - cfg.visibility_momentum) * instant_visibility
        )
        if detection.feature is not None:
            if self.feature is None:
                self.feature = detection.feature
            else:
                self.feature = cfg.feature_alpha * self.feature + (1.0 - cfg.feature_alpha) * detection.feature
                norm = np.linalg.norm(self.feature)
                if norm > 1e-12:
                    self.feature = self.feature / norm

    def mark_missed(self) -> None:
        self.misses += 1
        self.visibility *= 0.85

    def is_confirmed(self, cfg: LatenMOTConfig) -> bool:
        return self.hits >= cfg.pending_confirm_hits and self.visibility >= cfg.output_visibility_thresh

    def to_detection(self) -> Detection:
        avg_score = max(self.score, self.score_sum / max(1, self.hits))
        return Detection(
            xyxy=self.xyxy.copy(),
            tlwh=self.tlwh.copy(),
            score=float(avg_score),
            cls=self.cls,
            feature=self.feature,
        )


class LatenMOTTracker:
    def __init__(self, config: LatenMOTConfig) -> None:
        self.cfg = config
        self.kf = KalmanFilter()
        self.reid = ColorHistReID()
        self.tracks: List[Track] = []
        self.pending_candidates: List[PendingCandidate] = []
        self.frame_id = 0

    def _ensure_features(self, frame: np.ndarray, detections: Sequence[Detection], indices: Iterable[int]) -> None:
        if not self.cfg.use_reid:
            return
        for idx in indices:
            det = detections[idx]
            if det.feature is None:
                det.feature = self.reid.extract(frame, det.xyxy)

    def _reactivation_cost(self, lost_tracks: Sequence[Track], detections: Sequence[Detection]) -> np.ndarray:
        if not lost_tracks or not detections:
            return np.zeros((len(lost_tracks), len(detections)), dtype=np.float32)

        iou_dist = track_iou_cost(lost_tracks, detections, use_search_box=True)
        measurements = np.array([tlwh_to_xyah(det.tlwh) for det in detections], dtype=np.float32)
        cost = np.full_like(iou_dist, fill_value=np.inf, dtype=np.float32)
        for r, track in enumerate(lost_tracks):
            gap = self.frame_id - track.last_seen
            gate = self.kf.gating_distance(track.mean, track.covariance, measurements)
            gate_limit = self.cfg.lost_motion_gate * min(1.0 + 0.05 * gap, 2.0)
            if track.is_occluded:
                gate_limit *= 1.4
            for c, det in enumerate(detections):
                if gate[c] > gate_limit:
                    continue
                app_dist = cosine_distance(track.smooth_feature, det.feature)
                iou = 1.0 - float(iou_dist[r, c])
                can_use_motion = iou >= self.cfg.reactivation_min_iou
                can_use_reid = (
                    self.cfg.use_reid
                    and gap >= self.cfg.reid_after_frames
                    and track.smooth_feature is not None
                    and det.feature is not None
                    and app_dist <= self.cfg.appearance_thresh
                )
                if can_use_motion or can_use_reid:
                    if can_use_reid:
                        cost[r, c] = 0.45 * iou_dist[r, c] + 0.55 * app_dist
                    else:
                        cost[r, c] = iou_dist[r, c]
                    cost[r, c] += self.cfg.motion_lambda * min(float(gate[c]) / gate_limit, 2.0)
        return cost

    def _mark_unmatched_active(
        self,
        tracks: Sequence[Track],
        matched_tracks: Sequence[Track],
        detections: Sequence[Detection],
    ) -> None:
        tracked_boxes = [t.xyxy for t in matched_tracks if t.state == TrackState.TRACKED]
        det_boxes = [d.xyxy for d in detections]
        occluder_boxes = np.array(tracked_boxes + det_boxes, dtype=np.float32)
        for track in tracks:
            coverage = coverage_ratio(track.xyxy, occluder_boxes)
            if coverage >= self.cfg.occlusion_coverage_thresh:
                track.mark_occluded(self.frame_id, self.kf, self.cfg)
            else:
                track.mark_lost(self.frame_id, self.cfg)

    def _update_pending_candidates(
        self,
        frame: np.ndarray,
        detections: Sequence[Detection],
        coverage_by_det: dict[int, float],
    ) -> List[Track]:
        if not self.cfg.use_deferred_birth:
            return []

        self._ensure_features(frame, detections, range(len(detections)))
        confirmed_tracks: List[Track] = []
        if self.pending_candidates and detections:
            cand_boxes = np.array([c.xyxy for c in self.pending_candidates], dtype=np.float32)
            det_boxes = np.array([d.xyxy for d in detections], dtype=np.float32)
            cost = 1.0 - bbox_iou(cand_boxes, det_boxes)
            for r, candidate in enumerate(self.pending_candidates):
                if candidate.feature is None:
                    continue
                for c, det in enumerate(detections):
                    app_dist = cosine_distance(candidate.feature, det.feature)
                    if app_dist <= self.cfg.active_appearance_thresh:
                        cost[r, c] = min(cost[r, c], 0.65 * cost[r, c] + 0.35 * app_dist)
            matches, unmatched_candidates, unmatched_dets = linear_assignment_with_threshold(
                cost, max_cost=1.0 - self.cfg.pending_min_iou
            )
        else:
            matches = []
            unmatched_candidates = list(range(len(self.pending_candidates)))
            unmatched_dets = list(range(len(detections)))

        for cand_idx, det_idx in matches:
            candidate = self.pending_candidates[cand_idx]
            det = detections[det_idx]
            candidate.update(det, self.frame_id, self.cfg, coverage_by_det.get(id(det), 0.0))
            if candidate.is_confirmed(self.cfg):
                confirmed_tracks.append(Track(candidate.to_detection(), self.frame_id, self.kf))

        confirmed_indices = {
            idx for idx, candidate in enumerate(self.pending_candidates) if candidate.is_confirmed(self.cfg)
        }
        for cand_idx in unmatched_candidates:
            if cand_idx in confirmed_indices:
                continue
            self.pending_candidates[cand_idx].mark_missed()

        self.pending_candidates = [
            c
            for idx, c in enumerate(self.pending_candidates)
            if idx not in confirmed_indices and c.misses <= self.cfg.pending_max_misses
        ]

        for det_idx in unmatched_dets:
            det = detections[det_idx]
            if det.score >= self.cfg.new_track_thresh:
                self.pending_candidates.append(PendingCandidate(det, self.frame_id, self.cfg))

        return confirmed_tracks

    def update(self, frame: np.ndarray, detections: List[Detection]) -> List[Track]:
        self.frame_id += 1

        detections = [
            d
            for d in detections
            if d.tlwh[2] * d.tlwh[3] >= self.cfg.min_box_area and d.tlwh[2] > 1 and d.tlwh[3] > 1
        ]
        coverage_by_det = detection_coverages(detections)
        high_dets = [d for d in detections if d.score >= self.cfg.track_high_thresh]
        low_dets = [d for d in detections if self.cfg.track_low_thresh <= d.score < self.cfg.track_high_thresh]

        for track in self.tracks:
            if track.state in (TrackState.TRACKED, TrackState.LOST):
                track.predict(self.kf, self.frame_id, self.cfg)

        active_tracks = [t for t in self.tracks if t.state == TrackState.TRACKED]
        lost_tracks = [t for t in self.tracks if t.state == TrackState.LOST]

        # Stage 1: confident detections against active tracks using motion/IoU.
        # In crowded scenes, a cheap appearance fallback helps preserve IDs when
        # boxes jump because people overlap.
        if self.cfg.use_active_reid:
            self._ensure_features(frame, high_dets, range(len(high_dets)))
        stage1_matches, unmatched_active, unmatched_high = linear_assignment_with_threshold(
            active_association_cost(active_tracks, high_dets, self.cfg, self.kf),
            max_cost=1.0 - self.cfg.stage1_min_iou,
        )
        for trk_idx, det_idx in stage1_matches:
            self._ensure_features(frame, high_dets, [det_idx])
            det = high_dets[det_idx]
            active_tracks[trk_idx].update(
                det,
                self.frame_id,
                self.kf,
                self.cfg.feature_alpha,
                self.cfg,
                coverage_by_det.get(id(det), 0.0),
            )

        # Stage 2: low confidence detections can keep existing tracks alive.
        remaining_active = [active_tracks[i] for i in unmatched_active]
        stage2_matches, unmatched_remaining_active, _ = linear_assignment_with_threshold(
            motion_gated_iou_cost(
                remaining_active,
                low_dets,
                self.kf,
                self.cfg.motion_gate * 1.5,
                self.cfg.motion_lambda,
            ),
            max_cost=1.0 - self.cfg.stage2_min_iou,
        )
        for trk_idx, det_idx in stage2_matches:
            self._ensure_features(frame, low_dets, [det_idx])
            det = low_dets[det_idx]
            remaining_active[trk_idx].update(
                det,
                self.frame_id,
                self.kf,
                self.cfg.feature_alpha,
                self.cfg,
                coverage_by_det.get(id(det), 0.0),
            )

        matched_current_tracks = [
            t for t in self.tracks if t.state == TrackState.TRACKED and t.last_seen == self.frame_id
        ]
        self._mark_unmatched_active(
            [remaining_active[i] for i in unmatched_remaining_active],
            matched_current_tracks,
            detections,
        )

        # Lost re-activation: try old IDs before creating new ones.
        remaining_high = [high_dets[i] for i in unmatched_high]
        self._ensure_features(frame, remaining_high, range(len(remaining_high)))
        lost_tracks = [t for t in self.tracks if t.state == TrackState.LOST]
        react_cost = self._reactivation_cost(lost_tracks, remaining_high)
        react_matches, unmatched_lost, unmatched_remaining_high = linear_assignment_with_threshold(
            react_cost,
            max_cost=self.cfg.reactivation_cost_thresh,
        )
        for trk_idx, det_idx in react_matches:
            track = lost_tracks[trk_idx]
            det = remaining_high[det_idx]
            quality = max(0.0, 1.0 - float(react_cost[trk_idx, det_idx]) / self.cfg.reactivation_cost_thresh)
            track.reactivation_evidence = (
                self.cfg.reactivation_evidence_decay * track.reactivation_evidence + quality
            )
            strong_match = quality >= self.cfg.reactivation_strong_quality
            enough_evidence = track.reactivation_evidence >= self.cfg.reactivation_evidence_thresh
            if (not self.cfg.use_deferred_reactivation) or strong_match or enough_evidence:
                track.update(
                    det,
                    self.frame_id,
                    self.kf,
                    self.cfg.feature_alpha,
                    self.cfg,
                    coverage_by_det.get(id(det), 0.0),
                )

        for trk_idx in unmatched_lost:
            lost_tracks[trk_idx].reactivation_evidence *= self.cfg.reactivation_evidence_decay

        # Only create a new ID after active and lost tracks had a chance to match.
        birth_det_indices = list(unmatched_remaining_high)
        birth_dets = [remaining_high[i] for i in birth_det_indices]
        if self.cfg.use_deferred_birth:
            self.tracks.extend(self._update_pending_candidates(frame, birth_dets, coverage_by_det))
        else:
            for det in birth_dets:
                if det.score >= self.cfg.new_track_thresh:
                    self._ensure_features(frame, [det], [0])
                    self.tracks.append(Track(det, self.frame_id, self.kf))

        for trk_idx in unmatched_lost:
            track = lost_tracks[trk_idx]
            if self.frame_id - track.last_seen > self.cfg.track_buffer:
                track.mark_removed()

        self.tracks = [t for t in self.tracks if t.state != TrackState.REMOVED]
        visible_tracks = [
            t
            for t in self.tracks
            if t.state == TrackState.TRACKED
            and t.last_seen == self.frame_id
            and t.visibility >= self.cfg.output_visibility_thresh
        ]
        if self.cfg.draw_lost_frames > 0:
            visible_tracks.extend(
                t
                for t in self.tracks
                if t.state == TrackState.LOST and self.frame_id - t.last_seen <= self.cfg.draw_lost_frames
                and t.visibility >= self.cfg.lost_output_visibility_thresh
            )
        return visible_tracks


class YOLODetector:
    def __init__(
        self,
        weights: str,
        device: str,
        imgsz: int,
        conf: float,
        iou: float,
        max_det: int,
        agnostic_nms: bool,
        person_class: Optional[int],
    ) -> None:
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.max_det = max_det
        self.agnostic_nms = agnostic_nms
        self.person_class = person_class

    def __call__(self, frame: np.ndarray) -> List[Detection]:
        classes = None if self.person_class is None else [self.person_class]
        result = self.model.predict(
            frame,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            max_det=self.max_det,
            agnostic_nms=self.agnostic_nms,
            device=self.device,
            classes=classes,
            verbose=False,
        )[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []

        xyxy = result.boxes.xyxy.detach().cpu().numpy()
        conf = result.boxes.conf.detach().cpu().numpy()
        cls = result.boxes.cls.detach().cpu().numpy().astype(int)
        return [
            Detection(xyxy=box.astype(np.float32), tlwh=xyxy_to_tlwh(box), score=float(score), cls=int(c))
            for box, score, c in zip(xyxy, conf, cls)
        ]


def id_color(track_id: int) -> Tuple[int, int, int]:
    hue = (track_id * 37) % 180
    color = np.uint8([[[hue, 210, 245]]])
    bgr = cv2.cvtColor(color, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def draw_tracks(frame: np.ndarray, tracks: Sequence[Track]) -> np.ndarray:
    out = frame.copy()
    for track in tracks:
        x1, y1, x2, y2 = track.xyxy.astype(int)
        color = id_color(track.track_id) if track.state == TrackState.TRACKED else (180, 180, 180)
        thickness = 2 if track.state == TrackState.TRACKED else 1
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        prefix = "ID" if track.state == TrackState.TRACKED else ("OCC" if track.is_occluded else "LOST")
        label = f"{prefix} {track.track_id} {track.score:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        y_text = max(0, y1 - th - baseline - 4)
        cv2.rectangle(out, (x1, y_text), (x1 + tw + 8, y_text + th + baseline + 6), color, -1)
        cv2.putText(
            out,
            label,
            (x1 + 4, y_text + th + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
    return out


def parse_source(source: str):
    if source.isdigit():
        return int(source)
    return source


def run_video(args: argparse.Namespace) -> None:
    detector = YOLODetector(
        weights=args.weights,
        device=args.device,
        imgsz=args.imgsz,
        conf=args.track_low_thresh,
        iou=args.det_iou,
        max_det=args.max_det,
        agnostic_nms=args.agnostic_nms,
        person_class=args.person_class,
    )
    cfg = LatenMOTConfig(
        track_high_thresh=args.track_high_thresh,
        track_low_thresh=args.track_low_thresh,
        new_track_thresh=args.new_track_thresh,
        stage1_min_iou=args.stage1_min_iou,
        stage2_min_iou=args.stage2_min_iou,
        reactivation_min_iou=args.reactivation_min_iou,
        appearance_thresh=args.appearance_thresh,
        active_appearance_thresh=args.active_appearance_thresh,
        reactivation_cost_thresh=args.reactivation_cost_thresh,
        track_buffer=args.track_buffer,
        draw_lost_frames=args.draw_lost_frames,
        motion_gate=args.motion_gate,
        lost_motion_gate=args.lost_motion_gate,
        motion_lambda=args.motion_lambda,
        occlusion_coverage_thresh=args.occlusion_coverage_thresh,
        occlusion_velocity_damping=args.occlusion_velocity_damping,
        occlusion_reset_alpha=args.occlusion_reset_alpha,
        occlusion_box_enlarge=args.occlusion_box_enlarge,
        visibility_momentum=args.visibility_momentum,
        lost_visibility_decay=args.lost_visibility_decay,
        occluded_visibility_decay=args.occluded_visibility_decay,
        output_visibility_thresh=args.output_visibility_thresh,
        lost_output_visibility_thresh=args.lost_output_visibility_thresh,
        use_deferred_birth=not args.no_deferred_birth,
        pending_confirm_hits=args.pending_confirm_hits,
        pending_max_misses=args.pending_max_misses,
        pending_min_iou=args.pending_min_iou,
        use_deferred_reactivation=not args.no_deferred_reactivation,
        reactivation_evidence_thresh=args.reactivation_evidence_thresh,
        reactivation_strong_quality=args.reactivation_strong_quality,
        reactivation_evidence_decay=args.reactivation_evidence_decay,
        min_box_area=args.min_box_area,
        use_reid=not args.no_reid,
        use_active_reid=not args.no_active_reid,
        reid_after_frames=args.reid_after_frames,
        feature_alpha=args.feature_alpha,
    )
    tracker = LatenMOTTracker(cfg)

    cap = cv2.VideoCapture(parse_source(args.source))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open source: {args.source}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and not math.isnan(fps) and fps > 0 else args.fps
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*args.fourcc)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create output video: {output_path}")

    pbar_total = frame_count if frame_count > 0 else None
    mot_rows: List[str] = []
    with tqdm(total=pbar_total, desc="LatenMOT", unit="frame") as pbar:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            detections = detector(frame)
            tracks = tracker.update(frame, detections)
            if args.save_mot:
                for track in tracks:
                    x, y, w, h = track.tlwh
                    mot_rows.append(
                        f"{tracker.frame_id},{track.track_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{track.score:.4f},-1,-1,-1"
                    )
            vis = draw_tracks(frame, tracks)
            writer.write(vis)
            pbar.update(1)

    cap.release()
    writer.release()
    print(f"Saved: {output_path}")
    if args.save_mot:
        mot_path = Path(args.save_mot)
        mot_path.parent.mkdir(parents=True, exist_ok=True)
        mot_path.write_text("\n".join(mot_rows) + ("\n" if mot_rows else ""), encoding="utf-8")
        print(f"Saved MOT tracks: {mot_path}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LatenMOT tracker for Kaggle/Colab GPU T4")
    parser.add_argument("--weights", default="best_crossval_v11_non_early_stop (1).pt")
    parser.add_argument("--source", required=True, help="Video path or webcam index, e.g. 0")
    parser.add_argument("--output", default="latenmot_output.mp4")
    parser.add_argument("--device", default="0", help="Use 0 on GPU T4, or cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--det-iou", type=float, default=0.85, help="YOLO NMS IoU; higher keeps more overlapping crowd boxes")
    parser.add_argument("--max-det", type=int, default=1000, help="Maximum detections per frame before tracking")
    parser.add_argument("--agnostic-nms", action="store_true", help="Class-agnostic NMS; usually leave off for one-class people tracking")
    parser.add_argument("--person-class", type=int, default=0, help="Set -1 to keep all classes")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--fourcc", default="mp4v")
    parser.add_argument("--save-mot", default="", help="Optional MOTChallenge txt output path")

    parser.add_argument("--track-high-thresh", type=float, default=0.45)
    parser.add_argument("--track-low-thresh", type=float, default=0.05)
    parser.add_argument("--new-track-thresh", type=float, default=0.55)
    parser.add_argument("--stage1-min-iou", type=float, default=0.18)
    parser.add_argument("--stage2-min-iou", type=float, default=0.08)
    parser.add_argument("--reactivation-min-iou", type=float, default=0.08)
    parser.add_argument("--appearance-thresh", type=float, default=0.42)
    parser.add_argument("--active-appearance-thresh", type=float, default=0.48)
    parser.add_argument("--reactivation-cost-thresh", type=float, default=0.72)
    parser.add_argument("--track-buffer", type=int, default=60)
    parser.add_argument("--draw-lost-frames", type=int, default=12)
    parser.add_argument("--motion-gate", type=float, default=18.0)
    parser.add_argument("--lost-motion-gate", type=float, default=35.0)
    parser.add_argument("--motion-lambda", type=float, default=0.15)
    parser.add_argument("--occlusion-coverage-thresh", type=float, default=0.45)
    parser.add_argument("--occlusion-velocity-damping", type=float, default=0.55)
    parser.add_argument("--occlusion-reset-alpha", type=float, default=0.08)
    parser.add_argument("--occlusion-box-enlarge", type=float, default=1.25)
    parser.add_argument("--visibility-momentum", type=float, default=0.75)
    parser.add_argument("--lost-visibility-decay", type=float, default=0.92)
    parser.add_argument("--occluded-visibility-decay", type=float, default=0.96)
    parser.add_argument("--output-visibility-thresh", type=float, default=0.16)
    parser.add_argument("--lost-output-visibility-thresh", type=float, default=0.22)
    parser.add_argument("--pending-confirm-hits", type=int, default=3)
    parser.add_argument("--pending-max-misses", type=int, default=2)
    parser.add_argument("--pending-min-iou", type=float, default=0.18)
    parser.add_argument("--reactivation-evidence-thresh", type=float, default=0.95)
    parser.add_argument("--reactivation-strong-quality", type=float, default=0.72)
    parser.add_argument("--reactivation-evidence-decay", type=float, default=0.65)
    parser.add_argument("--no-deferred-birth", action="store_true")
    parser.add_argument("--no-deferred-reactivation", action="store_true")
    parser.add_argument("--reid-after-frames", type=int, default=2)
    parser.add_argument("--feature-alpha", type=float, default=0.9)
    parser.add_argument("--min-box-area", type=float, default=12.0)
    parser.add_argument("--no-reid", action="store_true")
    parser.add_argument("--no-active-reid", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if args.person_class < 0:
        args.person_class = None
    if args.device != "cpu":
        try:
            import torch

            if not torch.cuda.is_available():
                print("CUDA is not available; falling back to CPU.")
                args.device = "cpu"
        except Exception:
            args.device = "cpu"
    run_video(args)


if __name__ == "__main__":
    main()
