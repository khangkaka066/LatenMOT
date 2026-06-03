#!/usr/bin/env python3
"""
LatenMOT: lightweight detector + motion + two-stage association + lost-track
reactivation + conditional appearance ReID.

Designed for Kaggle/Colab T4:
    python latenmot_tracker.py --weights "best_crossval_v11_non_early_stop (1).pt" \
        --source input.mp4 --output output.mp4 --device 0
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
    track_high_thresh: float = 0.55
    track_low_thresh: float = 0.12
    new_track_thresh: float = 0.65
    stage1_min_iou: float = 0.25
    stage2_min_iou: float = 0.12
    reactivation_min_iou: float = 0.08
    appearance_thresh: float = 0.42
    reactivation_cost_thresh: float = 0.72
    track_buffer: int = 45
    min_box_area: float = 12.0
    use_reid: bool = True
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

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h = max(1.0, float(mean[3]))
        std_pos = np.array(
            [
                self.std_weight_position * h,
                self.std_weight_position * h,
                1e-2,
                self.std_weight_position * h,
            ],
            dtype=np.float32,
        )
        std_vel = np.array(
            [
                self.std_weight_velocity * h,
                self.std_weight_velocity * h,
                1e-5,
                self.std_weight_velocity * h,
            ],
            dtype=np.float32,
        )
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        mean = self.motion_mat @ mean
        covariance = self.motion_mat @ covariance @ self.motion_mat.T + motion_cov
        return mean.astype(np.float32), covariance.astype(np.float32)

    def project(self, mean: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h = max(1.0, float(mean[3]))
        std = np.array(
            [
                self.std_weight_position * h,
                self.std_weight_position * h,
                1e-1,
                self.std_weight_position * h,
            ],
            dtype=np.float32,
        )
        innovation_cov = np.diag(np.square(std))
        projected_mean = self.update_mat @ mean
        projected_cov = self.update_mat @ covariance @ self.update_mat.T + innovation_cov
        return projected_mean.astype(np.float32), projected_cov.astype(np.float32)

    def update(
        self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        projected_mean, projected_cov = self.project(mean, covariance)
        kalman_gain = covariance @ self.update_mat.T @ np.linalg.inv(projected_cov)
        innovation = measurement - projected_mean
        new_mean = mean + kalman_gain @ innovation
        new_covariance = covariance - kalman_gain @ projected_cov @ kalman_gain.T
        return new_mean.astype(np.float32), new_covariance.astype(np.float32)


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
        self.smooth_feature = detection.feature

    @property
    def tlwh(self) -> np.ndarray:
        return xyah_to_tlwh(self.mean[:4])

    @property
    def xyxy(self) -> np.ndarray:
        return tlwh_to_xyxy(self.tlwh)

    def predict(self, kf: KalmanFilter) -> None:
        if self.state != TrackState.TRACKED:
            self.mean[7] = 0
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)

    def update(self, detection: Detection, frame_id: int, kf: KalmanFilter, alpha: float) -> None:
        self.mean, self.covariance = kf.update(self.mean, self.covariance, tlwh_to_xyah(detection.tlwh))
        self.score = detection.score
        self.cls = detection.cls
        self.state = TrackState.TRACKED
        self.frame_id = frame_id
        self.last_seen = frame_id
        self.lost_since = None
        if detection.feature is not None:
            if self.smooth_feature is None:
                self.smooth_feature = detection.feature
            else:
                self.smooth_feature = alpha * self.smooth_feature + (1.0 - alpha) * detection.feature
                norm = np.linalg.norm(self.smooth_feature)
                if norm > 1e-12:
                    self.smooth_feature = self.smooth_feature / norm

    def mark_lost(self, frame_id: int) -> None:
        self.state = TrackState.LOST
        self.lost_since = frame_id if self.lost_since is None else self.lost_since

    def mark_removed(self) -> None:
        self.state = TrackState.REMOVED


class LatenMOTTracker:
    def __init__(self, config: LatenMOTConfig) -> None:
        self.cfg = config
        self.kf = KalmanFilter()
        self.reid = ColorHistReID()
        self.tracks: List[Track] = []
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

        iou_dist = iou_cost(lost_tracks, detections)
        cost = np.full_like(iou_dist, fill_value=np.inf, dtype=np.float32)
        for r, track in enumerate(lost_tracks):
            gap = self.frame_id - track.last_seen
            for c, det in enumerate(detections):
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
        return cost

    def update(self, frame: np.ndarray, detections: List[Detection]) -> List[Track]:
        self.frame_id += 1

        detections = [
            d
            for d in detections
            if d.tlwh[2] * d.tlwh[3] >= self.cfg.min_box_area and d.tlwh[2] > 1 and d.tlwh[3] > 1
        ]
        high_dets = [d for d in detections if d.score >= self.cfg.track_high_thresh]
        low_dets = [d for d in detections if self.cfg.track_low_thresh <= d.score < self.cfg.track_high_thresh]

        for track in self.tracks:
            if track.state in (TrackState.TRACKED, TrackState.LOST):
                track.predict(self.kf)

        active_tracks = [t for t in self.tracks if t.state == TrackState.TRACKED]
        lost_tracks = [t for t in self.tracks if t.state == TrackState.LOST]

        # Stage 1: confident detections against active tracks using motion/IoU.
        stage1_matches, unmatched_active, unmatched_high = linear_assignment_with_threshold(
            iou_cost(active_tracks, high_dets), max_cost=1.0 - self.cfg.stage1_min_iou
        )
        for trk_idx, det_idx in stage1_matches:
            self._ensure_features(frame, high_dets, [det_idx])
            active_tracks[trk_idx].update(high_dets[det_idx], self.frame_id, self.kf, self.cfg.feature_alpha)

        # Stage 2: low confidence detections can keep existing tracks alive.
        remaining_active = [active_tracks[i] for i in unmatched_active]
        stage2_matches, unmatched_remaining_active, _ = linear_assignment_with_threshold(
            iou_cost(remaining_active, low_dets), max_cost=1.0 - self.cfg.stage2_min_iou
        )
        for trk_idx, det_idx in stage2_matches:
            self._ensure_features(frame, low_dets, [det_idx])
            remaining_active[trk_idx].update(low_dets[det_idx], self.frame_id, self.kf, self.cfg.feature_alpha)

        for trk_idx in unmatched_remaining_active:
            remaining_active[trk_idx].mark_lost(self.frame_id)

        # Lost re-activation: try old IDs before creating new ones.
        remaining_high = [high_dets[i] for i in unmatched_high]
        self._ensure_features(frame, remaining_high, range(len(remaining_high)))
        lost_tracks = [t for t in self.tracks if t.state == TrackState.LOST]
        react_matches, unmatched_lost, unmatched_remaining_high = linear_assignment_with_threshold(
            self._reactivation_cost(lost_tracks, remaining_high),
            max_cost=self.cfg.reactivation_cost_thresh,
        )
        for trk_idx, det_idx in react_matches:
            lost_tracks[trk_idx].update(
                remaining_high[det_idx], self.frame_id, self.kf, self.cfg.feature_alpha
            )

        # Only create a new ID after active and lost tracks had a chance to match.
        for det_idx in unmatched_remaining_high:
            det = remaining_high[det_idx]
            if det.score >= self.cfg.new_track_thresh:
                self._ensure_features(frame, remaining_high, [det_idx])
                self.tracks.append(Track(det, self.frame_id, self.kf))

        for trk_idx in unmatched_lost:
            track = lost_tracks[trk_idx]
            if self.frame_id - track.last_seen > self.cfg.track_buffer:
                track.mark_removed()

        self.tracks = [t for t in self.tracks if t.state != TrackState.REMOVED]
        return [t for t in self.tracks if t.state == TrackState.TRACKED and t.last_seen == self.frame_id]


class YOLODetector:
    def __init__(
        self,
        weights: str,
        device: str,
        imgsz: int,
        conf: float,
        person_class: Optional[int],
    ) -> None:
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self.person_class = person_class

    def __call__(self, frame: np.ndarray) -> List[Detection]:
        classes = None if self.person_class is None else [self.person_class]
        result = self.model.predict(
            frame,
            imgsz=self.imgsz,
            conf=self.conf,
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
        color = id_color(track.track_id)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"ID {track.track_id} {track.score:.2f}"
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
        reactivation_cost_thresh=args.reactivation_cost_thresh,
        track_buffer=args.track_buffer,
        min_box_area=args.min_box_area,
        use_reid=not args.no_reid,
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
    parser.add_argument("--person-class", type=int, default=0, help="Set -1 to keep all classes")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--fourcc", default="mp4v")
    parser.add_argument("--save-mot", default="", help="Optional MOTChallenge txt output path")

    parser.add_argument("--track-high-thresh", type=float, default=0.55)
    parser.add_argument("--track-low-thresh", type=float, default=0.12)
    parser.add_argument("--new-track-thresh", type=float, default=0.65)
    parser.add_argument("--stage1-min-iou", type=float, default=0.25)
    parser.add_argument("--stage2-min-iou", type=float, default=0.12)
    parser.add_argument("--reactivation-min-iou", type=float, default=0.08)
    parser.add_argument("--appearance-thresh", type=float, default=0.42)
    parser.add_argument("--reactivation-cost-thresh", type=float, default=0.72)
    parser.add_argument("--track-buffer", type=int, default=45)
    parser.add_argument("--reid-after-frames", type=int, default=2)
    parser.add_argument("--feature-alpha", type=float, default=0.9)
    parser.add_argument("--min-box-area", type=float, default=12.0)
    parser.add_argument("--no-reid", action="store_true")
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
