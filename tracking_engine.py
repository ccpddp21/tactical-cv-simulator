# tracking_engine.py
# Multi-object tracking using the SORT algorithm
# SORT = Simple Online and Realtime Tracking
#
# The core insight: detection tells you WHAT is in each frame.
# Tracking tells you WHICH detection is the same entity across frames.
# Without tracking, you can't monitor movement or generate meaningful alerts.

import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
from dataclasses import dataclass, field
from typing import Optional
import time


# Kalman Filter for Single Object Tracking

class KalmanBoxTracker:
    """
    Tracks a signle object using a Kalman filter.

    The kalman filter is a mtathemeatical algortithm that:
    1. Predicts where an object will be next frame (based on velocity)
    2. Updates that prediction with the actual detection
    3. Handles noise and uncertantity in both prediction and measurement

    State vector: [x, y, scale, aspect_ratio, dx, dy, d_scale, d_aspect]
    Where dx, dy are velocitites and d_scale/d_aspect are scale change rates.
    """

    count = 0 # Class-level counter for uniwue IDs

    def __init__(self, bbox: tuple, class_name: str, threat_level: str):
        """
        Initialize tracker for a new detection.
        bbox: (x1, y1, x2, y2)
        class_name: Name of the detected object class (verify)
        threat_level: Level of threat associated with the object (verify)
        """
        # Give this track a unique ID
        KalmanBoxTracker.count += 1
        self.id = KalmanBoxTracker.count
        self.class_name = class_name
        self.threat_level = threat_level
        self.hit_streak = 0         # Consecutive frames with a detection
        self.time_since_update = 0  # Frames since last detection
        self.age = 0                # Total frames since track has existed

        #History for trajectory visualization
        self.history = []           # Past center position
        self.first_seen = time.time()
        self.last_seen = time.time()

        # Zone tracking
        self.current_zone = None    # Current zone the object is in
        self.zone_entry_time = None # Time when the object entered the current zone

        # Initialize Kalman filter
        # State dimension: 8 (x, y, scale, aspect, vx, vy, vs, va)
        # Measurement dimension: 4 (x, y, scale, aspect)
        self.kf = KalmanFilter(dim_x=8, dim_z=4)

        # State transition matrix (how state evelves each frame)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ])

        # Measurement function (which parts of state we can observe)
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ])

        # Measurement noise (how much we trust the detector)
        self.kf.R[2:, 2:] *= 10

        # Initial uncertainty for velocities
        self.kf.P[4:, 4:] *= 1000
        self.kf.P *= 10

        # Process noise (now much objects can change unexpectedly)
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        # Initialize state from first detection
        self.kf.x[:4] = self. _bbox_to_z(bbox)

    def _bbox_to_z(self, bbox: tuple) -> np.ndarray:
        """
        Convert bounding box (x1, y1, x2, y2) to Kalman state (x, y, s, r).
        """
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        x = x1 + w / 2      # Center x
        y = y1 + h / 2      # Center y
        s = w * h           # Scale (area)
        r = w / h if h > 0 else 1   # Aspect ratio
        return np.array([[x], [y], [s], [r]])

    def _z_to_bbox(self, z: np.ndarray) -> tuple:
        """
        Convert Kalman state back to bounding box.
        """
        x, y, s, r = z[0][0], z[1][0], z[2][0], z[3][0]
        w = np.sqrt(abs(s * r))
        h = abs(s / w) if w > 0 else 0
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2
        return (x1, y1, x2, y2)

    def predict(self) -> tuple:
        """
        Predict the object's position in the next frame.
        This is the Kalman filter's forward prediction step.
        """
        # Prevent negative scale
        if self.kf.x[6] + self.kf.x[2] <= 0:
            self.kf.x[6] = 0

        self.kf.predict()
        self.age += 1

        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

        # Return predicted bounding box
        predicted_bbox = self._z_to_bbox(self.kf.x[:4])
        return predicted_bbox

    def update(self, bbox: tuple, class_name: str, threat_level: str):
        """
        Update tracker with a new detection.
        This is the Kalman filter's measurement update step.
        """
        self.time_since_update = 0
        self.hit_streak += 1
        self.last_seen = time.time()
        self.class_name = class_name        # Update class in case it changed
        self.threat_level = threat_level

        self.kf.update(self._bbox_to_z(bbox))

        # Record trajectory
        x1, y1, x2, y2 = bbox
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        self.history.append(center)

        # Keep trajectory history manageable (last 50 positions)
        if len(self.history) > 50:
            self.history.pop(0)

    def get_bbox(self) -> tuple:
        """
        Get current estimated bounding box.
        """
        return self._z_to_bbox(self.kf.x[:4])

    def get_center(self) -> tuple:
        """
        Get current estimated center position.
        """
        bbox = self.get_bbox()
        x1 = bbox[0]
        y1 = bbox[1]
        x2 = bbox[2]
        y2 = bbox[3]
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def dwell_time(self) -> float:
        """
        How long this entity has been tracked in seconds.
        """
        return time.time() - self.first_seen


# IoU Calculation

def calculate_iou(bbox1: tuple, bbox2: tuple) -> float:
    """
    Calculate Intersection over Union between two bounding boxes.
    
    IoU = area of intersection / area of union
    IoU = 1.0: perfect overlap
    IoU = 0.0: no overlap

    Used to associate detections with existing tracks:
    high IoU = likely the same object
    """
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0


# SORT Tracker

class SORTTracker:
    """
    SORT: Simple Online and Realtime Tracking
    
    Algorithm:
    1. For each frame, get detections from the detection engine
    2. Predict where each existing track will be (Kalman prediction)
    3. Match predictions to new detections using IoU (Hungarian algorithm)
    4. Update mateched tracks with new detection positions
    5. Create new tracks for unmatched detections
    6. Delete tracks that haven't been detected for max_age frames
    """

    def __init__(
        self,
        max_age: int = 10,          # Frames to keep a track without detection
        min_hits: int = 3,          # Minimum detections before reporting a track
        iou_threshold: float = 0.3  # Minimum IoU to associate detection with track
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []          # List of active KalmanBoxTracker objects
        self.frame_count = 0

        # For analysis and alerts
        self.all_tracks_history = {}    # id - > track info for logging

    def update(self, detections: list) -> list:
        """
        Update all tracks with new frame's detections.

        detections: List of Detection objects from DetectionEngine
        Returns: List of active tracked entities with their current state
        """
        self.frame_count += 1

        # Step 1: Prediect new positions for all existing tracks
        predicted_bboxes = []
        to_delete = []

        for tracker in self.trackers:
            pred = tracker.predict()
            # Check if prediction is valid (no NaN values)
            if not any(np.isnan(p) for p in pred):
                predicted_bboxes.append(pred)
            else:
                to_delete.append(tracker)

        # Remove invalid trackers
        for t in to_delete:
            self.trackers.remove(t)

        # Step 2: Build IoU matrix between predictions and detections
        det_bboxes = [d.bbox for d in detections]

        matched_indices = []
        unmatched_detections = list(range(len(detections)))
        unmatched_trackers = list(range(len(self.trackers)))

        if len(predicted_bboxes) > 0 and len(det_bboxes) > 0:
            # Build cost matrix (1 - IoU, since we minimize cost)
            iou_matrix = np.zeros((len(predicted_bboxes), len(det_bboxes)))
            for t_idx, pred in enumerate(predicted_bboxes):
                for d_idx, det_bbox in enumerate(det_bboxes):
                    iou_matrix[t_idx, d_idx] = calculate_iou(pred, det_bbox)

            # Hungarian algortithm finds optimal assignment
            row_ind, col_ind = linear_sum_assignment(-iou_matrix)

            matched_index = []
            for r, c in zip(row_ind, col_ind):
                if iou_matrix[r, c] >= self.iou_threshold:
                    matched_indices.append((r, c))
                    if r in unmatched_trackers:
                        unmatched_trackers.remove(r)
                    if c in unmatched_detections:
                        unmatched_detections.remove(c)

        # Step 3: Update matched trackers
        for t_idx, d_idx in matched_indices:
            det = detections[d_idx]
            self.trackers[t_idx].update(
                det.bbox, det.class_name, det.threat_level
            )

        # Step 4: Create new trackers for unmatched detections
        for d_idx in unmatched_detections:
            det = detections[d_idx]
            new_tracker = KalmanBoxTracker(
                det.bbox, det.class_name, det.threat_level
            )
            self.trackers.append(new_tracker)
            self.all_tracks_history[new_tracker.id] = {
                "id": new_tracker.id,
                "class": det.class_name,
                "threat": det.threat_level,
                "first_seen": time.time()
            }

        # Step 5 Remove old tracks
        self.trackers = [
            t for t in self.trackers
            if t.time_since_update <= self.max_age
        ]

        # Step 6: Return confirmed tracks (seen enough times to report)
        confirmed = []
        for tracker in self.trackers:
            if tracker.hit_streak >= self.min_hits or self.frame_count <= self.min_hits:
                confirmed.append(tracker)

        return confirmed

    def get_active_count(self) -> dict:
        """
        Count active tracks by threat level.
        """
        for tracker in self.trackers:
            if tracker.time_since_update == 0:      # Currently detected
                counts[tracker.threat_level] = counts.get(
                    tracker.threat_level, 0
                ) + 1
                counts["TOTAL"] += 1
        return counts