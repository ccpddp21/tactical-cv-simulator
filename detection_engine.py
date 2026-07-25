# detection_engine.py
# YOLOv8-powered object detection engine
# This is the component that "sees" -- identifying objects in each video frame

import cv2
import numpy as np
from ultralytics import YOLO
from dataclasses import dataclass, field
from typing import Optional
import time


# Data Structures

@dataclass
class Detection:
    """
    A single object detection result from one video frame.
    This is the core data unit flowing through the entire pipeline.
    """
    class_id: int       # YOLO class index (0=person, 2=car, etc.)
    class_name: str     # Human-readable class name
    confidence: float   # How confident YOLO is (0.0 to 1.0)
    bbox: tuple         # Bounding box: (x1, y1, x2, y2) in pixels
    center: tuple       # Center point: (cx, cy)
    area: float         # Bounding box area in pixels
    threat_level:str    # Our classification: LOW, MEDIUM, HIGH
    timestamp: float    # When this detection occurred


@dataclass
class FrameResult:
    """
    All detections from a single video frame.
    """
    frame_number: int
    timestamp: float
    detections: list # List of Detection objects
    processing_time_ms: float
    frame_width: int
    frame_height: int

# Threat Classification
# Maps YOLO object classes to threat levels for our simulation
# In a real defense system this would be far more sophisticated
# and based on context, behavior analysis, and mission parameters
THREAT_CLASSIFICATION = {
    # People
    "person": "MEDIUM",
    # Vehicles -- differentiated by type
    "car": "MEDIUM",
    "truck": "HIGH",
    "bus": "MEDIUM",
    "motorcycle": "MEDIUM",
    "bicycle": "LOW",
    # Aircraft
    "airplane": "HIGH",
    "helicopter": "HIGH", # YOLOv8 may classify as airplane
    # Other
    "boat": "MEDIUM",
    "train": "LOW",
    "traffic light": "LOW",
    "backpack": "LOW",
    "handbag": "LOW",
    "suitcase": "MEDIUM", # Could contain items of interest
    # Default for unclassified
    "default": "LOW"
}

# Color coding for visualization
THREAT_COLORS = {
    "HIGH": (0, 0, 255), # Red (BGR format for OpenCV)
    "MEDIUM": (0, 165, 255), # Orange
    "LOW": (0, 255, 0), # Green
    "UNKNOWN": (128, 128, 128) # Gray
}


# Detection Engine

class DetectionEngine:
    """
    YOLOv8-based real-time object detection engine.

    This class wraps the YOLO model and handles:
    - Model initialization and configuration
    - Per-frame inference
    - Result parsing and classification
    - Performance metrics tracking
    """

    def __init__(
        self,
        model_size: str = "n", # n, s, m, l, x
        confidence_threshold: float = 0.4,
        iou_threshold: float = 0.45, # Non-maximum suppression threshold
        target_classes: Optional[list] = None # Filter to specific classes only
    ):
        """
        Initialize the detection engine.

        model_size: YOLO model variant (n=fastest, x=most accurate)
        confidence_threshold: Minimum confidence to report a detection
        iou_threshold: How much overlap to allow before suppressing duplicate boxes
        target_classes: If set, only detect these class names
        """
        self.model_path = f"yolov8{model_size}.pt"
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.target_classes = target_classes

        # Performance tracking
        self.frame_count = 0
        self.total_processing_time = 0
        self.total_detections = 0

        print(f"Loading YOLOv8{model_size} model...")
        self.model = YOLO(self.model_path)
        self.class_names = self.model.names # Dict of {id: class_name}
        print(f"[check] Model loaded. Classes available: {len(self.class_names)}")

        # Filter class IDs if target_classes specified
        self.target_class_ids = None
        if target_classes:
            self.target_class_ids = [
                k for k, v in self.class_names.items()
                if v in target_classes
            ]
            print(f"[check] Filtering to {len(self.target_class_ids)} target classes: {target_classes}")

    def classify_threat(self, class_name: str) -> str:
        """
        Classify detected object into a threat level.
        This is where domain knowledge meets computer vision.
        """
        return THREAT_CLASSIFICATION.get(class_name, THREAT_CLASSIFICATION["default"])
    
    def process_frame(self, frame: np.ndarray, frame_number: int) -> FrameResult:
        """
        Run YOLOv8 inference on a single video frame.

        frame: numpy array of shape (height, width, 3) in BGR format
        frame_number: sequential frame index
        Returns: FrameResult with all detections
        """
        start_time = time.time()
        timestamp = time.time()
        height, width = frame.shape[:2]

        # Run YOLO inference
        # verbose=False suppresses per-frame console output
        results = self.model(
            frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            classes=self.target_class_ids,
            verbose=False
        )

        detections = []

        # Parse YOLO results
        # results[0] because we process one frame at a time
        for result in results[0].boxes:

            # Extract bounding box coordinates
            x1, y1, x2, y2 = map(int, result.xyxy[0].tolist())
            
            # Extract class and confidence
            class_id = int(result.cls[0])
            class_name = self.class_names[class_id]
            confidence = float(result.conf[0])
           
            # Calculate center point and area
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            area = (x2 - x1) * (y2 - y1)
            
            # Classify threat level
            threat_level = self.classify_threat(class_name)
            detection = Detection(
                class_id=class_id,
                class_name=class_name,
                confidence=confidence,
                bbox=(x1, y1, x2, y2),
                center=(cx, cy),
                area=area,
                threat_level=threat_level,
                timestamp=timestamp
            )
            detections.append(detection)

        processing_time = (time.time() - start_time) * 1000 # Convert to ms
        
        # Update performance stats
        self.frame_count += 1
        self.total_processing_time += processing_time
        self.total_detections += len(detections)
    
        return FrameResult(
            frame_number=frame_number,
            timestamp=timestamp,
            detections=detections,
            processing_time_ms=processing_time,
            frame_width=width,
            frame_height=height
        )

    def annotate_frame(self, frame: np.ndarray, frame_result: FrameResult) -> np.ndarray:
        """
        Draw detection bounding boxes and labels on the frame.
        Returns annotated frame for display.
        """
        annotated = frame.copy()

        for det in frame_result.detections:
            x1, y1, x2, y2 = det.bbox
            color = THREAT_COLORS.get(det.threat_level, THREAT_COLORS["UNKNOWN"])
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Draw label background
            label = f"{det.class_name} {det.confidence:.2f} [{det.threat_level}]"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(
                annotated,
                (x1, y1 - label_size[1] - 8),
                (x1 + label_size[0], y1),
                color, -1 # -1 fills the rectangle
            )

            # Draw label text
            cv2.putText(
                annotated, label,
                (x1, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1 # White text
            )

            # Draw center point
            cv2.circle(annotated, det.center, 3, color, -1)

        # Draw performance overlay
        avg_fps = (
            1000 / (self.total_processing_time / self.frame_count)
            if self.frame_count > 0 else 0
        )
        cv2.putText(
            annotated,
            f"FPS: {avg_fps:.1f} | Detections: {len(frame_result.detections)}",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (255, 255, 0), 2
        )

        return annotated

    def get_performance_stats(self) -> dict:
        """Return performance metrics for monitoring."""
        if self.frame_count == 0:
            return {}

        avg_processing_ms = self.total_processing_time / self.frame_count
        return {
            "frames_processed": self.frame_count,
            "avg_processing_ms": round(avg_processing_ms, 1),
            "avg_fps": round(1000 / avg_processing_ms, 1),
            "total_detections": self.total_detections,
            "avg_detections_per_frame": round(
                self.total_detections / self.frame_count, 1
            )
        }


# Video Processor

class VideoProcessor:
    """
    Handles reading video from various sources and feeding frames
    to the detection engine.

    Supports: video files, webcam, and generated synthetic frames
    """

    def __init__(self, source):
        """
        source: path to video file, 0 for webcam, or 'synthetic' for simulation
        """
        self.source = source
        self.cap = None
        self.frame_number = 0

        if source != "synthetic":
            self.cap = cv2.VideoCapture(source)
            if not self.cap.isOpened():
                raise ValueError(f"Cannot open video source: {source}")

            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            print(f"[check] Video loaded: {self.width}x{self.height} @ {self.fps:.1f}fps")
            print(f" Total frames: {self.total_frames}")

    def read_frame(self):
        """Read the next frame. Returns (success, frame)."""
        if self.cap is None:
            return False, None

        ret, frame = self.cap.read()
        if ret:
            self.frame_number += 1
        return ret, frame

    def release(self):
        """Release video resources."""
        if self.cap:
            self.cap.release()

    @property
    def progress_pct(self) -> float:
        """Current position as percentage of total video."""
        if not self.cap or self.total_frames <= 0:
            return 0
        return (self.frame_number / self.total_frames) * 100

# Quick Test
if __name__ == "__main__":
    import sys

    video_source = sys.argv[1] if len(sys.argv) > 1 else "test_video.mp4"

    print(f"\nTesting detection engine on: {video_source}")
    print("=" * 50)

    engine = DetectionEngine(
        model_size="n",
        confidence_threshold=0.4
    )
    
    processor = VideoProcessor(video_source)
    
    # Process 30 frames as a test
    for i in range(30):
        ret, frame = processor.read_frame()
        if not ret:
            break

        result = engine.process_frame(frame, i)

        if result.detections:
            print(f"Frame {i}: {len(result.detections)} detections")
            for det in result.detections:
                print(
                    f" - {det.class_name} "
                    f"(conf: {det.confidence:.2f}, "
                    f"threat: {det.threat_level})"
            )
        else:
            print(f"Frame {i}: No detections")

    processor.release()
    stats = engine.get_performance_stats()
    print(f"\nPerformance: {stats}")
    print("\n[check] Detection engine test complete")