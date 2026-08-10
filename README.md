# Tactical Computer Vision Simulator

A real-time computer vision and tactical simulation portfolio project that combines **YOLOv8 object detection**, **multi-object tracking**, **simulation state management**, **zone-based alerting**, and a **Streamlit situational-awareness dashboard**.

The application processes video input frame-by-frame, detects and classifies objects, maintains persistent entity IDs across frames, tracks entity movement, evaluates tactical-zone interactions, and displays live alerts and operational metrics.

## Features

- **Real-Time Object Detection** — YOLOv8 inference over video streams
- **Video Processing & Annotation** — OpenCV frame capture, bounding boxes, labels, and overlays
- **Multi-Object Tracking** — SORT-style tracking with persistent entity IDs
- **Trajectory Prediction** — Kalman-filter-based object state prediction
- **Detection-to-Track Association** — IoU matching with the Hungarian algorithm
- **Threat Classification** — Maps detected object classes to LOW, MEDIUM, or HIGH threat levels
- **Simulation State Machine** — Tracks entity states such as `UNKNOWN`, `DETECTED`, `TRACKED`, and `LOST`
- **Zone-Based Alerting** — Generates events when tracked entities interact with defined tactical zones
- **Situational Awareness Dashboard** — Live video, entity data, threat metrics, zone occupancy, tactical visualization, and alert logging
- **Multiple Input Sources** — Supports video files and webcam input

## Technical Architecture

```text
Video File / Webcam
        |
        v
+-----------------------+
|   Detection Engine    |
| YOLOv8 + OpenCV       |
| boxes/classes/threats |
+-----------+-----------+
            |
            v
+-----------------------+
|    Tracking Engine    |
| SORT / Kalman Filter  |
| IoU + Hungarian Match |
+-----------+-----------+
            |
            v
+-----------------------+
| Simulation State Mgr  |
| entity states / zones |
| alert generation      |
+-----------+-----------+
            |
            v
+-----------------------+
| Streamlit Dashboard   |
| live feed / map / log |
| metrics / entities    |
+-----------------------+
```

## Technologies Used

| Technology | Purpose |
| --- | --- |
| **Python 3.9+** | Core application language |
| **Ultralytics YOLOv8** | Real-time object detection and classification |
| **OpenCV** | Video capture, frame processing, bounding boxes, labels, and image operations |
| **NumPy** | Numerical operations and array processing |
| **SORT** | Multi-object tracking approach |
| **FilterPy** | Kalman filter implementation for track prediction |
| **SciPy** | Hungarian assignment via `linear_sum_assignment` |
| **Streamlit** | Real-time situational-awareness web dashboard |
| **Plotly** | Interactive charts and tactical visualizations |
| **Pandas** | Dashboard data preparation and tabular analytics |
| **scikit-image** | Image-processing utilities |
| **Pillow** | Image handling |
| **python-dotenv** | Environment-variable support |
| **Requests** | HTTP/network utility support |

## Project Structure

```text
tactical-cv-simulator/
├── app.py                       # Streamlit situational-awareness dashboard
├── detection_engine.py          # YOLOv8 detection and video processing
├── tracking_engine.py           # SORT / Kalman multi-object tracking
├── simulation_environment.py    # Entity states, zones, and alert logic
├── utils.py                     # Shared utilities
├── test_video.mp4               # Optional local test video
├── data/                        # Application/test data
└── logs/                        # Generated logs
```

## Prerequisites

- **Python 3.9 or newer**
- `pip`
- A local MP4 video for testing, or a webcam
- Internet access during initial setup if the YOLO model weights have not already been downloaded

Check your Python version:

```bash
python --version
```

## Installation

### 1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd tactical-cv-simulator
```

### 2. Create a virtual environment

**macOS / Linux**

```bash
python -m venv venv
source venv/bin/activate
```

**Windows PowerShell**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Windows Command Prompt**

```cmd
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

If the repository contains a `requirements.txt` file:

```bash
pip install -r requirements.txt
```

Otherwise, install the dependencies directly:

```bash
pip install ultralytics opencv-python-headless numpy streamlit plotly pandas scipy filterpy scikit-image pillow python-dotenv requests
```

### 4. Download/verify the YOLOv8 model

The model can download automatically on first use. To verify the development model explicitly:

```bash
python -c "from ultralytics import YOLO; model = YOLO('yolov8n.pt'); print('YOLOv8n downloaded successfully')"
```

The project uses **YOLOv8n (Nano)** as the recommended development model because it prioritizes inference speed and has relatively low compute requirements.

## Test Video

Place an MP4 video in the project root and name it:

```text
test_video.mp4
```

A video containing pedestrians, vehicles, or other common objects works well for demonstrating the detection and tracking pipeline.

The dashboard can also accept webcam input by using `0` as the video source.

## Running the Application

Start the complete situational-awareness dashboard with:

```bash
streamlit run app.py
```

Streamlit will print the local application address in the terminal and normally open it in your browser.

In the application sidebar:

1. Set **Video Source** to `test_video.mp4`, or enter `0` for a webcam.
2. Keep the YOLO model size on **`n`** for the fastest development experience.
3. Adjust the confidence threshold if needed.
4. Click **Start**.
5. Observe object detections, bounding boxes, threat classifications, persistent tracking IDs, zone activity, and generated alerts.

## Testing the Detection Engine Independently

Run the detector without the full Streamlit dashboard:

```bash
python detection_engine.py test_video.mp4
```

The script processes a sample of frames and prints detected object classes, confidence values, and threat levels.

If the video produces no detections, try lowering the configured confidence threshold (the project guide suggests `0.25` as a troubleshooting value).

You can also verify that YOLO loads correctly:

```bash
python -c "from ultralytics import YOLO; m = YOLO('yolov8n.pt'); print(m.info())"
```

## Detection Pipeline

For each video frame, the detection engine:

1. Reads a frame with OpenCV.
2. Sends the frame through YOLOv8.
3. Extracts bounding-box coordinates, class IDs, class names, and confidence scores.
4. Calculates each detection's center point and bounding-box area.
5. Assigns a simulation threat classification.
6. Passes the resulting detections into the tracking pipeline.
7. Annotates frames for visualization and records processing metrics.

## Multi-Object Tracking

The tracking engine follows the **SORT (Simple Online and Realtime Tracking)** approach.

It combines:

- **Kalman filtering** to predict where an existing tracked object should appear next
- **Intersection over Union (IoU)** to measure overlap between predicted and detected bounding boxes
- **Hungarian assignment** to determine optimal one-to-one detection/track associations
- **Persistent IDs** so the same entity can be recognized across multiple frames
- Track aging/removal when an entity is no longer detected

This changes the system from simple per-frame detection into a temporal surveillance pipeline capable of reasoning about movement.

## Simulation & Alerting

Tracked entities are passed into a tactical simulation layer that maintains entity state and evaluates zone interactions.

The project demonstrates state transitions such as:

```text
UNKNOWN -> DETECTED -> TRACKED -> LOST
```

The simulation can generate alerts based on tracked entities interacting with defined geographic/tactical zones, enabling concepts such as perimeter monitoring, geofencing, dwell detection, and threat-aware event logging.

## Dashboard

The Streamlit dashboard serves as the system's situational-awareness interface and brings the individual pipeline components together.

It includes concepts such as:

- Live annotated video
- Active tracked entities
- Threat-level metrics
- Detected entity class statistics
- Zone occupancy
- Tactical visualization
- Time-stamped alert/event log
- Runtime configuration controls

## Defense-Relevant Concepts Demonstrated

This portfolio project demonstrates:

- Computer vision inference integration
- Real-time ML processing pipelines
- Persistent multi-object tracking
- Kalman filter prediction
- IoU-based association
- Hungarian assignment
- Simulation state machines
- Zone-based geofencing and alert logic
- Threat-classification pipelines
- Trajectory and entity-state management
- Real-time situational-awareness visualization
- Modular separation between detection, tracking, simulation, and UI components

## Notes

This project is a **training and portfolio simulation**. The threat-classification rules are explicit application-level mappings for demonstration purposes; they are not a real-world threat assessment system.

Performance depends on the selected YOLO model, video resolution, CPU/GPU capabilities, and number of objects being processed. Start with `yolov8n` while developing, then evaluate larger YOLOv8 variants if additional detection accuracy is required.
