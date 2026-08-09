# app.py -- Tactical Situational Awareness Dashboard

import streamlit as st
import cv2
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import time
from PIL import Image

from detection_engine import DetectionEngine, VideoProcessor, THREAT_COLORS
from tracking_engine import SORTTracker, KalmanBoxTracker
from simulation_environment import TacticalSimulation, ZoneType

# Page Config
st.set_page_config(
    page_title="Tactical CV Simulator",
    page_icon="[Target]",
    layout="wide",
    intial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .metric-high { color: #ff4444; font-weight: bold; }
    .metric-medium { color: #ff8c00; font-weight: bold; }
    .metric-low { color: #44ff44; font-weight: bold; }
    .alert-critical { background: #ff000022; border-left: 3px solid red; padding: 5px; }
    .alert-warning { background: #ff8c0022; border-left: 3px solid orange; padding: 5px; }
    .alert-info { background: #0088ff22; border-left: 3px solid blue; padding: 5px; }
</style>
""", unsafe_allow_html=True)

# Session State
if "engine" not in st.session_state:
    st.session_state.engine = None
if "tracker" not in st.session_state:
    st.session_state.tracker = None
if "simulation" not in st.session_state:
    st.session_state.simulation = None
if "processor" not in st.session_state:
    st.session_state.processor = None
if "running" not in st.session_state:
    st.session_state.running = False
if "all_alerts" not in st.session_state:
    st.session_state.all_alerts = []

# Sidebar Controls
with st.sidebar:
    st.title("[Target] Tactical CV Simulator")
    st.caption("Computer Vision Object Tracking Situational Awareness")

    st.divider()
    st.header("[Settings] Configuration")

    video_source = st.text_input(
        "Video Source",
        value="test_video.mp4",
        help="Path to video file. Enter '0' for webcam."
    )

    model_size = st.select_slider(
        "YOLO Model Size",
        options=["n", "s", "m"],
        value="n",
        help="n=fastest, m=most accurate"
    )

    confidence = st.slider(
        "Detection Confidence",
        min_value=0.2,
        max_value=0.9,
        value=0.4,
        step=0.05
    )

    max_track_age = st.slider(
        "Track Max Age (frames)",
        min_value=5,
        max_value=30,
        value=10,
        help="Frames before losing a track without detection"
    )

    show_zones = st.checkbox("Show Zone Overlays", value=True)
    show_trajectories = st.checkbox("Show Trajectories", value=True)

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        start_btn = st.button(
            "[Play] Start",
            use_container_width=True,
            type="primary"
        )
    with col2:
        stop_btn = st.button(
            "[Stop] Stop",
            use_container_width=True
        )

    if start_btn:
        # Initialize all components
        try:
            KalmanBoxTracker.count = 0  # Reset tracker IDs

            src = int(video_source) if video_source == "0" else video_source
            st.session_state.processor = VideoProcessor(src)

            st.session_state.engine = DetectionEngine(
                model_size=model_size,
                confidence_threshold=confidence
            )

            st.session_state.tracker = SORTTracker(
                max_age=max_track_age
            )

            w = st.session_state.processor.width
            h = st.session_state.processor.height
            st.session_state.simulation = TacticalSimulation(width=w, height=h)

            st.session_state.running = True
            st.session_state.all_alerts = []
            st.success("[check] System initialized")
        except Exception as e:
            st.error(f"Init error: {e}")

    if stop_btn:
        st.session_state.running = False
        if st.session_state.processor:
            st.session_state.processor.release()
        st.info("System stopped")

    st.divider()

    # Architecture info
    st.header("[Brain] System Architecture")
    st.markdown("""
    **Pipeline:**
    1. Video frame capture
    2. YOLOv8 object detection
    3. SORT multi-object tracking
    4. Zone-based alert generation
    5. Dashboard visualization

    **Models:**
    - Detection: YOLOv8n (COCO 80 classes)
    - Tracking: SORT + Kalman Filter

    **Alert Types:**
    - [RED] CRITICAL: Exclusion zone / High threat
    - [ORANGE] WARNING: Restricted zone entry
    - [BLUE] INFO: Monitoring zone / Dwell
    """)

    # Main Dashboard
    st.title("[Target] Tactical Situaltional Awareness System")

    if not st.session_state.running:
        st.info(
            "Configure settings in the siderbar and click **[Play] Start** "
            "to begin processing."
        )

        # Show architecture diagram
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("""
            ### Input
            Video file or webcam stream fed frame by frame into the pipeline
            """)
        with col2:
            st.markdown("""
            ### [Target] Detection
            YOLOv8 identifies objects and classifies them with bounding boxes
            """)
        with col3:
            st.markdown("""
            ### [Signal] Tracking
            SORT + Kalman Filter assigns persistent IDs across frames
            """)
        with col4:
            st.markdown("""
            ### [Map] Simulation
            Zone logic generates alerts and tracks entity behavior over time
            """)
    else:
        # Create dashboard layout
        col_video, col_metrics = st.columns([3, 1])

        video_placeholder = col_video.empty()

        with col_metrics:
            metric_placeholder = st.empty()
        
        # Alert log and tactical map
        col_alerts, col_map = st.columns([1, 1])

        with col_alerts:
            st.subheader("[ALERT] Alert Log")
            alert_placeholder = st.empty()

        with col_map:
            st.subheader("[Map] Entity Distribution")
            map_placeholder = st.empty()

        # Zone occupancy
        zone_placeholder = st.empty()

        # Processing Loop
        frame_count = 0

        while st.session_state.running:
            ret, frame = st.session_state.processor.read_frame()

            if not ret:
                st.info("Video complete.")
                st.session_state.running = False
                break

            frame_count += 1

            # Skip frames for speed (process every other frame)
            if frame_count % 2 != 0:
                continue

            # Detection
            frame_result = st.session_state.engine.process_frame(
                frame, frame_count
            )

            # Tracking
            confirmed_tracks = st.session_state.tracker.update(
                frame_result.detections
            )

            # Simulation Update
            new_alerts = st.session_state.simulation.update(confirmed_tracks)
            st.session_state.all_alerts.extend(new_alerts)

            # Annotate Frame
            annotated = frame.copy()

            # Draw zone overlays
            if show_zones:
                h_frame, w_frame = annotated.shape[:2]
                zone_overlay = annotated.copy()

                zone_colors_map = {
                    ZoneType.EXCLUSION: (0, 0, 100),
                    ZoneType.RESTRICTED: (0, 60, 120),
                    ZoneType.MONITORING: (0, 80, 80),
                    ZoneType.CLEAR: (0, 60, 0)
                }

                for zone in st.session_state.simulation.zones:
                    x1p, y1p, x2p, y2p = zone.bounds
                    x1 = int(x1p * w_frame)
                    y1 = int(y1p * h_frame)
                    x2 = int(x2p * w_frame)
                    y2 = int(y2p * h_frame)

                    fill_color = zone_colors_map.get(zone.zone_type, (50, 50, 50))
                    cv2.rectangle(zone_overlay, (x1, y1), (x2, y2), fill_color, -1)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), zone.color, 2)
                    cv2.putText(
                        annotated, zone.name,
                        (x1 + 5, y1 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        zone.color, 1
                    )

                cv2.addWeighted(zone_overlay, 0.2, annotated, 0.8, 0, annotated)

            # Draw tracked entities
            for tracker in confirmed_tracks:
                bbox = tracker.get_bbox()
                center = tracker.get_center()
                color = THREAT_COLORS.get(tracker.threat_level, (128, 128, 128))

                x1, y1, x2, y2 = bbox
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                label = f"ID:{tracker.id} {tracker.class_name} [{tracker.threat_level}]"
                cv2.putText(
                    annotated,
                    label,
                    (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    color, 1
                )

            # Draw trajectory
            if show_trajectories and len(tracker.history) > 1:
                for i in range(1, len(tracker.history)):
                    cv2.line(
                        annotated,
                        tracker.history[i - 1],
                        tracker.history[i],
                        color,
                        1
                    )

        # Display frame
        rgb_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        video_placeholder.image(pil_image, use_container_width=True)

        # Dashboard Data
        data = st.session_state.simulation.get_dashboard_data()
        perf = st.session_state.engine.get_performance_stats()

        # Metrics panel
        with metric_placeholder.container():
            st.metric("[Target] Active Tracks", data["active_entity_count"])
            st.metric("[RED] High Threat", data["threat_counts"]["HIGH"])
            st.metric("[ORANGE] Medium", data["threat_counts"]["MEDIUM"])
            st.metric("[GREEN] Low", data["threat_counts"]["LOW"])
            st.metric("[ALERT] Total Alerts", data["total_alerts"])
            st.metric(" FPS", perf.get("avg_fps", 0))
            st.metric(
                " Runtime",
                f"{data['elapsed_seconds']:.0f}s"
            )

        # Alert log
        with alert_placeholder.container():
            if data["recent_alerts"]:
                for alert in data["recent_alerts"][:8]:
                    severity_emoji = {
                        "CRITICAL": "[RED]",
                        "WARNING": "[ORANGE]",
                        "INFO": "[BLUE]"
                    }.get(alert.severity, "")

                    t = time.strftime(
                        "%H:%M:%S",
                        time.localtime(alert.timestamp)
                    )
                    st.markdown(
                        f"{severity_emoji} `{t}` -- {alert.message}"
                    )
            else:
                st.caption("No alerts yet...")

        # Entity distribution chart
        if  data["class_counts"]:
            with map_placeholder.container():
                df_classes = pd.DataFrame(
                    list(data["class_counts"].items()),
                    columns=["Class", "Count"]
                )
                fig = px.bar(
                    df_classes,
                    x="Class",
                    y="Count",
                    color="Count",
                    title="Detected Entity Classes",
                    color_continuous_scale="Reds"
                )
                fig.update_layout(
                    height=250,
                    margin=dict(t=30, b=0, l=0, r=0)
                )
                st.plotly_chart(fig, use_container_width=True)

        # Zone occupancy
        with zone_placeholder.container():
            if data["zone_occupancy"]:
                cols = st.columns(len(data["zone_occupancy"]))
                for col, (zone_name, count) in zip(
                    cols, data["zone_occupancy"].items()
                ):
                    col.metric(zone_name, count)

        time.sleep(0.01)