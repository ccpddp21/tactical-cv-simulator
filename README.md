# tactical-cv-simulator
Computer Vision Real-Time Detection Simulation Situational Awareness

## Technical Architecture
- **Detection:** YOLOv8n real-time object detection (80 COCO classes)
- **Tracking:** SORT algorithm with Kalman filter prediction
- **Association:** Hungarian algorithm for optimal detection-to-track matching
- **Simulation:** Zone-based state machine with alert generation
- **Dashboard:** Streamlit real-time visualization

## Defense-Relevant Concepts Demonstrated
- Multi-object tracking with persistent entity IDs
- Zone-based geofencing and alert logic
- Threat classification pipeline
- Real-time situational awareness dashboard
- Entity state machines (DETECTED -> TRACKED -> LOST)