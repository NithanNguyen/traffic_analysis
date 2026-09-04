# Real-Time Traffic Flow Analysis & Vehicle Speed Estimation

A computer-vision pipeline that detects, tracks, and classifies vehicles from a live (or simulated RTSP) video stream to estimate per-vehicle speed, road occupancy density, and overall traffic state in real time.

## Features

- **Multi-class vehicle detection** — car, bus, truck, and motorbike detection using a fine-tuned YOLOv8 model.
- **Multi-object tracking** — persistent ID assignment across frames via ByteTrack, which retains low-confidence detections to keep tracks alive under partial occlusion.
- **Speed estimation (km/h)** — pixel-to-real-world coordinate mapping via perspective transform, computed per calibrated speed zone, with an exponential moving average (EMA) filter to suppress jitter.
- **Occupancy density estimation (%)** — road-area occupancy derived from the aggregate real-world footprint of tracked vehicles within defined ROI zones.
- **Rule-based traffic state classification** — combines smoothed average speed and occupancy to label the current state (e.g., Free Flow, Moderate, Slow, High Density, Traffic Jam, Stopped, Empty Road).
- **Live analytics dashboard** — on-screen overlay with per-class vehicle counts, average speed, and an occupancy trend chart, rendered with OpenCV and Matplotlib.
- **Multithreaded video ingestion** — a producer-consumer capture thread decouples frame reading from AI inference, dropping stale frames to keep the pipeline in near real time.

## Architecture

```
RTSP Stream
   │
   ▼
Threaded Video Capture (producer-consumer, bounded queue, frame-drop)
   │
   ▼
YOLOv8 Detection + ByteTrack Multi-Object Tracking
   │
   ├──▶ Speed Estimator (perspective transform + EMA smoothing)
   ├──▶ Occupancy Calculator (per-class footprint area / ROI area)
   │
   ▼
Traffic Analytics (rule-based state classification + trend buffering)
   │
   ▼
Dashboard Rendering (OpenCV overlay + Matplotlib chart)
```

The capture thread exists to solve a throughput mismatch: camera input runs at ~30 FPS while inference runs at ~20 FPS. A bounded queue (`maxsize=10`) with automatic drop of the oldest frame keeps the displayed video close to real time instead of accumulating lag.

## Tech Stack

| Component | Technology |
|---|---|
| Object detection | YOLOv8 (Ultralytics), fine-tuned on a Vietnamese traffic dataset |
| Multi-object tracking | ByteTrack |
| Image/video processing | OpenCV |
| Charting | Matplotlib (`Agg` backend) |
| Concurrency | Python `threading` (producer-consumer pattern) |
| Stream simulation | MediaMTX + FFmpeg (RTSP server + looped video publishing) |

## Project Structure

```
traffic_analysis/
├── main.py                # Main entry point (RTSP → detection/tracking → analytics → dashboard)
├── test.py                 # Experimental variant of main.py (different default zone config)
├── my_tracker.yaml         # ByteTrack tracker parameters
├── traffic_config.json     # Per-video speed/occupancy zone definitions (calibration data)
├── traffic4.pt              # Fine-tuned YOLOv8 weights (not tracked in Git)
├── tools/
│   ├── mediamtx.exe        # Local RTSP server (not tracked in Git)
│   └── ffmpeg.exe          # Video-to-RTSP publisher (not tracked in Git)
└── video/
    └── *.mp4               # Source test videos (not tracked in Git)
```

## Installation

**Prerequisites:** Python 3.9+, and optionally a CUDA-capable GPU for faster inference.

```bash
pip install ultralytics opencv-python numpy matplotlib
```

Download [MediaMTX](https://github.com/bluenviron/mediamtx) and [FFmpeg](https://ffmpeg.org/download.html), and place the executables under `tools/`.

Place the fine-tuned model weights (`traffic4.pt`) in the project root. If absent, the script falls back to the stock `yolov8n.pt` weights.

## Usage

1. **Start the RTSP server:**
   ```bash
   cd tools
   .\mediamtx.exe
   ```
2. **Publish a test video as an RTSP stream** (loops indefinitely):
   ```bash
   cd tools
   .\ffmpeg -re -stream_loop -1 -i TestVideo1.mp4 -c:v copy -rtsp_transport tcp -f rtsp rtsp://localhost:8554/live_stream
   ```
3. **Run the pipeline:**
   ```bash
   python main.py
   ```
   Press `q` to exit the display window.

## Configuration

**`traffic_config.json`** — maps a video/stream key to calibration zones:
- `speed_zones`: polygon `points` (pixel coordinates) plus the corresponding real-world `real_w` and `real_h` (in meters), used to build the perspective-transform matrix for speed estimation.
- `occupancy_zones`: polygon `points` plus real-world dimensions, used as the denominator for road-occupancy percentage.

**`my_tracker.yaml`** — ByteTrack parameters:
- `track_high_thresh` / `track_low_thresh`: confidence thresholds for initializing vs. retaining a track.
- `new_track_thresh`: minimum confidence required to spawn a new track ID.
- `track_buffer`: number of frames a lost track is kept alive before being discarded.
- `match_thresh`, `fuse_score`: association matching parameters between detections and existing tracks.

## Results

<img width="1852" height="773" alt="Screenshot from 2026-09-04 11-44-14" src="https://github.com/user-attachments/assets/766072dc-35d8-4264-82a8-e881f18f6b02" />

- Sustains **~7-9 FPS** on consumer-grade hardware with a discrete laptop GPU.
- The fine-tuned YOLOv8-Nano model reliably classifies the four target vehicle classes under challenging conditions (night, rain, dense traffic).
- The multithreaded capture design keeps end-to-end latency low, maintaining visual synchronization with the live stream.
