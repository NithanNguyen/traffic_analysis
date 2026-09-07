# Real-Time Traffic Flow Analysis & Vehicle Speed Estimation

**Detects, tracks and classifies vehicles from a live RTSP stream to produce per-vehicle speed (km/h), road-occupancy density (%) and a rule-based traffic state, rendered on a live OpenCV dashboard.**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/Detector-YOLOv8--Nano-orange)](https://github.com/ultralytics/ultralytics)
[![Tracker](https://img.shields.io/badge/Tracker-ByteTrack-green)](https://github.com/ifzhang/ByteTrack)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

![Live analytics dashboard](assets/images/dashboard.png)

---

## Context

Course project for **CS311.Q11 — AI Programming Techniques**.

Manual traffic monitoring does not scale, and congestion decisions need data at the moment congestion happens rather than after the fact. This project targets three measurable outputs from a single fixed camera: vehicle counts by class, movement speed in km/h, and how much of the road surface is actually occupied.

---

## Features

- **Multi-class vehicle detection** — car, bus, truck and motorbike, using a YOLOv8-Nano model fine-tuned on a Vietnamese traffic dataset, at a 0.25 confidence threshold.
- **Multi-object tracking** — persistent IDs via ByteTrack, which associates low-confidence boxes instead of discarding them, keeping tracks alive through partial occlusion.
- **Speed estimation** — per-zone perspective transform maps the bounding-box ground point into metres; speed is smoothed by an exponential moving average (α = 0.1).
- **Occupancy density** — aggregate real-world footprint of vehicles inside the ROI divided by the ROI's real area, expressed as a percentage.
- **Rule-based state classification** — seven states from Empty Road to Traffic Jam, derived from smoothed speed and occupancy.
- **Multithreaded ingestion** — a producer-consumer capture thread with a bounded queue drops stale frames so the display stays synchronised with the stream.

---

## Architecture

```text
RTSP stream
   └─▶ Threaded video capture  (producer-consumer, queue maxsize=10, drop-oldest)
         └─▶ YOLOv8 detection + ByteTrack tracking
               ├─▶ Speed estimator      (perspective transform → EMA filter)
               ├─▶ Occupancy calculator (Σ vehicle area / ROI area)
               └─▶ Traffic analytics    (state classification + trend buffer)
                     └─▶ Dashboard renderer (OpenCV overlay + Matplotlib chart)
```

The capture thread exists to solve a throughput mismatch. `cv2.VideoCapture.read()` blocks, and the camera produces frames faster than inference consumes them, so a sequential loop accumulates 2–5 seconds of display lag. Reading on a separate thread into a 10-frame queue that discards its oldest entry when full keeps the processed frame close to the current one. RTSP transport is forced to TCP to avoid packet-loss artifacts.

---

## Methodology

### Speed estimation

Four source points in pixel space and their real-world rectangle are passed to `cv2.getPerspectiveTransform()` to build a matrix per speed zone. The bottom-centre of each bounding box is taken as the vehicle's ground contact point, mapped into metres, and compared against its previous mapped position. Speed is Euclidean distance over elapsed wall-clock time, converted with `× 3.6`.

Raw frame-to-frame speed is noisy because bounding boxes jitter and frames are occasionally dropped, so four guards are applied:

| Guard | Value | Purpose |
|---|---|---|
| Minimum time delta | `Δt > 0.02 s` | Avoids division by a near-zero interval |
| Minimum track age | `> 5 frames` | Suppresses speed output from newly-formed, unstable tracks |
| Outlier rejection | `> 100 km/h` → hold previous value | Discards spikes caused by ID switches or box jumps |
| Smoothing | `v = 0.9·v_prev + 0.1·v_new` | EMA filter; removes display flicker |

Track history is discarded when a vehicle leaves the zone, and reset when it crosses into a different zone.

### Occupancy density

Each detected class is assigned a nominal real-world footprint rather than being measured, since bounding-box area in pixels does not translate to road area under perspective:

| Class | Assumed area |
|---|---|
| `motor` | 2 m² |
| `car` | 8 m² (also the fallback for unmatched classes) |
| `truck` | 30 m² |
| `bus` | 35 m² |

Occupancy is the sum of footprints for vehicles inside the ROI polygons, divided by the total real area of those polygons, capped at 100 %.

### Traffic state classification

Evaluated on the smoothed average speed and smoothed occupancy, in order; the first matching rule wins.

| Condition | State |
|---|---|
| 0 vehicles in ROI and occupancy < 1 % | Empty Road |
| Speed < 5 km/h and occupancy > 15 % | Stopped / Red Light |
| Occupancy > 45 % and speed < 20 km/h | Traffic Jam |
| Occupancy > 45 % | High Density |
| Speed < 25 km/h | Slow Traffic |
| Occupancy > 15 % | Moderate |
| Otherwise | Free Flow |

The averages are computed over rolling buffers — the last 50 speed samples above 5 km/h, and the last 30 occupancy samples — so the displayed state does not flip on a single frame. The trend chart redraws every 3 seconds over a 30-point history to limit CPU cost.

---

## Tech Stack

| Component | Technology |
|---|---|
| Detection | YOLOv8-Nano (Ultralytics), fine-tuned on a Vietnamese traffic dataset |
| Tracking | ByteTrack |
| Image and video processing | OpenCV (FFmpeg backend) |
| Charting | Matplotlib, `Agg` backend |
| Concurrency | Python `threading` + `queue` |
| Stream simulation | MediaMTX (RTSP server) + FFmpeg (looped publisher) |

---

## Requirements

| Requirement | Notes |
|---|---|
| Python | 3.9 or newer |
| GPU | Optional; CUDA is used automatically when available |
| MediaMTX | Only needed to simulate an RTSP source from a local video file |
| FFmpeg | Only needed to publish that file to the RTSP server |
| Model weights | `traffic4.pt` in the project root; falls back to stock `yolov8n.pt` if absent |

---

## Quick Start

```bash
git clone https://github.com/NithanNguyen/traffic_analysis.git && cd traffic_analysis
pip install ultralytics opencv-python numpy matplotlib
cd tools && ./mediamtx        # terminal 1: start the RTSP server
./ffmpeg -re -stream_loop -1 -i TestVideo1.mp4 -c:v copy -rtsp_transport tcp -f rtsp rtsp://localhost:8554/live_stream   # terminal 2
python main.py                # terminal 3
```

Press `q` in the display window to exit. Place the MediaMTX and FFmpeg executables under `tools/` and the test videos under `video/` — neither is distributed with the repository.

---

## Repository Structure

```
traffic_analysis/
├── main.py               # Entry point: RTSP → detection/tracking → analytics → dashboard
├── test.py               # Experimental variant of main.py, different default config key
├── my_tracker.yaml       # ByteTrack parameters
├── traffic_config.json   # Per-video speed and occupancy zone calibration
├── traffic4.pt           # Fine-tuned YOLOv8 weights (supplied separately)
├── tools/                # MediaMTX and FFmpeg binaries (ffmpeg.exe is git-ignored)
└── video/                # Source test videos (*.mp4 is git-ignored)
```

---

## Configuration

**`traffic_config.json`** — keyed by video filename. Each key holds two lists:

| Field | Meaning |
|---|---|
| `speed_zones[].points` | Four pixel coordinates defining the zone quadrilateral |
| `speed_zones[].real_w` / `real_h` | Real dimensions in metres, used to build the perspective matrix |
| `occupancy_zones[].points` | ROI polygon in pixel coordinates |
| `occupancy_zones[].real_w` / `real_h` | Real dimensions in metres, used as the occupancy denominator |

**`my_tracker.yaml`** — ByteTrack association parameters:

| Parameter | Effect |
|---|---|
| `track_high_thresh` / `track_low_thresh` | Confidence bands for initialising versus retaining a track |
| `new_track_thresh` | Minimum confidence to spawn a new track ID |
| `track_buffer` | Frames a lost track is kept before deletion |
| `match_thresh`, `fuse_score` | Detection-to-track association matching |

**Constants in `main.py`** — `SHOW_SPEED_ZONES` and `SHOW_OCCUPANCY_ZONES` toggle the zone overlays; `TARGET_HEIGHT` (720) and `DASHBOARD_WIDTH` (450) set the display geometry; `VEHICLE_REAL_AREAS` holds the per-class footprints listed above.

---

## Results

**Performance.** The pipeline sustains roughly 7–9 FPS end to end on consumer hardware with a discrete laptop GPU. Perceived latency stays low because the capture thread discards stale frames rather than queueing them, so the displayed frame tracks the live stream even when inference falls behind the source frame rate.

**Detection.** The fine-tuned YOLOv8-Nano model separates the four target classes under night, rain and dense-traffic conditions. No quantitative evaluation (mAP, precision/recall per class) was run, so this is a qualitative observation from the demo footage rather than a measured result.

---

## Limitations

- **Manual calibration.** Speed and occupancy zones are hand-aligned point by point. Any change in camera angle invalidates the perspective matrices and requires re-calibration.
- **Hardcoded configuration key.** `main.py` loads zones for the literal key `"TestVideo3.mp4"` while streaming from the RTSP URL; `test.py` uses `"TestVideo1.mp4"`. Running against a different source requires editing the call or adding a matching key to `traffic_config.json`.
- **Assumed vehicle footprints.** Occupancy uses fixed per-class areas, so a compact car and a large SUV contribute identically.
- **Environmental degradation.** Detection accuracy drops in heavy rain, dense fog and severe occlusion.
- **No edge deployment.** The model is not quantised or pruned, so it is not suited to low-power embedded hardware as-is.
- **Flow analysis only.** There is no licence-plate recognition (ANPR), so individual vehicles cannot be identified.

---

## Roadmap

1. **Edge deployment** — quantisation and pruning to run on embedded devices at lower cost and latency.
2. **Auto-calibration** — lane-detection to place and align ROI zones without manual point selection.
3. **Centralised management** — a web application and database aggregating historical data across multiple cameras.

---

## Acknowledgements

Built on [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics), [ByteTrack](https://github.com/ifzhang/ByteTrack), [OpenCV](https://opencv.org/) and [MediaMTX](https://github.com/bluenviron/mediamtx).

## License

Released under the MIT License.
