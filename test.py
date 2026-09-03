import cv2
import numpy as np
from ultralytics import YOLO
import time
import os
import math
import sys
import threading
import queue
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from collections import deque
from pathlib import Path
import json  

# --- CẤU HÌNH ---
MODEL_PATH = "traffic4.pt"    
# [THAY ĐỔI] URL RTSP
VIDEO_SOURCE = "rtsp://localhost:8554/live_stream" 
JSON_CONFIG_FILE = "traffic_config.json" 
TRACKER_CONFIG = "my_tracker.yaml" 

# --- CỜ HIỂN THỊ (VISUALIZATION FLAGS) ---
SHOW_OCCUPANCY_ZONES = True  
SHOW_SPEED_ZONES = True      

# Kích thước chuẩn hóa
TARGET_HEIGHT = 720
DASHBOARD_WIDTH = 450

# --- COLORS & STYLE (Giữ nguyên) ---
C_BG = (30, 30, 30)         
C_CARD = (50, 50, 50)       
C_TEXT_MAIN = (230, 230, 230)
C_TEXT_SUB = (180, 180, 180)
C_ACCENT = (0, 191, 255)    
C_GREEN = (100, 200, 100)
C_YELLOW = (0, 215, 255)    
C_RED = (80, 80, 255)       

CLASS_COLOR_MAP = {
    'car': (0, 255, 127),     
    'bus': (0, 140, 255),    
    'truck': (203, 192, 255),
    'motor': (255, 255, 0)    
}

# --- CẤU HÌNH DIỆN TÍCH XE (M2) ---
VEHICLE_REAL_AREAS = {
    'car': 8.0,      
    'bus': 35.0,     
    'truck': 30.0,   
    'motor': 2.0     
}
DEFAULT_VEHICLE_AREA = 8.0
ROI_COLOR = (0, 165, 255)
ROI_ALPHA = 0.2

# --- CLASS: THREADED VIDEO CAPTURE ---
class VideoCaptureThreading:
    def __init__(self, src):
        self.src = src
        # Force TCP để ổn định hơn cho RTSP
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        self.cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        
        if not self.cap.isOpened():
            print(f"[ERROR] Failed to open connection to: {src}")
            return
        
        self.q = queue.Queue(maxsize=10)
        self.stop_event = threading.Event()
        self.t = threading.Thread(target=self._reader)
        self.t.daemon = True
        self.t.start()

    def _reader(self):
        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            
            if not self.q.empty():
                try: self.q.get_nowait()
                except queue.Empty: pass
            self.q.put(frame)

    def read(self):
        try: 
            # Wait a bit longer for RTSP packets
            return True, self.q.get(timeout=1)
        except queue.Empty:
            return False, None

    def release(self):
        self.stop_event.set()
        self.t.join()
        self.cap.release()

# --- HÀM ĐỌC JSON ---
def load_config_from_json(json_path, video_name):
    """
    Đọc file JSON, tìm cấu hình của video_name, 
    chuyển đổi List -> Numpy Array với kiểu dữ liệu đúng.
    """
    if not os.path.exists(json_path):
        print(f"[ERROR] Không tìm thấy file cấu hình: {json_path}")
        return None, None

    with open(json_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print(f"[ERROR] File JSON bị lỗi cú pháp.")
            return None, None

    video_config = data.get(video_name)
    if not video_config:
        print(f"[ERROR] Chưa có cấu hình cho video: {video_name} trong file JSON.")
        return None, None

    # Xử lý Speed Zones (cần float32 cho cv2.getPerspectiveTransform)
    speed_zones_out = []
    for item in video_config.get("speed_zones", []):
        poly = np.array(item["points"], dtype=np.float32)
        speed_zones_out.append( (poly, item["real_w"], item["real_h"]) )

    # Xử lý Occupancy Zones (cần int32 cho cv2.polylines/pointPolygonTest)
    occupancy_zones_out = []
    for item in video_config.get("occupancy_zones", []):
        poly = np.array(item["points"], dtype=np.int32)
        occupancy_zones_out.append( {
            'poly': poly, 
            'real_w': item["real_w"], 
            'real_h': item["real_h"]
        })

    print(f"[INFO] Đã tải cấu hình cho: {video_name}")
    return speed_zones_out, occupancy_zones_out

# --- CLASS: TỐC ĐỘ (Giữ nguyên logic) ---
class SpeedEstimator:
    def __init__(self):
        self.track_history = {} 
        self.zones = [] 

    def add_zone(self, zone_id, src_points, real_width, real_length):
        dst_points = np.array([[0, 0], [real_width, 0], [real_width, real_length], [0, real_length]], dtype=np.float32)
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        self.zones.append({'id': zone_id, 'poly': src_points.astype(np.int32), 'matrix': matrix})

    def transform_point(self, point, matrix):
        p = np.array([[[point[0], point[1]]]], dtype=np.float32)
        dst = cv2.perspectiveTransform(p, matrix)
        return dst[0][0]

    def estimate_speed(self, track_id, box_coords, video_timestamp):
        x1, y1, x2, y2 = box_coords
        center_bottom = (int((x1 + x2) / 2), int(y2))
        
        active_zone = None
        for zone in self.zones:
            if cv2.pointPolygonTest(zone['poly'], center_bottom, False) >= 0:
                active_zone = zone
                break
        
        if active_zone is None: 
            if track_id in self.track_history:
                del self.track_history[track_id]
            return 0

        current_time = video_timestamp 
        current_real_pos = self.transform_point(center_bottom, active_zone['matrix'])
        speed_kmh = 0
        
        if track_id in self.track_history:
            prev_data = self.track_history[track_id]
            if prev_data.get('zone_id') == active_zone['id']:
                dist = math.sqrt((current_real_pos[0] - prev_data['pos'][0])**2 + (current_real_pos[1] - prev_data['pos'][1])**2)
                time_diff = current_time - prev_data['time']
                tracked_frames = prev_data.get('frames', 0) + 1

                if time_diff > 0.02: 
                    temp_speed = (dist / time_diff) * 3.6
                    if tracked_frames > 5:
                        last_speed = prev_data.get('speed', 0)
                        if temp_speed > 100: speed_kmh = last_speed
                        else:
                            if last_speed == 0: speed_kmh = temp_speed
                            else: speed_kmh = 0.9 * last_speed + 0.1 * temp_speed
                    else: speed_kmh = 0
            else: tracked_frames = 1
        else: tracked_frames = 1

        self.track_history[track_id] = {
            'pos': current_real_pos, 
            'time': current_time, 
            'speed': speed_kmh, 
            'zone_id': active_zone['id'],
            'frames': tracked_frames
        }
        return int(speed_kmh)

    def draw_zones(self, frame):
        for zone in self.zones:
            cv2.polylines(frame, [zone['poly']], True, (0, 0, 200), 1)

# --- CLASS: ANALYTICS (Occupancy) ---
FLOW_WINDOW_SECONDS = 3
MAX_HISTORY_POINTS = 30

class TrafficAnalytics:
    def __init__(self):
        self.start_time = time.time()
        self.last_update_time = time.time()
        
        # Lưu trữ lịch sử Occupancy
        self.occupancy_history = deque(maxlen=MAX_HISTORY_POINTS) 
        self.timestamps = deque(maxlen=MAX_HISTORY_POINTS)
        
        self.speed_buffer = deque(maxlen=50)
        self.occupancy_buffer = deque(maxlen=30) 
        
        # Cấu hình Matplotlib
        plt.style.use('dark_background')
        # Tăng kích thước figsize một chút để đủ chỗ cho số liệu trục
        self.fig, self.ax = plt.subplots(figsize=(6, 3.5), dpi=80) 
        self.fig.patch.set_facecolor('#2D2D2D') # Màu nền bao quanh
        self.chart_image = None
        
        # Khởi tạo giá trị 0
        for i in range(MAX_HISTORY_POINTS): 
            self.timestamps.append(0)
            self.occupancy_history.append(0.0)

    def update_stats(self, new_count, speed_list, current_occupancy_rate):
        # Buffer dữ liệu để tính trung bình
        for s in speed_list:
            if s > 5: self.speed_buffer.append(s)
        self.occupancy_buffer.append(current_occupancy_rate)

        now = time.time()
        # Cập nhật biểu đồ mỗi 3 giây
        if now - self.last_update_time >= FLOW_WINDOW_SECONDS:
            avg_occ = self.get_avg_occupancy()
            
            self.occupancy_history.append(avg_occ)
            self.timestamps.append(int(now - self.start_time))
            
            self.last_update_time = now
            try: self._render_chart()
            except: pass

    def get_avg_speed(self):
        if not self.speed_buffer: return 0
        return int(sum(self.speed_buffer) / len(self.speed_buffer))
    
    def get_avg_occupancy(self):
        if not self.occupancy_buffer: return 0.0
        return sum(self.occupancy_buffer) / len(self.occupancy_buffer)

    def _render_chart(self):
        self.ax.clear()
        
        # Dữ liệu
        x = list(self.timestamps)
        y = list(self.occupancy_history)
        
        # Màu nền bên trong biểu đồ
        self.ax.set_facecolor('#2D2D2D')
        
        # Vẽ Line Chart
        self.ax.fill_between(x, y, color='#00ADB5', alpha=0.2)
        self.ax.plot(x, y, color='#00ADB5', linewidth=2, label='Occupancy')
        
        # --- CẤU HÌNH TRỤC (AXES) ---
        # 1. Giới hạn trục Y (0-100%)
        self.ax.set_ylim(0, 100)
        
        # 2. Tiêu đề và Label
        self.ax.set_title('Occupancy Trend (%)', color='#EEEEEE', fontsize=10, fontweight='bold', pad=10)
        self.ax.set_ylabel('Density (%)', color='#C0C0C0', fontsize=8)
        self.ax.set_xlabel('Time (s)', color='#C0C0C0', fontsize=8)
        
        # 3. Tùy chỉnh màu sắc số (Ticks) và đường viền (Spines)
        self.ax.tick_params(axis='x', colors='#C0C0C0', labelsize=7)
        self.ax.tick_params(axis='y', colors='#C0C0C0', labelsize=7)
        
        # Đổi màu đường viền khung biểu đồ cho đỡ gắt
        for spine in self.ax.spines.values():
            spine.set_edgecolor('#555555')

        # 4. Lưới (Grid)
        self.ax.grid(True, color='#444444', linestyle=':', linewidth=0.5, alpha=0.7)
        
        # Tự động căn chỉnh lề để không bị mất chữ
        self.fig.tight_layout()
        
        # Chuyển đổi sang định dạng ảnh cho OpenCV
        canvas = FigureCanvas(self.fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        self.chart_image = cv2.cvtColor(np.asarray(buf), cv2.COLOR_RGBA2BGR)

    def draw_dashboard(self, vehicle_counts, density_str, status_text, status_color, fps):
        H, W = TARGET_HEIGHT, DASHBOARD_WIDTH
        dash = np.zeros((H, W, 3), dtype=np.uint8)
        dash[:] = C_BG

        # HEADER
        cv2.rectangle(dash, (0, 0), (W, 60), (40, 40, 40), -1)
        cv2.putText(dash, "TRAFFIC ANALYTICS", (20, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, C_ACCENT, 1)
        cv2.putText(dash, f"FPS: {fps}", (W - 100, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 1)

        y_off = 80
        # STATUS
        self._draw_card(dash, 20, y_off, W-40, 100, C_CARD)
        cv2.circle(dash, (70, y_off + 50), 25, status_color, -1)
        font_scale_status = 0.9 if len(status_text) < 10 else 0.7
        cv2.putText(dash, status_text, (120, y_off + 60), cv2.FONT_HERSHEY_DUPLEX, font_scale_status, (255, 255, 255), 1)
        y_off += 120

        # METRICS
        card_w = (W - 50) // 2
        # CARD 1
        self._draw_card(dash, 20, y_off, card_w, 80, C_CARD)
        cv2.putText(dash, "OCCUPANCY", (35, y_off + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_TEXT_SUB, 1)
        cv2.putText(dash, density_str, (30, y_off + 60), cv2.FONT_HERSHEY_DUPLEX, 1.0, C_TEXT_MAIN, 1)

        # CARD 2
        avg_spd = self.get_avg_speed()
        self._draw_card(dash, 20 + card_w + 10, y_off, card_w, 80, C_CARD)
        cv2.putText(dash, "AVG KM/H", (20 + card_w + 25, y_off + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_TEXT_SUB, 1)
        spd_color = C_GREEN if avg_spd > 40 else (C_YELLOW if avg_spd > 20 else C_RED)
        cv2.putText(dash, str(avg_spd), (20 + card_w + 25, y_off + 60), cv2.FONT_HERSHEY_DUPLEX, 1.2, spd_color, 1)
        y_off += 90

        # COUNTS
        self._draw_card(dash, 20, y_off, W-40, 120, C_CARD)
        row_y = y_off + 30
        col_x = 35
        for i, (k, v) in enumerate(vehicle_counts.items()):
            color = CLASS_COLOR_MAP.get(k, (255,255,255))
            cv2.circle(dash, (col_x, row_y), 5, color, -1)
            cv2.putText(dash, f"{k.upper()}: {v}", (col_x + 15, row_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_TEXT_MAIN, 1)
            if i % 2 == 0: col_x += 180
            else: col_x = 35; row_y += 30
        y_off += 140

        # CHART (Vẽ hình đã render)
        if self.chart_image is not None:
            target_w = W - 40
            h_c, w_c = self.chart_image.shape[:2]
            target_h = int(h_c * (target_w / w_c))
            
            # Đảm bảo không vẽ tràn màn hình
            if y_off + target_h < H:
                chart_resized = cv2.resize(self.chart_image, (target_w, target_h))
                dash[y_off:y_off+target_h, 20:20+target_w] = chart_resized

        return dash

    def _draw_card(self, img, x, y, w, h, color):
        cv2.rectangle(img, (x, y), (x + w, y + h), color, -1)

# --- MAIN ---
def main():
    if not os.path.exists(MODEL_PATH) and not os.path.exists("yolov8n.pt"):
        print("Đang tải model mặc định..."); model = YOLO("yolov8n.pt")
    else:
        model = YOLO(MODEL_PATH)
        
    CLASS_NAMES = model.names
    vehicle_counts = {name: 0 for name in CLASS_NAMES.values()}
    id_da_dem = set()
    analytics = TrafficAnalytics()

    # --- [MỚI] LOAD CONFIG TỪ JSON ---
    # Lưu ý: VIDEO_SOURCE hiện tại là URL RTSP. File JSON cần có key tương ứng
    # hoặc bạn sửa code bên dưới để dùng một key cố định (ví dụ "TestVideo2.mp4") nếu muốn dùng lại config cũ.
    print(f"[INFO] Đang tải cấu hình cho: {VIDEO_SOURCE}")
    speed_zones_data, occupancy_zones_data = load_config_from_json(JSON_CONFIG_FILE, "TestVideo1.mp4")
    
    if speed_zones_data is None:
        print("[ERROR] Không thể khởi chạy do thiếu cấu hình trong JSON.")
        print(f" -> Hãy kiểm tra xem file {JSON_CONFIG_FILE} có key là '{VIDEO_SOURCE}' hay chưa.")
        return

    # KHỞI TẠO TỐC ĐỘ TỪ DATA JSON
    speed_estimator = SpeedEstimator()
    for i, (poly, w, h) in enumerate(speed_zones_data):
        speed_estimator.add_zone(f"SpeedZone{i+1}", poly, w, h)

    # --- TÍNH TỔNG DIỆN TÍCH THỰC (M2) ---
    total_roi_real_area = sum([z['real_w'] * z['real_h'] for z in occupancy_zones_data])
    if total_roi_real_area == 0: total_roi_real_area = 1 
    print(f"[INFO] Total Analysis Area: {total_roi_real_area} m2")

    # --- INPUT VIDEO (RTSP) ---
    print(f"[INFO] Connecting to stream: {VIDEO_SOURCE}")
    cap = VideoCaptureThreading(VIDEO_SOURCE)
    
    print("[INFO] Waiting for stream buffer...")
    time.sleep(2.0)

    # --- RETRY LOGIC FOR FIRST FRAME ---
    sample = None
    for i in range(10):
        ret, sample = cap.read()
        if sample is not None:
            break
        print(f"[WARNING] Waiting for stream data... ({i+1}/10)")
        time.sleep(1.0)
        
    if sample is None:
        print("[ERROR] No video feed received.")
        cap.release()
        return

    # Lấy kích thước để hiển thị
    frame_height, frame_width = sample.shape[:2]
    scale_ratio = TARGET_HEIGHT / frame_height
    resized_width = int(frame_width * scale_ratio)
    
    print(f"[INFO] System Started. Display size: {resized_width + DASHBOARD_WIDTH}x{TARGET_HEIGHT}")
    print("[INFO] Press 'q' to exit.")

    fps = 0; prev_frame_time = 0

    try:
        while True:
            ret, frame = cap.read()
            
            # Xử lý khi buffer rỗng hoặc mất frame
            if not ret or frame is None:
                # Nếu là file nội bộ thì thoát
                if isinstance(VIDEO_SOURCE, str) and os.path.exists(VIDEO_SOURCE) and os.path.isfile(VIDEO_SOURCE):
                    print("[INFO] End of file.")
                    break
                
                # RTSP thì đợi
                print("[WARNING] Frame buffer empty, waiting for stream...")
                time.sleep(0.1)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
                continue
            
            # [QUAN TRỌNG] RTSP dùng time.time() để tính tốc độ chuẩn hơn là timestamp frame
            video_timestamp_s = time.time()

            overlay = frame.copy()

            tracker_arg = TRACKER_CONFIG if os.path.exists(TRACKER_CONFIG) else "bytetrack.yaml"
            results = model.track(frame, persist=True, tracker=tracker_arg, verbose=False, conf=0.25)
            
            vehicles_in_roi_count = 0
            current_frame_speeds = []
            current_occupied_real_area = 0.0

            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                ids = results[0].boxes.id.int().cpu().tolist()
                clss = results[0].boxes.cls.int().cpu().tolist()

                for box, track_id, cls in zip(boxes, ids, clss):
                    x1, y1, x2, y2 = box
                    cls_name = CLASS_NAMES.get(cls, str(cls))
                    cx, cy = int((x1+x2)/2), int(y2)
                    
                    # 1. Tốc độ
                    speed = speed_estimator.estimate_speed(track_id, (x1, y1, x2, y2), video_timestamp_s)
                    if speed > 0: current_frame_speeds.append(speed)

                    # 2. Kiểm tra xe nằm trong Zone nào để tính Occupancy (Dùng data từ JSON)
                    is_in_any_roi = False
                    for zone_cfg in occupancy_zones_data:
                        if cv2.pointPolygonTest(zone_cfg['poly'], (cx, cy), False) > 0:
                            is_in_any_roi = True
                            v_area = VEHICLE_REAL_AREAS.get(cls_name.lower(), DEFAULT_VEHICLE_AREA)
                            current_occupied_real_area += v_area
                            break 

                    if is_in_any_roi:
                        vehicles_in_roi_count += 1
                        if track_id not in id_da_dem:
                            id_da_dem.add(track_id)
                            if cls_name in vehicle_counts: vehicle_counts[cls_name] += 1
                    
                    color = CLASS_COLOR_MAP.get(cls_name, (255,255,255))
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    cv2.putText(frame, f"{cls_name}", (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    if speed > 5:
                        cv2.putText(frame, f"{speed}km/h", (int(x1), int(y2)+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

            # Vẽ Speed Zones
            if SHOW_SPEED_ZONES:
                speed_estimator.draw_zones(frame)
            
            # Vẽ Occupancy Zones (Dùng data từ JSON)
            if SHOW_OCCUPANCY_ZONES:
                occupancy_polys = [z['poly'] for z in occupancy_zones_data]
                cv2.polylines(frame, occupancy_polys, True, ROI_COLOR, 2)
                cv2.fillPoly(overlay, occupancy_polys, ROI_COLOR)
                cv2.addWeighted(overlay, ROI_ALPHA, frame, 1 - ROI_ALPHA, 0, frame)

            # --- TÍNH TOÁN CHỈ SỐ TRAFFIC ---
            # 1. Tính Occupancy Rate tức thời
            occupancy_rate_instant = (current_occupied_real_area / total_roi_real_area) * 100
            if occupancy_rate_instant > 100: occupancy_rate_instant = 100
            
            # [UPDATE] Truyền occupancy vào hàm update_stats
            analytics.update_stats(len(id_da_dem), current_frame_speeds, occupancy_rate_instant)
            
            # [UPDATE] Lấy giá trị đã làm mượt (Smoothed)
            avg_spd = analytics.get_avg_speed()
            avg_occ = analytics.get_avg_occupancy() # Dùng cái này thay vì occupancy_rate tức thời
            
            density_display_str = f"{avg_occ:.1f}%"

            # --- 2. Logic Trạng thái (REFINED v3) ---
            # Định nghĩa lại ngưỡng cho rõ ràng
            SPD_STOPPED = 5.0      # Xe gần như đứng yên
            SPD_CRAWL   = 20.0     # Nhích từng chút (kẹt nặng)
            SPD_SLOW    = 25.0     # Di chuyển chậm
            
            OCC_EMPTY   = 1.0      # Đường trống
            OCC_LIGHT   = 15.0     # Mật độ thấp
            OCC_HEAVY   = 45.0     # Mật độ cao (Ùn ứ)

            status_text = "FREE FLOW"
            status_color = C_GREEN

            # CASE 1: Đường vắng tanh (ưu tiên check số lượng thực tế detect được để tránh false alarm)
            if vehicles_in_roi_count == 0 and avg_occ < OCC_EMPTY:
                status_text, status_color = "EMPTY ROAD", (200, 200, 200)

            # CASE 2: Xe đứng yên (Có xe nhưng tốc độ ~ 0) -> TẮC CỨNG hoặc ĐÈN ĐỎ
            elif avg_spd < SPD_STOPPED and avg_occ > OCC_LIGHT:
                status_text, status_color = "STOPPED / RED LIGHT", C_RED
            
            # CASE 3: Mật độ cao (Ùn tắc nghiêm trọng)
            elif avg_occ > OCC_HEAVY:
                if avg_spd < SPD_CRAWL:
                    status_text, status_color = "TRAFFIC JAM", (0, 0, 255) # Đỏ đậm
                else:
                    # Mật độ cao nhưng vẫn đi được nhanh -> Lưu lượng lớn (High Throughput)
                    status_text, status_color = "HIGH DENSITY", (0, 140, 255) # Cam

            # CASE 4: Mật độ trung bình/thấp nhưng tốc độ chậm (Có chướng ngại vật hoặc dò đường)
            elif avg_spd < SPD_SLOW:
                status_text, status_color = "SLOW TRAFFIC", C_YELLOW
            
            # CASE 5: Còn lại
            elif avg_occ > OCC_LIGHT:
                 status_text, status_color = "MODERATE", (0, 215, 255) # Xanh lơ
            else:
                 status_text, status_color = "FREE FLOW", C_GREEN
            
            # --- FPS ---
            curr_time = time.time()
            if prev_frame_time > 0:
                fps = 0.9 * fps + 0.1 * (1/(curr_time - prev_frame_time))
            prev_frame_time = curr_time

            # --- RENDER ---
            final_frame_view = cv2.resize(frame, (resized_width, TARGET_HEIGHT))
            dash_img = analytics.draw_dashboard(vehicle_counts, density_display_str, status_text, status_color, int(fps))
            combined_img = np.hstack((final_frame_view, dash_img))
            
            cv2.imshow("Traffic AI System", combined_img)
            
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    except KeyboardInterrupt: pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\n[DONE] System Terminated.")

if __name__ == "__main__":
    main()