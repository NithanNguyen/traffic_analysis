'''
# --- Run MediaMTX and FFMPEG
cd tools
.\mediamtx.exe

cd tools
.\ffmpeg -re -stream_loop -1 -i TestVideo1.mp4 -c:v copy -rtsp_transport tcp -f rtsp rtsp://localhost:8554/live_stream

'''

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
# Sử dụng backend 'Agg' cho Matplotlib để không hiển thị cửa sổ popup (GUI),
# chỉ dùng để render ra ảnh (buffer) giúp tăng tốc độ xử lý server/background.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from collections import deque
from pathlib import Path
import json  

# ==========================================
# 1. CẤU HÌNH HỆ THỐNG & HẰNG SỐ
# ==========================================

# Đường dẫn model và video đầu vào
MODEL_PATH = "traffic4.pt"    
# [LƯU Ý] URL RTSP giả lập (hoặc camera thật). Cần đảm bảo server RTSP đang chạy.
VIDEO_SOURCE = "rtsp://localhost:8554/live_stream" 
JSON_CONFIG_FILE = "traffic_config.json" 
TRACKER_CONFIG = "my_tracker.yaml" # File cấu hình bộ theo dõi (ByteTrack/BotSort)

# Cờ hiển thị (Bật/Tắt các lớp vẽ lên video)
SHOW_OCCUPANCY_ZONES = True  # Hiển thị vùng tính mật độ
SHOW_SPEED_ZONES = True      # Hiển thị vùng đo tốc độ

# Kích thước chuẩn hóa để hiển thị lên màn hình (không ảnh hưởng logic xử lý ảnh gốc)
TARGET_HEIGHT = 720
DASHBOARD_WIDTH = 450

# --- BẢNG MÀU (COLORS) & GIAO DIỆN ---
C_BG = (30, 30, 30)          # Màu nền Dashboard (Xám đậm)
C_CARD = (50, 50, 50)        # Màu nền các thẻ thông tin
C_TEXT_MAIN = (230, 230, 230)
C_TEXT_SUB = (180, 180, 180)
C_ACCENT = (0, 191, 255)     # Màu điểm nhấn (Xanh dương)
C_GREEN = (100, 200, 100)
C_YELLOW = (0, 215, 255)    
C_RED = (80, 80, 255)       

# Map màu sắc cho từng loại phương tiện
CLASS_COLOR_MAP = {
    'car': (0, 255, 127),     
    'bus': (0, 140, 255),    
    'truck': (203, 192, 255),
    'motor': (255, 255, 0)    
}

# --- CẤU HÌNH DIỆN TÍCH XE (M2) ---
# Dùng để tính toán mật độ chiếm dụng đường (Occupancy)
# Ví dụ: 1 xe bus chiếm 35m2, 1 xe máy chiếm 2m2
VEHICLE_REAL_AREAS = {
    'car': 8.0,      
    'bus': 35.0,     
    'truck': 30.0,   
    'motor': 2.0     
}
DEFAULT_VEHICLE_AREA = 8.0 # Giá trị mặc định nếu class không khớp
ROI_COLOR = (0, 165, 255)  # Màu vùng ROI (Cam)
ROI_ALPHA = 0.2            # Độ trong suốt của vùng ROI

# ==========================================
# 2. XỬ LÝ VIDEO ĐA LUỒNG (THREADING)
# ==========================================
class VideoCaptureThreading:
    """
    Class này chạy việc đọc frame từ RTSP trên một luồng (thread) riêng biệt.
    Lý do: Hàm cv2.read() là hàm chặn (blocking). Nếu mạng lag, nó sẽ làm treo
    toàn bộ chương trình (bao gồm cả việc AI detect).
    Dùng Thread giúp AI luôn có frame mới nhất để xử lý mà không bị đợi.
    """
    def __init__(self, src):
        self.src = src
        # [QUAN TRỌNG] Bắt buộc dùng TCP cho RTSP thay vì UDP mặc định.
        # TCP đảm bảo gói tin đến đủ, tránh vỡ hình (artifacts) dù độ trễ cao hơn chút xíu.
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        self.cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        
        if not self.cap.isOpened():
            print(f"[ERROR] Failed to open connection to: {src}")
            return
        
        # Hàng đợi (Queue) chỉ giữ tối đa 10 frame mới nhất để tránh tràn RAM
        self.q = queue.Queue(maxsize=10)
        self.stop_event = threading.Event()
        self.t = threading.Thread(target=self._reader)
        self.t.daemon = True # Thread sẽ tự tắt khi chương trình chính tắt
        self.t.start()

    def _reader(self):
        """Hàm chạy ngầm liên tục đọc frame từ Camera"""
        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01) # Nghỉ nhẹ để giảm tải CPU nếu mất tín hiệu
                continue
            
            # Nếu hàng đợi đầy, vứt bỏ frame cũ đi để lấy chỗ cho frame mới (Non-blocking)
            if not self.q.empty():
                try: self.q.get_nowait()
                except queue.Empty: pass
            self.q.put(frame)

    def read(self):
        """Hàm lấy frame ra để xử lý"""
        try: 
            # Chờ tối đa 1s để lấy frame, nếu không có thì báo lỗi
            return True, self.q.get(timeout=1)
        except queue.Empty:
            return False, None

    def release(self):
        self.stop_event.set()
        self.t.join()
        self.cap.release()

# ==========================================
# 3. HÀM TIỆN ÍCH (UTILS)
# ==========================================
def load_config_from_json(json_path, video_name):
    """
    Đọc file JSON cấu hình toạ độ các vùng (Speed/Occupancy).
    Input: Đường dẫn file JSON, Tên Key video (ví dụ 'TestVideo1.mp4').
    Output: Danh sách các vùng Speed và Occupancy đã chuẩn hóa sang Numpy Array.
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

# ==========================================
# 4. LOGIC TÍNH TỐC ĐỘ (SPEED ESTIMATION)
# ==========================================
class SpeedEstimator:
    def __init__(self):
        self.track_history = {} # Lưu lịch sử vị trí của từng track_id
        self.zones = [] 

    def add_zone(self, zone_id, src_points, real_width, real_length):
        """
        Tạo ma trận biến đổi phối cảnh (Perspective Transform Matrix).
        Biến đổi từ hình thang trên camera (src_points) -> hình chữ nhật thực tế (dst_points).
        """
        dst_points = np.array([[0, 0], [real_width, 0], [real_width, real_length], [0, real_length]], dtype=np.float32)
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        self.zones.append({'id': zone_id, 'poly': src_points.astype(np.int32), 'matrix': matrix})

    def transform_point(self, point, matrix):
        """Chuyển đổi điểm pixel (x,y) sang điểm thực tế (mét)"""
        p = np.array([[[point[0], point[1]]]], dtype=np.float32)
        dst = cv2.perspectiveTransform(p, matrix)
        return dst[0][0]

    def estimate_speed(self, track_id, box_coords, video_timestamp):
        """
        Hàm cốt lõi tính tốc độ:
        1. Xác định xe đang ở vùng (Zone) nào.
        2. Chuyển đổi vị trí pixel chân xe sang vị trí thực (mét).
        3. Tính quãng đường (Distance) và thời gian (Time Diff) so với frame trước.
        4. V = S/t * 3.6 (m/s -> km/h).
        5. Dùng bộ lọc mũ (Exponential Moving Average) để làm mượt tốc độ, tránh nhảy số.
        """
        x1, y1, x2, y2 = box_coords
        center_bottom = (int((x1 + x2) / 2), int(y2)) # Điểm chạm đất của xe
        
        # Tìm vùng xe đang đi vào
        active_zone = None
        for zone in self.zones:
            # pointPolygonTest >= 0 nghĩa là điểm nằm trong hoặc trên cạnh đa giác
            if cv2.pointPolygonTest(zone['poly'], center_bottom, False) >= 0:
                active_zone = zone
                break
        
        # Nếu xe ra khỏi vùng đo tốc độ, xóa lịch sử để tiết kiệm bộ nhớ
        if active_zone is None: 
            if track_id in self.track_history:
                del self.track_history[track_id]
            return 0

        current_time = video_timestamp 
        current_real_pos = self.transform_point(center_bottom, active_zone['matrix'])
        speed_kmh = 0
        
        if track_id in self.track_history:
            prev_data = self.track_history[track_id]
            # Chỉ tính nếu xe vẫn ở cùng một vùng (zone_id khớp)
            if prev_data.get('zone_id') == active_zone['id']:
                # Tính khoảng cách Euclidean thực tế (mét)
                dist = math.sqrt((current_real_pos[0] - prev_data['pos'][0])**2 + (current_real_pos[1] - prev_data['pos'][1])**2)
                time_diff = current_time - prev_data['time']
                tracked_frames = prev_data.get('frames', 0) + 1

                # Chỉ tính khi thời gian đủ lớn (>0.02s) để tránh lỗi chia số quá nhỏ
                if time_diff > 0.02: 
                    temp_speed = (dist / time_diff) * 3.6
                    
                    # Cần ít nhất 5 frames theo dõi để tốc độ ổn định
                    if tracked_frames > 5:
                        last_speed = prev_data.get('speed', 0)
                        # Lọc nhiễu: Nếu tốc độ tăng đột ngột > 100km/h (do nhảy bounding box), giữ nguyên tốc độ cũ
                        if temp_speed > 100: speed_kmh = last_speed
                        else:
                            if last_speed == 0: speed_kmh = temp_speed
                            # EMA Filter: 90% giá trị cũ + 10% giá trị mới -> Làm mượt số hiển thị
                            else: speed_kmh = 0.9 * last_speed + 0.1 * temp_speed
                    else: speed_kmh = 0
            else: tracked_frames = 1 # Reset nếu đổi vùng
        else: tracked_frames = 1

        # Cập nhật lại lịch sử
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

# ==========================================
# 5. PHÂN TÍCH & THỐNG KÊ (ANALYTICS)
# ==========================================
FLOW_WINDOW_SECONDS = 3     # Chu kỳ cập nhật biểu đồ (3 giây/lần)
MAX_HISTORY_POINTS = 30     # Số điểm tối đa trên biểu đồ

class TrafficAnalytics:
    def __init__(self):
        self.start_time = time.time()
        self.last_update_time = time.time()
        
        # Deque: Hàng đợi 2 đầu, tự động đẩy dữ liệu cũ ra khi đầy
        self.occupancy_history = deque(maxlen=MAX_HISTORY_POINTS) 
        self.timestamps = deque(maxlen=MAX_HISTORY_POINTS)
        
        self.speed_buffer = deque(maxlen=50)   # Lưu 50 giá trị tốc độ gần nhất để tính trung bình
        self.occupancy_buffer = deque(maxlen=30) 
        
        # Cấu hình Matplotlib (Vẽ biểu đồ)
        plt.style.use('dark_background')
        self.fig, self.ax = plt.subplots(figsize=(6, 3.5), dpi=80) 
        self.fig.patch.set_facecolor('#2D2D2D') # Màu nền bao quanh
        self.chart_image = None
        
        # Khởi tạo giá trị 0 ban đầu cho biểu đồ chạy đẹp
        for i in range(MAX_HISTORY_POINTS): 
            self.timestamps.append(0)
            self.occupancy_history.append(0.0)

    def update_stats(self, new_count, speed_list, current_occupancy_rate):
        """Hàm nhận dữ liệu thô từ vòng lặp chính và lưu vào buffer"""
        for s in speed_list:
            if s > 5: self.speed_buffer.append(s) # Chỉ tính xe đang chạy > 5km/h vào trung bình
        self.occupancy_buffer.append(current_occupancy_rate)

        now = time.time()
        # Chỉ vẽ lại biểu đồ mỗi FLOW_WINDOW_SECONDS giây để giảm tải CPU
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
        """Vẽ biểu đồ bằng Matplotlib và chuyển thành ảnh OpenCV"""
        self.ax.clear()
        
        x = list(self.timestamps)
        y = list(self.occupancy_history)
        
        self.ax.set_facecolor('#2D2D2D')
        self.ax.fill_between(x, y, color='#00ADB5', alpha=0.2)
        self.ax.plot(x, y, color='#00ADB5', linewidth=2, label='Occupancy')
        
        self.ax.set_ylim(0, 100) # Trục Y luôn cố định 0-100%
        
        # Trang trí biểu đồ
        self.ax.set_title('Occupancy Trend (%)', color='#EEEEEE', fontsize=10, fontweight='bold', pad=10)
        self.ax.set_ylabel('Density (%)', color='#C0C0C0', fontsize=8)
        self.ax.set_xlabel('Time (s)', color='#C0C0C0', fontsize=8)
        self.ax.tick_params(axis='x', colors='#C0C0C0', labelsize=7)
        self.ax.tick_params(axis='y', colors='#C0C0C0', labelsize=7)
        
        for spine in self.ax.spines.values():
            spine.set_edgecolor('#555555')
        self.ax.grid(True, color='#444444', linestyle=':', linewidth=0.5, alpha=0.7)
        
        self.fig.tight_layout()
        
        # Chuyển đổi: Matplotlib Figure -> Buffer -> Numpy Array (Image)
        canvas = FigureCanvas(self.fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        self.chart_image = cv2.cvtColor(np.asarray(buf), cv2.COLOR_RGBA2BGR)

    def draw_dashboard(self, vehicle_counts, density_str, status_text, status_color, fps):
        """Vẽ giao diện dashboard bên phải màn hình"""
        H, W = TARGET_HEIGHT, DASHBOARD_WIDTH
        dash = np.zeros((H, W, 3), dtype=np.uint8)
        dash[:] = C_BG # Tô màu nền

        # --- HEADER ---
        cv2.rectangle(dash, (0, 0), (W, 60), (40, 40, 40), -1)
        cv2.putText(dash, "TRAFFIC ANALYTICS", (20, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, C_ACCENT, 1)
        cv2.putText(dash, f"FPS: {fps}", (W - 100, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 1)

        y_off = 80
        # --- THẺ TRẠNG THÁI (Free Flow / Jam) ---
        self._draw_card(dash, 20, y_off, W-40, 100, C_CARD)
        cv2.circle(dash, (70, y_off + 50), 25, status_color, -1)
        font_scale_status = 0.9 if len(status_text) < 10 else 0.7
        cv2.putText(dash, status_text, (120, y_off + 60), cv2.FONT_HERSHEY_DUPLEX, font_scale_status, (255, 255, 255), 1)
        y_off += 120

        # --- THẺ SỐ LIỆU (Occupancy & Speed) ---
        card_w = (W - 50) // 2
        # Card 1: Occupancy
        self._draw_card(dash, 20, y_off, card_w, 80, C_CARD)
        cv2.putText(dash, "OCCUPANCY", (35, y_off + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_TEXT_SUB, 1)
        cv2.putText(dash, density_str, (30, y_off + 60), cv2.FONT_HERSHEY_DUPLEX, 1.0, C_TEXT_MAIN, 1)

        # Card 2: Speed Avg
        avg_spd = self.get_avg_speed()
        self._draw_card(dash, 20 + card_w + 10, y_off, card_w, 80, C_CARD)
        cv2.putText(dash, "AVG KM/H", (20 + card_w + 25, y_off + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_TEXT_SUB, 1)
        spd_color = C_GREEN if avg_spd > 40 else (C_YELLOW if avg_spd > 20 else C_RED) # Đổi màu theo mức độ
        cv2.putText(dash, str(avg_spd), (20 + card_w + 25, y_off + 60), cv2.FONT_HERSHEY_DUPLEX, 1.2, spd_color, 1)
        y_off += 90

        # --- THẺ ĐẾM XE (Counts) ---
        self._draw_card(dash, 20, y_off, W-40, 120, C_CARD)
        row_y = y_off + 30
        col_x = 35
        for i, (k, v) in enumerate(vehicle_counts.items()):
            color = CLASS_COLOR_MAP.get(k, (255,255,255))
            cv2.circle(dash, (col_x, row_y), 5, color, -1)
            cv2.putText(dash, f"{k.upper()}: {v}", (col_x + 15, row_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_TEXT_MAIN, 1)
            # Logic xếp layout 2 cột
            if i % 2 == 0: col_x += 180
            else: col_x = 35; row_y += 30
        y_off += 140

        # --- CHART ---
        if self.chart_image is not None:
            target_w = W - 40
            h_c, w_c = self.chart_image.shape[:2]
            target_h = int(h_c * (target_w / w_c)) # Resize giữ tỷ lệ khung hình
            
            if y_off + target_h < H:
                chart_resized = cv2.resize(self.chart_image, (target_w, target_h))
                dash[y_off:y_off+target_h, 20:20+target_w] = chart_resized

        return dash

    def _draw_card(self, img, x, y, w, h, color):
        cv2.rectangle(img, (x, y), (x + w, y + h), color, -1)

# ==========================================
# 6. CHƯƠNG TRÌNH CHÍNH (MAIN)
# ==========================================
def main():
    # Load Model (Nếu có GPU RTX 3050, Ultralytics sẽ tự động dùng CUDA)
    if not os.path.exists(MODEL_PATH) and not os.path.exists("yolov8n.pt"):
        print("Đang tải model mặc định..."); model = YOLO("yolov8n.pt")
    else:
        model = YOLO(MODEL_PATH)
        
    CLASS_NAMES = model.names
    vehicle_counts = {name: 0 for name in CLASS_NAMES.values()}
    id_da_dem = set() # Set dùng để lưu ID xe đã đếm để tránh đếm trùng
    analytics = TrafficAnalytics()

    # --- LOAD CONFIG TỪ JSON ---
    # [QUAN TRỌNG] Ở đây bạn đang fix cứng tên video là "TestVideo1.mp4"
    # Nếu chạy RTSP thật, cần đổi key trong JSON thành URL RTSP hoặc sửa dòng dưới thành key tương ứng.
    print(f"[INFO] Đang tải cấu hình cho: {VIDEO_SOURCE}")
    speed_zones_data, occupancy_zones_data = load_config_from_json(JSON_CONFIG_FILE, "TestVideo3.mp4")
    
    if speed_zones_data is None:
        print("[ERROR] Không thể khởi chạy do thiếu cấu hình trong JSON.")
        return

    # Khởi tạo SpeedEstimator với dữ liệu từ JSON
    speed_estimator = SpeedEstimator()
    for i, (poly, w, h) in enumerate(speed_zones_data):
        speed_estimator.add_zone(f"SpeedZone{i+1}", poly, w, h)

    # Tính tổng diện tích vùng ROI (để làm mẫu số tính %)
    total_roi_real_area = sum([z['real_w'] * z['real_h'] for z in occupancy_zones_data])
    if total_roi_real_area == 0: total_roi_real_area = 1 
    print(f"[INFO] Total Analysis Area: {total_roi_real_area} m2")

    # --- KẾT NỐI RTSP ---
    print(f"[INFO] Connecting to stream: {VIDEO_SOURCE}")
    cap = VideoCaptureThreading(VIDEO_SOURCE)
    
    print("[INFO] Waiting for stream buffer...")
    time.sleep(2.0) # Chờ luồng VideoCaptureThreading nạp đầy buffer

    # Thử đọc frame đầu tiên để kiểm tra kết nối
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

    # Tính tỷ lệ scale để hiển thị vừa màn hình (TARGET_HEIGHT = 720)
    frame_height, frame_width = sample.shape[:2]
    scale_ratio = TARGET_HEIGHT / frame_height
    resized_width = int(frame_width * scale_ratio)
    
    print(f"[INFO] System Started. Display size: {resized_width + DASHBOARD_WIDTH}x{TARGET_HEIGHT}")
    print("[INFO] Press 'q' to exit.")

    fps = 0; prev_frame_time = 0

    try:
        # --- VÒNG LẶP XỬ LÝ CHÍNH ---
        while True:
            ret, frame = cap.read()
            
            # Xử lý khi buffer rỗng hoặc mất kết nối
            if not ret or frame is None:
                if isinstance(VIDEO_SOURCE, str) and os.path.exists(VIDEO_SOURCE) and os.path.isfile(VIDEO_SOURCE):
                    print("[INFO] End of file.")
                    break
                
                print("[WARNING] Frame buffer empty, waiting for stream...")
                time.sleep(0.1)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
                continue
            
            # Lấy timestamp hiện tại (quan trọng cho RTSP để tính vận tốc chính xác)
            video_timestamp_s = time.time()

            overlay = frame.copy() # Tạo lớp phủ để vẽ trong suốt

            # --- YOLO TRACKING ---
            # persist=True: Giữ ID đối tượng qua các frame liên tiếp (quan trọng để đếm và tính tốc độ)
            # conf=0.25: Chỉ nhận các xe có độ tin cậy > 25%
            tracker_arg = TRACKER_CONFIG if os.path.exists(TRACKER_CONFIG) else "bytetrack.yaml"
            results = model.track(frame, persist=True, tracker=tracker_arg, verbose=False, conf=0.25)
            
            vehicles_in_roi_count = 0
            current_frame_speeds = []
            current_occupied_real_area = 0.0

            # --- XỬ LÝ KẾT QUẢ DETECTION ---
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                ids = results[0].boxes.id.int().cpu().tolist()
                clss = results[0].boxes.cls.int().cpu().tolist()

                for box, track_id, cls in zip(boxes, ids, clss):
                    x1, y1, x2, y2 = box
                    cls_name = CLASS_NAMES.get(cls, str(cls))
                    cx, cy = int((x1+x2)/2), int(y2)
                    
                    # 1. Tính toán Tốc độ
                    speed = speed_estimator.estimate_speed(track_id, (x1, y1, x2, y2), video_timestamp_s)
                    if speed > 0: current_frame_speeds.append(speed)

                    # 2. Kiểm tra xe nằm trong Zone nào để tính Occupancy
                    is_in_any_roi = False
                    for zone_cfg in occupancy_zones_data:
                        # Kiểm tra điểm (cx, cy) có nằm trong đa giác không
                        if cv2.pointPolygonTest(zone_cfg['poly'], (cx, cy), False) > 0:
                            is_in_any_roi = True
                            # Cộng dồn diện tích xe vào tổng diện tích chiếm dụng frame này
                            v_area = VEHICLE_REAL_AREAS.get(cls_name.lower(), DEFAULT_VEHICLE_AREA)
                            current_occupied_real_area += v_area
                            break 

                    if is_in_any_roi:
                        vehicles_in_roi_count += 1
                        # Logic Đếm xe: Chỉ đếm nếu ID chưa từng xuất hiện trong Set
                        if track_id not in id_da_dem:
                            id_da_dem.add(track_id)
                            if cls_name in vehicle_counts: vehicle_counts[cls_name] += 1
                    
                    # Vẽ Bounding Box và thông tin
                    color = CLASS_COLOR_MAP.get(cls_name, (255,255,255))
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    cv2.putText(frame, f"{cls_name}", (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    if speed > 5:
                        cv2.putText(frame, f"{speed}km/h", (int(x1), int(y2)+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

            # --- VẼ CÁC VÙNG CẤU HÌNH ---
            if SHOW_SPEED_ZONES:
                speed_estimator.draw_zones(frame)
            
            if SHOW_OCCUPANCY_ZONES:
                occupancy_polys = [z['poly'] for z in occupancy_zones_data]
                cv2.polylines(frame, occupancy_polys, True, ROI_COLOR, 2)
                cv2.fillPoly(overlay, occupancy_polys, ROI_COLOR)
                # Trộn ảnh gốc và overlay để tạo hiệu ứng trong suốt
                cv2.addWeighted(overlay, ROI_ALPHA, frame, 1 - ROI_ALPHA, 0, frame)

            # --- TÍNH TOÁN LOGIC TRẠNG THÁI GIAO THÔNG ---
            # 1. Tính % Occupancy tức thời
            occupancy_rate_instant = (current_occupied_real_area / total_roi_real_area) * 100
            if occupancy_rate_instant > 100: occupancy_rate_instant = 100
            
            # Cập nhật số liệu vào bộ phân tích
            analytics.update_stats(len(id_da_dem), current_frame_speeds, occupancy_rate_instant)
            
            # Lấy giá trị trung bình (đã làm mượt) để hiển thị và xét trạng thái
            avg_spd = analytics.get_avg_speed()
            avg_occ = analytics.get_avg_occupancy()
            
            density_display_str = f"{avg_occ:.1f}%"

            # 2. Logic đánh giá trạng thái (Rule-based)
            # Các ngưỡng (Thresholds)
            SPD_STOPPED = 5.0      # Xe gần như đứng yên
            SPD_CRAWL   = 20.0     # Nhích từng chút
            SPD_SLOW    = 25.0     # Di chuyển chậm
            
            OCC_EMPTY   = 1.0      # Đường trống
            OCC_LIGHT   = 15.0     # Mật độ thấp
            OCC_HEAVY   = 45.0     # Mật độ cao

            status_text = "FREE FLOW"
            status_color = C_GREEN

            # Quyết định trạng thái dựa trên sự kết hợp giữa Tốc độ và Mật độ
            if vehicles_in_roi_count == 0 and avg_occ < OCC_EMPTY:
                status_text, status_color = "EMPTY ROAD", (200, 200, 200)
                # Đường không có xe, rất thông thoáng

            elif avg_spd < SPD_STOPPED and avg_occ > OCC_LIGHT:
                status_text, status_color = "STOPPED / RED LIGHT", C_RED 
                # Tắc cứng hoặc đèn đỏ
            
            elif avg_occ > OCC_HEAVY:
                if avg_spd < SPD_CRAWL:
                    status_text, status_color = "TRAFFIC JAM", (0, 0, 255) 
                    # Mật độ xe dày đặc và các xe di chuyển rất chậm chạp. Đây là định nghĩa điển hình của kẹt xe
                else:
                    status_text, status_color = "HIGH DENSITY", (0, 140, 255) 
                    # Mặc dù đường rất đông xe (mật độ cao), nhưng các xe vẫn di chuyển được với tốc độ ổn định

            elif avg_spd < SPD_SLOW:
                status_text, status_color = "SLOW TRAFFIC", C_YELLOW
                # Đường không quá đông nhưng các xe đi chậm lại, có thể do tín hiệu đèn hoặc các yếu tố tạm thời khác
            
            elif avg_occ > OCC_LIGHT:
                 status_text, status_color = "MODERATE", (0, 215, 255)
                 # Giao thông ổn định, lượng xe vừa phải, di chuyển bình thường
            else:
                 status_text, status_color = "FREE FLOW", C_GREEN
                 # Đường thông thoáng, xe cộ di chuyển nhanh và không bị cản trở
            
            # --- TÍNH FPS HIỂN THỊ ---
            curr_time = time.time()
            if prev_frame_time > 0:
                fps = 0.9 * fps + 0.1 * (1/(curr_time - prev_frame_time))
            prev_frame_time = curr_time

            # --- RENDER RA MÀN HÌNH ---
            # Resize frame video cho khớp layout
            final_frame_view = cv2.resize(frame, (resized_width, TARGET_HEIGHT))
            # Lấy ảnh Dashboard từ class Analytics
            dash_img = analytics.draw_dashboard(vehicle_counts, density_display_str, status_text, status_color, int(fps))
            # Ghép video và dashboard theo chiều ngang
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