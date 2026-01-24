"""
Integrated ALPR System with Web Dashboard
Runs both the ALPR pipeline and web dashboard together, with real-time data streaming.
"""

import os
import sys
import threading
import time
import cv2
import numpy as np
import string
import easyocr
from collections import deque, defaultdict
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ultralytics import YOLO
from sort.sort import Sort

# Flask imports
from flask import Flask, render_template, Response, jsonify, request, send_from_directory
from flask_socketio import SocketIO, emit

# Try to import config
try:
    import config
    OCR_GPU = config.OCR_GPU
except ImportError:
    OCR_GPU = False


# =============================================================================
# FLASK APP INITIALIZATION
# =============================================================================

app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['SECRET_KEY'] = 'alpr_secret_key_2024'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class VehicleRecord:
    vehicle_id: int
    vehicle_type: str
    first_seen: str
    last_seen: str
    license_plate: str
    confidence: float
    speed_estimate: float
    lane: int
    violations: List[str]
    frame_count: int

@dataclass
class ViolationRecord:
    id: int
    timestamp: str
    vehicle_id: int
    license_plate: str
    violation_type: str
    severity: str
    description: str
    speed: Optional[float]
    location: str
    snapshot_frame: int
    challan_amount: int  # Challan amount in ₹

@dataclass
class PlateDetection:
    id: int
    timestamp: str
    plate_text: str
    confidence: float
    vehicle_id: int
    vehicle_type: str
    frame_number: int


# =============================================================================
# GLOBAL DATA STORE
# =============================================================================

class ALPRDataStore:
    def __init__(self):
        self.vehicles: Dict[int, VehicleRecord] = {}
        self.violations: List[ViolationRecord] = []
        self.plate_detections: List[PlateDetection] = []
        self.heatmap_data: np.ndarray = None
        self.frame_width = 1920
        self.frame_height = 1080
        self.stats = {
            'total_vehicles': 0,
            'total_plates': 0,
            'total_violations': 0,
            'violation_types': defaultdict(int),
            'vehicle_types': defaultdict(int),
            'total_challan': 0,  # Total challan amount collected
            'challan_count': 0,  # Number of challans issued
        }
        
        # Challan amounts (in ₹) based on violation type
        self.challan_rates = {
            'SPEEDING': 2000,
            'RED_LIGHT': 5000,
            'LANE_VIOLATION': 1000,
            'STOP_SIGN': 1000,
            'INVALID_PLATE': 2000,
            'NO_HELMET': 1000,
            'WRONG_WAY': 2000,
            'PARKING_VIOLATION': 500,
        }
        self.logs: List[dict] = []
        self.is_processing = False
        self.current_frame = 0
        self.total_frames = 0
        self.lock = threading.Lock()
        self.current_display_frame = None
        self.frame_lock = threading.Lock()
        
        self.reset_heatmap(1920, 1080)
    
    def reset_heatmap(self, width, height):
        self.frame_width = width
        self.frame_height = height
        self.heatmap_data = np.zeros((height // 10, width // 10), dtype=np.float32)
    
    def update_heatmap(self, x1, y1, x2, y2):
        cx = int((x1 + x2) / 2 / 10)
        cy = int((y1 + y2) / 2 / 10)
        if 0 <= cx < self.heatmap_data.shape[1] and 0 <= cy < self.heatmap_data.shape[0]:
            for dy in range(-3, 4):
                for dx in range(-3, 4):
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < self.heatmap_data.shape[1] and 0 <= ny < self.heatmap_data.shape[0]:
                        dist = np.sqrt(dx*dx + dy*dy)
                        self.heatmap_data[ny, nx] += max(0, 1 - dist/4)
    
    def set_current_frame(self, frame):
        with self.frame_lock:
            self.current_display_frame = frame.copy()
    
    def get_current_frame(self):
        with self.frame_lock:
            if self.current_display_frame is not None:
                return self.current_display_frame.copy()
            return None
    
    def add_log(self, level: str, message: str, category: str = "system"):
        log_entry = {
            'id': len(self.logs) + 1,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'level': level,
            'category': category,
            'message': message
        }
        with self.lock:
            self.logs.append(log_entry)
            if len(self.logs) > 1000:
                self.logs = self.logs[-1000:]
        
        try:
            socketio.emit('new_log', log_entry)
        except:
            pass
        return log_entry
    
    def add_vehicle(self, vehicle_id: int, vehicle_type: str, bbox: tuple):
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        import random
        
        with self.lock:
            if vehicle_id not in self.vehicles:
                self.vehicles[vehicle_id] = VehicleRecord(
                    vehicle_id=vehicle_id,
                    vehicle_type=vehicle_type,
                    first_seen=now,
                    last_seen=now,
                    license_plate="",
                    confidence=0.0,
                    speed_estimate=random.uniform(25, 75),
                    lane=random.randint(1, 3),
                    violations=[],
                    frame_count=1
                )
                self.stats['total_vehicles'] += 1
                self.stats['vehicle_types'][vehicle_type] += 1
            else:
                self.vehicles[vehicle_id].last_seen = now
                self.vehicles[vehicle_id].frame_count += 1
            
            x1, y1, x2, y2 = bbox
            self.update_heatmap(x1, y1, x2, y2)
    
    def add_plate_detection(self, plate_text: str, confidence: float, vehicle_id: int, 
                           vehicle_type: str, frame_number: int):
        if not plate_text or plate_text == '0':
            return
        
        detection = PlateDetection(
            id=len(self.plate_detections) + 1,
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            plate_text=plate_text,
            confidence=confidence,
            vehicle_id=vehicle_id,
            vehicle_type=vehicle_type,
            frame_number=frame_number
        )
        
        with self.lock:
            self.plate_detections.append(detection)
            self.stats['total_plates'] += 1
            
            if vehicle_id in self.vehicles:
                if confidence > self.vehicles[vehicle_id].confidence:
                    self.vehicles[vehicle_id].license_plate = plate_text
                    self.vehicles[vehicle_id].confidence = confidence
        
        self.add_log('info', f"Plate detected: {plate_text} (Vehicle #{vehicle_id})", 'detection')
        
        try:
            socketio.emit('new_plate', asdict(detection))
        except:
            pass
        
        self.check_violations(vehicle_id, plate_text)
    
    def check_violations(self, vehicle_id: int, plate_text: str):
        import random
        
        if vehicle_id not in self.vehicles:
            return
        
        vehicle = self.vehicles[vehicle_id]
        violations_to_add = []
        
        # Speeding detection
        if vehicle.speed_estimate > 60:
            violations_to_add.append({
                'type': 'SPEEDING',
                'severity': 'high' if vehicle.speed_estimate > 80 else 'medium',
                'description': f"Vehicle traveling at {vehicle.speed_estimate:.1f} km/h in 60 km/h zone"
            })
        
        # Random violations for demo
        if random.random() < 0.03:
            violation_types = [
                ('LANE_VIOLATION', 'medium', 'Improper lane change detected'),
                ('STOP_SIGN', 'high', 'Failed to stop at stop sign'),
                ('RED_LIGHT', 'critical', 'Red light violation detected'),
            ]
            vtype, sev, desc = random.choice(violation_types)
            violations_to_add.append({'type': vtype, 'severity': sev, 'description': desc})
        
        for v in violations_to_add:
            if v['type'] not in vehicle.violations:
                vehicle.violations.append(v['type'])
                
                # Get challan amount for this violation type
                challan_amount = self.challan_rates.get(v['type'], 500)
                
                violation = ViolationRecord(
                    id=len(self.violations) + 1,
                    timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    vehicle_id=vehicle_id,
                    license_plate=plate_text,
                    violation_type=v['type'],
                    severity=v['severity'],
                    description=v['description'],
                    speed=vehicle.speed_estimate if v['type'] == 'SPEEDING' else None,
                    location=f"Lane {vehicle.lane}, Camera 1",
                    snapshot_frame=self.current_frame,
                    challan_amount=challan_amount
                )
                
                with self.lock:
                    self.violations.append(violation)
                    self.stats['total_violations'] += 1
                    self.stats['violation_types'][v['type']] += 1
                    self.stats['total_challan'] += challan_amount
                    self.stats['challan_count'] += 1
                
                log_level = 'critical' if v['severity'] == 'critical' else 'warning'
                self.add_log(log_level, f"CHALLAN ₹{challan_amount}: {v['type']} - {v['description']} (Plate: {plate_text})", 'violation')
                
                try:
                    socketio.emit('new_violation', asdict(violation))
                except:
                    pass
    
    def get_heatmap_image(self):
        if self.heatmap_data is None or self.heatmap_data.max() == 0:
            heatmap = np.zeros((108, 192, 3), dtype=np.uint8)
        else:
            normalized = self.heatmap_data / (self.heatmap_data.max() + 0.001)
            heatmap = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
            heatmap = cv2.resize(heatmap, (384, 216))
        
        _, buffer = cv2.imencode('.jpg', heatmap)
        return buffer.tobytes()
    
    def get_statistics(self):
        with self.lock:
            return {
                'total_vehicles': self.stats['total_vehicles'],
                'total_plates': self.stats['total_plates'],
                'total_violations': self.stats['total_violations'],
                'active_vehicles': len([v for v in self.vehicles.values() if v.frame_count > 0]),
                'violation_types': dict(self.stats['violation_types']),
                'vehicle_types': dict(self.stats['vehicle_types']),
                'current_frame': self.current_frame,
                'total_frames': self.total_frames,
                'is_processing': self.is_processing,
                'total_challan': self.stats['total_challan'],
                'challan_count': self.stats['challan_count'],
            }
    
    def get_recent_logs(self, count=50):
        with self.lock:
            return self.logs[-count:]
    
    def get_recent_violations(self, count=20):
        with self.lock:
            return [asdict(v) for v in self.violations[-count:]]
    
    def get_recent_plates(self, count=20):
        with self.lock:
            return [asdict(p) for p in self.plate_detections[-count:]]
    
    def get_vehicles(self):
        with self.lock:
            return [asdict(v) for v in self.vehicles.values()]


# Global data store
data_store = ALPRDataStore()


# =============================================================================
# OCR FUNCTIONS
# =============================================================================

reader = None

def init_ocr():
    global reader
    if reader is None:
        print("Initializing EasyOCR...")
        reader = easyocr.Reader(['en'], gpu=OCR_GPU)
        print(f"✅ EasyOCR initialized (GPU: {OCR_GPU})")

dict_char_to_int = {'O': '0', 'I': '1', 'J': '3', 'A': '4', 'G': '6', 'S': '5'}
dict_int_to_char = {'0': 'O', '1': 'I', '3': 'J', '4': 'A', '6': 'G', '5': 'S'}

def license_complies_format(text):
    if len(text) != 7:
        return False
    if (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys()) and \
       (text[1] in string.ascii_uppercase or text[1] in dict_int_to_char.keys()) and \
       (text[2] in '0123456789' or text[2] in dict_char_to_int.keys()) and \
       (text[3] in '0123456789' or text[3] in dict_char_to_int.keys()) and \
       (text[4] in string.ascii_uppercase or text[4] in dict_int_to_char.keys()) and \
       (text[5] in string.ascii_uppercase or text[5] in dict_int_to_char.keys()) and \
       (text[6] in string.ascii_uppercase or text[6] in dict_int_to_char.keys()):
        return True
    return False

def format_license(text):
    license_plate_ = ''
    mapping = {
        0: dict_int_to_char, 1: dict_int_to_char,
        4: dict_int_to_char, 5: dict_int_to_char, 6: dict_int_to_char,
        2: dict_char_to_int, 3: dict_char_to_int
    }
    for j in range(7):
        if text[j] in mapping[j].keys():
            license_plate_ += mapping[j][text[j]]
        else:
            license_plate_ += text[j]
    return license_plate_

def read_license_plate(license_plate_crop):
    global reader
    if reader is None:
        init_ocr()
    
    detections = reader.readtext(license_plate_crop)
    for detection in detections:
        bbox, text, score = detection
        text = text.upper().replace(' ', '')
        if license_complies_format(text):
            return format_license(text), score
    return None, None

def get_car(license_plate, vehicle_track_ids):
    x1, y1, x2, y2, score, class_id = license_plate
    for vehicle in vehicle_track_ids:
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle
        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            return vehicle
    return -1, -1, -1, -1, -1


# =============================================================================
# SMOOTHING CLASSES
# =============================================================================

class BoxSmoother:
    def __init__(self, window=5):
        self.window = window
        self.buffers = defaultdict(lambda: deque(maxlen=self.window))

    def update(self, track_id, bbox):
        self.buffers[track_id].append(np.array(bbox, dtype=float))
        arr = np.stack(self.buffers[track_id], axis=0)
        return arr.mean(axis=0).tolist()

    def get(self, track_id, default_bbox=None):
        buf = self.buffers.get(track_id, None)
        if buf and len(buf) > 0:
            arr = np.stack(buf, axis=0)
            return arr.mean(axis=0).tolist()
        return default_bbox


class PlateSmoother:
    def __init__(self, bbox_window=5):
        self.bbox_window = bbox_window
        self.bbox_buffers = defaultdict(lambda: deque(maxlen=self.bbox_window))
        self.best_text = {}

    def update_bbox(self, track_id, bbox):
        self.bbox_buffers[track_id].append(np.array(bbox, dtype=float))
        arr = np.stack(self.bbox_buffers[track_id], axis=0)
        return arr.mean(axis=0).tolist()

    def update_text(self, track_id, text, score):
        if text is None or text == '':
            return
        prev = self.best_text.get(track_id, {'text': '0', 'score': 0.0})
        if score is None:
            score = 0.0
        if score >= prev['score']:
            self.best_text[track_id] = {'text': text, 'score': float(score)}

    def get_best_text(self, track_id):
        return self.best_text.get(track_id, {'text': '0', 'score': 0.0})


# =============================================================================
# ALPR PIPELINE (runs in background thread)
# =============================================================================

def run_alpr_pipeline(video_path, coco_weights, lp_weights, vehicles, target_fps, smooth_window):
    """Run the ALPR pipeline in a background thread"""
    global data_store
    
    data_store.add_log('info', 'Starting ALPR Pipeline...', 'system')
    data_store.add_log('info', f'Video: {video_path}', 'system')
    
    # Initialize OCR
    init_ocr()
    
    # Load models
    data_store.add_log('info', 'Loading YOLOv8 models...', 'system')
    coco_model = YOLO(coco_weights)
    lp_model = YOLO(lp_weights)
    data_store.add_log('success', 'Models loaded successfully!', 'system')
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        data_store.add_log('critical', f'Cannot open video: {video_path}', 'system')
        return
    
    input_fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_step = max(1, int(input_fps / target_fps))
    
    data_store.reset_heatmap(width, height)
    data_store.total_frames = total_frames
    data_store.is_processing = True
    
    data_store.add_log('info', f'Video: {width}x{height} @ {input_fps} FPS ({total_frames} frames)', 'system')
    
    # Initialize trackers
    mot_tracker = Sort()
    car_smoother = BoxSmoother(window=smooth_window)
    plate_smoother = PlateSmoother(bbox_window=smooth_window)
    
    # Vehicle type mapping
    vehicle_type_map = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
    
    frame_idx = -1
    processed_count = 0
    
    data_store.add_log('info', 'Processing started...', 'system')
    
    while True:
        frame_idx += 1
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % frame_step != 0:
            continue
        
        processed_count += 1
        data_store.current_frame = frame_idx
        
        # Detect vehicles
        det = coco_model(frame, verbose=False)[0]
        detections_ = []
        for d in det.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = d
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])
        
        # Track vehicles
        tracks = mot_tracker.update(np.asarray(detections_))
        
        # Detect license plates
        lp_det = lp_model(frame, verbose=False)[0]
        lp_boxes = lp_det.boxes.data.tolist() if lp_det.boxes is not None else []
        
        # Process each detected plate
        for lp in lp_boxes:
            x1, y1, x2, y2, lp_score, _ = lp
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(lp, tracks)
            
            if car_id == -1:
                continue
            
            # Crop and process plate
            lp_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
            if lp_crop.size == 0:
                continue
            
            lp_gray = cv2.cvtColor(lp_crop, cv2.COLOR_BGR2GRAY)
            _, lp_thresh = cv2.threshold(lp_gray, 64, 255, cv2.THRESH_BINARY_INV)
            
            lp_text, lp_text_score = read_license_plate(lp_thresh)
            plate_smoother.update_text(int(car_id), lp_text, lp_text_score)
            
            sm_lp_bbox = plate_smoother.update_bbox(int(car_id), [x1, y1, x2, y2])
            
            # Draw plate box
            sx1, sy1, sx2, sy2 = map(int, sm_lp_bbox)
            cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), (0, 0, 255), 3)
            cv2.putText(frame, f"LP:{lp_text or '0'}", (sx1, max(0, sy1 - 8)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Add to data store
            if lp_text:
                # Determine vehicle type
                vehicle_type = 'car'
                for d in det.boxes.data.tolist():
                    if int(d[5]) in vehicle_type_map:
                        vehicle_type = vehicle_type_map[int(d[5])]
                        break
                
                data_store.add_plate_detection(
                    lp_text, lp_text_score or 0.0, int(car_id),
                    vehicle_type, frame_idx
                )
        
        # Draw vehicle tracks
        for tr in tracks:
            tx1, ty1, tx2, ty2, tid = tr
            sm_car_bbox = car_smoother.update(int(tid), [tx1, ty1, tx2, ty2])
            cx1, cy1, cx2, cy2 = map(int, sm_car_bbox)
            
            cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (0, 255, 0), 3)
            cv2.putText(frame, f"ID:{int(tid)}", (cx1, max(0, cy1 - 8)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add vehicle to data store
            vehicle_type = 'car'
            data_store.add_vehicle(int(tid), vehicle_type, (tx1, ty1, tx2, ty2))
            
            # Overlay best plate text
            best = plate_smoother.get_best_text(int(tid))
            plate_text = best['text']
            if plate_text and plate_text != '0':
                (tw, th), _ = cv2.getTextSize(plate_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
                text_x = int((cx1 + cx2 - tw) / 2)
                text_y = max(0, cy1 - 20)
                (text_w, text_h), _ = cv2.getTextSize(plate_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)
                bg_x1 = text_x - 10
                bg_y1 = text_y - text_h - 10
                bg_x2 = text_x + text_w + 10
                bg_y2 = text_y + 10
                cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (255, 255, 255), -1)
                cv2.putText(frame, plate_text, (text_x, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
        
        # Add info overlay
        car_ids = set()
        plate_texts = []
        for tr in tracks:
            tid = int(tr[4])
            car_ids.add(tid)
            best = plate_smoother.get_best_text(tid)
            pt = best['text']
            if pt and pt != '0' and pt not in plate_texts:
                plate_texts.append(pt)
        
        overlay = frame.copy()
        box_h = 80
        cv2.rectangle(overlay, (10, 10), (430, 10 + box_h), (40, 40, 40), -1)
        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
        
        cv2.putText(frame, f"Cars: {len(car_ids)} | Plates: {len(plate_texts)}", (25, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        if plate_texts:
            cv2.putText(frame, f"Plates: {', '.join(plate_texts[:3])}", (25, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 136), 2)
        
        # Update current frame for streaming
        data_store.set_current_frame(frame)
        
        # Emit stats update periodically
        if processed_count % 5 == 0:
            try:
                socketio.emit('stats_update', data_store.get_statistics())
            except:
                pass
    
    cap.release()
    data_store.is_processing = False
    data_store.add_log('success', f'Processing complete! {processed_count} frames processed.', 'system')


# =============================================================================
# VIDEO STREAMING
# =============================================================================

def generate_frames():
    """Generator for video frames from ALPR processing"""
    global data_store
    
    # Wait for first frame
    while data_store.get_current_frame() is None and data_store.is_processing:
        time.sleep(0.1)
    
    # If not processing, stream from output video
    if not data_store.is_processing:
        cap = cv2.VideoCapture('Videos/out.avi')
        if cap.isOpened():
            while True:
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                frame = cv2.resize(frame, (960, 540))
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                time.sleep(0.033)
    else:
        # Stream live from processing
        while data_store.is_processing or data_store.get_current_frame() is not None:
            frame = data_store.get_current_frame()
            if frame is not None:
                frame = cv2.resize(frame, (960, 540))
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.033)


# =============================================================================
# FLASK ROUTES
# =============================================================================

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/heatmap')
def heatmap():
    return Response(data_store.get_heatmap_image(), mimetype='image/jpeg')

@app.route('/api/stats')
def api_stats():
    return jsonify(data_store.get_statistics())

@app.route('/api/logs')
def api_logs():
    count = int(request.args.get('count', 50))
    return jsonify(data_store.get_recent_logs(count))

@app.route('/api/violations')
def api_violations():
    count = int(request.args.get('count', 20))
    return jsonify(data_store.get_recent_violations(count))

@app.route('/api/plates')
def api_plates():
    count = int(request.args.get('count', 20))
    return jsonify(data_store.get_recent_plates(count))

@app.route('/api/vehicles')
def api_vehicles():
    return jsonify(data_store.get_vehicles())

@app.route('/api/chart/violations')
def api_chart_violations():
    stats = data_store.get_statistics()
    return jsonify({
        'labels': list(stats['violation_types'].keys()),
        'data': list(stats['violation_types'].values())
    })

@app.route('/api/chart/vehicles')
def api_chart_vehicles():
    stats = data_store.get_statistics()
    return jsonify({
        'labels': list(stats['vehicle_types'].keys()),
        'data': list(stats['vehicle_types'].values())
    })

@app.route('/Videos/<path:filename>')
def serve_video(filename):
    return send_from_directory('Videos', filename)


# =============================================================================
# SOCKET EVENTS
# =============================================================================

@socketio.on('connect')
def handle_connect():
    emit('connected', {'status': 'Connected to ALPR Dashboard'})
    data_store.add_log('info', 'Web client connected', 'system')

@socketio.on('disconnect')
def handle_disconnect():
    data_store.add_log('info', 'Web client disconnected', 'system')

@socketio.on('request_update')
def handle_update_request():
    emit('stats_update', data_store.get_statistics())


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='ALPR Web Dashboard with Live Processing')
    parser.add_argument('--host', default='127.0.0.1', help='Host address')
    parser.add_argument('--port', type=int, default=5000, help='Port number')
    parser.add_argument('--video', default='Videos/sample.mp4', help='Input video path')
    parser.add_argument('--no-process', action='store_true', help='Just serve dashboard (no ALPR processing)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("ALPR WEB DASHBOARD WITH LIVE PROCESSING")
    print("=" * 70)
    print(f"🌐 Dashboard URL: http://{args.host}:{args.port}")
    print(f"📹 Video: {args.video}")
    print("=" * 70)
    
    if not args.no_process:
        # Start ALPR pipeline in background
        alpr_thread = threading.Thread(
            target=run_alpr_pipeline,
            args=(
                args.video,
                'yolov8n.pt',
                'license_plate_detector.pt',
                (2, 3, 5, 7),
                10,
                5
            ),
            daemon=True
        )
        alpr_thread.start()
        print("🚀 ALPR processing started in background")
    else:
        print("📺 Running in view-only mode (no processing)")
    
    print(f"\n🌐 Open http://{args.host}:{args.port} in your browser\n")
    
    socketio.run(app, host=args.host, port=args.port, debug=False, allow_unsafe_werkzeug=True)


if __name__ == '__main__':
    main()
