"""
ALPR Web Dashboard
A comprehensive web interface for viewing ALPR output, logs, violations, and analytics.
"""

import os
import json
import random
import threading
import time
from datetime import datetime, timedelta
from flask import Flask, render_template, Response, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional

# Initialize Flask app
app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['SECRET_KEY'] = 'alpr_secret_key_2024'
socketio = SocketIO(app, cors_allowed_origins="*")

# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class VehicleRecord:
    """Record of a detected vehicle"""
    vehicle_id: int
    vehicle_type: str
    first_seen: str
    last_seen: str
    license_plate: str
    confidence: float
    speed_estimate: float  # km/h (estimated)
    lane: int
    violations: List[str]
    frame_count: int

@dataclass
class ViolationRecord:
    """Record of a traffic violation"""
    id: int
    timestamp: str
    vehicle_id: int
    license_plate: str
    violation_type: str
    severity: str  # low, medium, high, critical
    description: str
    speed: Optional[float]
    location: str
    snapshot_frame: int

@dataclass
class PlateDetection:
    """Record of license plate detection"""
    id: int
    timestamp: str
    plate_text: str
    confidence: float
    vehicle_id: int
    vehicle_type: str
    frame_number: int

# =============================================================================
# GLOBAL DATA STORAGE
# =============================================================================

class ALPRDataStore:
    """Central data storage for ALPR analytics"""
    
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
            'vehicles_per_minute': [],
            'violation_types': defaultdict(int),
            'vehicle_types': defaultdict(int),
            'hourly_traffic': defaultdict(int),
            'speed_distribution': []
        }
        self.logs: List[dict] = []
        self.is_processing = False
        self.current_frame = 0
        self.total_frames = 0
        self.lock = threading.Lock()
        
        # Initialize heatmap
        self.reset_heatmap(1920, 1080)
    
    def reset_heatmap(self, width, height):
        """Initialize/reset the heatmap"""
        self.frame_width = width
        self.frame_height = height
        self.heatmap_data = np.zeros((height // 10, width // 10), dtype=np.float32)
    
    def update_heatmap(self, x1, y1, x2, y2):
        """Update heatmap with vehicle position"""
        cx = int((x1 + x2) / 2 / 10)
        cy = int((y1 + y2) / 2 / 10)
        if 0 <= cx < self.heatmap_data.shape[1] and 0 <= cy < self.heatmap_data.shape[0]:
            # Add gaussian-like spread
            for dy in range(-3, 4):
                for dx in range(-3, 4):
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < self.heatmap_data.shape[1] and 0 <= ny < self.heatmap_data.shape[0]:
                        dist = np.sqrt(dx*dx + dy*dy)
                        self.heatmap_data[ny, nx] += max(0, 1 - dist/4)
    
    def add_log(self, level: str, message: str, category: str = "system"):
        """Add a log entry"""
        log_entry = {
            'id': len(self.logs) + 1,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'level': level,
            'category': category,
            'message': message
        }
        with self.lock:
            self.logs.append(log_entry)
            if len(self.logs) > 1000:  # Keep last 1000 logs
                self.logs = self.logs[-1000:]
        
        # Emit to connected clients
        socketio.emit('new_log', log_entry)
        return log_entry
    
    def add_vehicle(self, vehicle_id: int, vehicle_type: str, bbox: tuple):
        """Add or update a vehicle record"""
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        with self.lock:
            if vehicle_id not in self.vehicles:
                self.vehicles[vehicle_id] = VehicleRecord(
                    vehicle_id=vehicle_id,
                    vehicle_type=vehicle_type,
                    first_seen=now,
                    last_seen=now,
                    license_plate="",
                    confidence=0.0,
                    speed_estimate=random.uniform(20, 80),  # Simulated speed
                    lane=random.randint(1, 3),
                    violations=[],
                    frame_count=1
                )
                self.stats['total_vehicles'] += 1
                self.stats['vehicle_types'][vehicle_type] += 1
            else:
                self.vehicles[vehicle_id].last_seen = now
                self.vehicles[vehicle_id].frame_count += 1
            
            # Update heatmap
            x1, y1, x2, y2 = bbox
            self.update_heatmap(x1, y1, x2, y2)
    
    def add_plate_detection(self, plate_text: str, confidence: float, vehicle_id: int, 
                           vehicle_type: str, frame_number: int):
        """Add a license plate detection"""
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
            
            # Update vehicle record
            if vehicle_id in self.vehicles:
                if confidence > self.vehicles[vehicle_id].confidence:
                    self.vehicles[vehicle_id].license_plate = plate_text
                    self.vehicles[vehicle_id].confidence = confidence
        
        # Log the detection
        self.add_log('info', f"Plate detected: {plate_text} (Vehicle #{vehicle_id}, Confidence: {confidence:.2f})", 'detection')
        
        # Emit to clients
        socketio.emit('new_plate', asdict(detection))
        
        # Check for violations
        self.check_violations(vehicle_id, plate_text)
    
    def check_violations(self, vehicle_id: int, plate_text: str):
        """Check for traffic violations (simulated rules)"""
        if vehicle_id not in self.vehicles:
            return
        
        vehicle = self.vehicles[vehicle_id]
        violations_to_add = []
        
        # Rule 1: Speeding (simulated - speed > 60 km/h in zone)
        if vehicle.speed_estimate > 60:
            violations_to_add.append({
                'type': 'SPEEDING',
                'severity': 'high' if vehicle.speed_estimate > 80 else 'medium',
                'description': f"Vehicle traveling at {vehicle.speed_estimate:.1f} km/h in 60 km/h zone"
            })
        
        # Rule 2: Check plate format (invalid plates)
        if len(plate_text) < 5:
            violations_to_add.append({
                'type': 'INVALID_PLATE',
                'severity': 'medium',
                'description': f"License plate format appears invalid: {plate_text}"
            })
        
        # Rule 3: Random violation simulation (for demo)
        if random.random() < 0.05:  # 5% chance
            violation_types = [
                ('LANE_VIOLATION', 'medium', 'Improper lane change detected'),
                ('STOP_SIGN', 'high', 'Failed to stop at stop sign'),
                ('RED_LIGHT', 'critical', 'Red light violation detected'),
            ]
            vtype, sev, desc = random.choice(violation_types)
            violations_to_add.append({
                'type': vtype,
                'severity': sev,
                'description': desc
            })
        
        # Add violations
        for v in violations_to_add:
            if v['type'] not in vehicle.violations:
                vehicle.violations.append(v['type'])
                
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
                    snapshot_frame=self.current_frame
                )
                
                with self.lock:
                    self.violations.append(violation)
                    self.stats['total_violations'] += 1
                    self.stats['violation_types'][v['type']] += 1
                
                # Log violation
                log_level = 'critical' if v['severity'] == 'critical' else 'warning'
                self.add_log(log_level, f"VIOLATION: {v['type']} - {v['description']} (Plate: {plate_text})", 'violation')
                
                # Emit to clients
                socketio.emit('new_violation', asdict(violation))
    
    def get_heatmap_image(self):
        """Generate heatmap visualization"""
        if self.heatmap_data is None or self.heatmap_data.max() == 0:
            # Return empty heatmap
            heatmap = np.zeros((108, 192, 3), dtype=np.uint8)
        else:
            # Normalize and colorize
            normalized = self.heatmap_data / (self.heatmap_data.max() + 0.001)
            heatmap = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        # Encode as JPEG
        _, buffer = cv2.imencode('.jpg', heatmap)
        return buffer.tobytes()
    
    def get_statistics(self):
        """Get current statistics"""
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
                'is_processing': self.is_processing
            }
    
    def get_recent_logs(self, count=50):
        """Get recent logs"""
        with self.lock:
            return self.logs[-count:]
    
    def get_recent_violations(self, count=20):
        """Get recent violations"""
        with self.lock:
            return [asdict(v) for v in self.violations[-count:]]
    
    def get_recent_plates(self, count=20):
        """Get recent plate detections"""
        with self.lock:
            return [asdict(p) for p in self.plate_detections[-count:]]
    
    def get_vehicles(self):
        """Get all vehicles"""
        with self.lock:
            return [asdict(v) for v in self.vehicles.values()]


# Global data store
data_store = ALPRDataStore()


# =============================================================================
# VIDEO STREAMING
# =============================================================================

class VideoStreamer:
    """Stream video output to web clients"""
    
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = None
        self.lock = threading.Lock()
    
    def get_frame(self):
        """Get a single frame from the video"""
        with self.lock:
            if self.cap is None or not self.cap.isOpened():
                self.cap = cv2.VideoCapture(self.video_path)
                if not self.cap.isOpened():
                    # Return placeholder frame
                    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(placeholder, "Video not available", (150, 240),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    _, buffer = cv2.imencode('.jpg', placeholder)
                    return buffer.tobytes()
            
            ret, frame = self.cap.read()
            if not ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop video
                ret, frame = self.cap.read()
            
            if ret:
                # Resize for web streaming
                frame = cv2.resize(frame, (960, 540))
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                return buffer.tobytes()
            
            return None
    
    def generate_frames(self):
        """Generator for video frames"""
        while True:
            frame = self.get_frame()
            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.033)  # ~30 FPS


# Video streamer instance
video_streamer = None


# =============================================================================
# FLASK ROUTES
# =============================================================================

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    global video_streamer
    if video_streamer is None:
        video_streamer = VideoStreamer('Videos/out.avi')
    return Response(video_streamer.generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/heatmap')
def heatmap():
    """Heatmap image route"""
    return Response(data_store.get_heatmap_image(),
                   mimetype='image/jpeg')

@app.route('/api/stats')
def api_stats():
    """API: Get statistics"""
    return jsonify(data_store.get_statistics())

@app.route('/api/logs')
def api_logs():
    """API: Get recent logs"""
    count = int(request.args.get('count', 50))
    return jsonify(data_store.get_recent_logs(count))

@app.route('/api/violations')
def api_violations():
    """API: Get recent violations"""
    count = int(request.args.get('count', 20))
    return jsonify(data_store.get_recent_violations(count))

@app.route('/api/plates')
def api_plates():
    """API: Get recent plate detections"""
    count = int(request.args.get('count', 20))
    return jsonify(data_store.get_recent_plates(count))

@app.route('/api/vehicles')
def api_vehicles():
    """API: Get all vehicles"""
    return jsonify(data_store.get_vehicles())

@app.route('/api/chart/violations')
def api_chart_violations():
    """API: Get violation chart data"""
    stats = data_store.get_statistics()
    return jsonify({
        'labels': list(stats['violation_types'].keys()),
        'data': list(stats['violation_types'].values())
    })

@app.route('/api/chart/vehicles')
def api_chart_vehicles():
    """API: Get vehicle type chart data"""
    stats = data_store.get_statistics()
    return jsonify({
        'labels': list(stats['vehicle_types'].keys()),
        'data': list(stats['vehicle_types'].values())
    })

@app.route('/Videos/<path:filename>')
def serve_video(filename):
    """Serve video files"""
    return send_from_directory('Videos', filename)


# =============================================================================
# SOCKETIO EVENTS
# =============================================================================

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    emit('connected', {'status': 'Connected to ALPR Dashboard'})
    data_store.add_log('info', 'Client connected to dashboard', 'system')

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    data_store.add_log('info', 'Client disconnected from dashboard', 'system')

@socketio.on('request_update')
def handle_update_request():
    """Handle update request from client"""
    emit('stats_update', data_store.get_statistics())


# =============================================================================
# DEMO DATA GENERATOR (for testing without ALPR running)
# =============================================================================

def generate_demo_data():
    """Generate demo data for testing the dashboard"""
    vehicle_types = ['car', 'truck', 'motorcycle', 'bus']
    plate_chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    
    data_store.add_log('info', 'ALPR Dashboard started', 'system')
    data_store.add_log('info', 'Loading video: Videos/out.avi', 'system')
    
    vehicle_id = 0
    
    while True:
        # Simulate vehicle detection
        vehicle_id += 1
        vtype = random.choice(vehicle_types)
        
        # Random bounding box
        x1 = random.randint(100, 1500)
        y1 = random.randint(100, 800)
        x2 = x1 + random.randint(100, 300)
        y2 = y1 + random.randint(80, 200)
        
        data_store.add_vehicle(vehicle_id, vtype, (x1, y1, x2, y2))
        
        # Simulate plate detection (70% chance)
        if random.random() < 0.7:
            plate = ''.join(random.choices(plate_chars, k=7))
            confidence = random.uniform(0.6, 0.99)
            data_store.current_frame += random.randint(1, 10)
            data_store.add_plate_detection(plate, confidence, vehicle_id, vtype, 
                                          data_store.current_frame)
        
        # Update stats
        data_store.current_frame += 1
        data_store.total_frames = 3600
        data_store.is_processing = True
        
        # Emit stats update
        socketio.emit('stats_update', data_store.get_statistics())
        
        time.sleep(random.uniform(0.5, 2.0))


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def run_dashboard(host='127.0.0.1', port=5000, demo_mode=False):
    """Run the ALPR dashboard"""
    print("=" * 70)
    print("ALPR WEB DASHBOARD")
    print("=" * 70)
    print(f"🌐 Dashboard URL: http://{host}:{port}")
    print(f"📹 Video Stream: http://{host}:{port}/video_feed")
    print(f"🔥 Heatmap: http://{host}:{port}/heatmap")
    print("=" * 70)
    
    if demo_mode:
        print("🎮 Running in DEMO mode - generating simulated data")
        demo_thread = threading.Thread(target=generate_demo_data, daemon=True)
        demo_thread.start()
    
    socketio.run(app, host=host, port=port, debug=False, allow_unsafe_werkzeug=True)


if __name__ == '__main__':
    import argparse
    from flask import request
    
    parser = argparse.ArgumentParser(description='ALPR Web Dashboard')
    parser.add_argument('--host', default='127.0.0.1', help='Host address')
    parser.add_argument('--port', type=int, default=5000, help='Port number')
    parser.add_argument('--demo', action='store_true', help='Run in demo mode with simulated data')
    
    args = parser.parse_args()
    run_dashboard(host=args.host, port=args.port, demo_mode=args.demo)
