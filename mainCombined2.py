# alpr_realtime_single_loop.py
import cv2
import numpy as np
from collections import deque, defaultdict
from ultralytics import YOLO

from util import get_car, read_license_plate
from sort.sort import Sort


# -----------------------------
# Smoothing utilities
# -----------------------------
class BoxSmoother:
    """
    Maintains a moving average of bounding boxes per track_id.
    """
    def __init__(self, window=5):
        self.window = window
        self.buffers = defaultdict(lambda: deque(maxlen=self.window))

    def update(self, track_id, bbox):
        """
        bbox: [x1, y1, x2, y2] floats
        """
        self.buffers[track_id].append(np.array(bbox, dtype=float))
        arr = np.stack(self.buffers[track_id], axis=0)  # N x 4
        return arr.mean(axis=0).tolist()

    def get(self, track_id, default_bbox=None):
        buf = self.buffers.get(track_id, None)
        if buf and len(buf) > 0:
            arr = np.stack(buf, axis=0)
            return arr.mean(axis=0).tolist()
        return default_bbox


class PlateSmoother:
    """
    Maintains the best plate text per track_id based on score,
    and a small buffer for plate bbox smoothing.
    """
    def __init__(self, bbox_window=5):
        self.bbox_window = bbox_window
        self.bbox_buffers = defaultdict(lambda: deque(maxlen=self.bbox_window))
        self.best_text = {}  # track_id -> {'text': str, 'score': float}

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


# -----------------------------
# Main real-time pipeline
# -----------------------------
def alpr_realtime(
    video_path="Videos/sample2.mp4",
    coco_weights="yolov8n.pt",
    lp_weights="license_plate_detector.pt",
    vehicles=(2, 3, 5, 7),
    target_fps=10,
    smooth_window=5,
    show=True,
    save_path="Videos/out.avi",
    codec="XVID"
):
    """
    Single-loop real-time ALPR:
    - Detect vehicles and plates
    - Track with SORT
    - OCR plates
    - Smooth boxes and plate text inline
    - Draw overlays
    - Display and save the same frame (default dimensions)
    """
    # Models
    coco_model = YOLO(coco_weights)
    lp_model = YOLO(lp_weights)

    # Video IO
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    frame_step = max(1, int(fps / target_fps))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
    if not out.isOpened():
        raise RuntimeError("VideoWriter failed to open. Try a different codec/container.")

    if show:
        cv2.namedWindow("ALPR-Realtime", cv2.WINDOW_NORMAL)

    # Tracking and smoothing
    mot_tracker = Sort()
    car_smoother = BoxSmoother(window=smooth_window)
    plate_smoother = PlateSmoother(bbox_window=smooth_window)

    frame_idx = -1
    ret = True

    while ret:
        frame_idx += 1
        ret, frame = cap.read()
        if not ret:
            break

        # Frame skipping to hit target FPS
        if frame_idx % frame_step != 0:
            continue

        # 1) Vehicle detection
        det = coco_model(frame)[0]
        detections_ = []
        for d in det.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = d
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])

        # 2) Tracking
        tracks = mot_tracker.update(np.asarray(detections_))  # [x1,y1,x2,y2,id]

        # 3) License plate detection + OCR
        lp_det = lp_model(frame)[0]
        lp_boxes = lp_det.boxes.data.tolist() if lp_det.boxes is not None else []

        # Map plates to cars
        for lp in lp_boxes:
            x1, y1, x2, y2, lp_score, _ = lp
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(lp, tracks)
            if car_id == -1:
                continue

            # Crop and OCR
            lp_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
            if lp_crop.size == 0:
                continue
            lp_gray = cv2.cvtColor(lp_crop, cv2.COLOR_BGR2GRAY)
            _, lp_thresh = cv2.threshold(lp_gray, 64, 255, cv2.THRESH_BINARY_INV)
            lp_text, lp_text_score = read_license_plate(lp_thresh)

            # Update best text per track
            plate_smoother.update_text(int(car_id), lp_text, lp_text_score)

            # Smooth plate bbox
            sm_lp_bbox = plate_smoother.update_bbox(int(car_id), [x1, y1, x2, y2])

            # Draw smoothed plate bbox
            sx1, sy1, sx2, sy2 = map(int, sm_lp_bbox)
            cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), (0, 0, 255), 3)
            cv2.putText(frame, f"LP:{lp_text or '0'}", (sx1, max(0, sy1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 4) Smooth and draw car tracks
        for tr in tracks:
            tx1, ty1, tx2, ty2, tid = tr
            sm_car_bbox = car_smoother.update(int(tid), [tx1, ty1, tx2, ty2])
            cx1, cy1, cx2, cy2 = map(int, sm_car_bbox)

            cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (0, 255, 0), 3)
            cv2.putText(frame, f"ID:{int(tid)}", (cx1, max(0, cy1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Overlay best plate text above car if available
            best = plate_smoother.get_best_text(int(tid))
            plate_text = best['text']
            if plate_text and plate_text != '0':
                (tw, th), _ = cv2.getTextSize(plate_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
                text_x = int((cx1 + cx2 - tw) / 2)
                text_y = max(0, cy1 - 20)
                # Measure text size
                (text_w, text_h), _ = cv2.getTextSize(plate_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)

                # Define background box
                bg_x1 = text_x - 10
                bg_y1 = text_y - text_h - 10
                bg_x2 = text_x + text_w + 10
                bg_y2 = text_y + 10

                # Draw white background
                cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (255, 255, 255), -1)

                # Draw black text on top
                cv2.putText(frame, plate_text, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)


        # 5) Write and display the same frame (default dimensions)
        out.write(frame)
        if show:
            cv2.imshow("ALPR-Realtime", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Cleanup
    out.release()
    cap.release()
    if show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    alpr_realtime(
        video_path="Videos/sample2.mp4",
        coco_weights="yolov8n.pt",
        lp_weights="license_plate_detector.pt",
        vehicles=(2, 3, 5, 7),
        target_fps=10,          # adjust for performance
        smooth_window=5,        # moving average window
        show=True,              # show live window
        save_path="Videos/out.avi",
        codec="XVID"            # safer on Windows; use 'mp4v' for MP4
    )
