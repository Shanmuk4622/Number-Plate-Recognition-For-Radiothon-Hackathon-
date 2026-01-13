# alpr_realtime.py
# Combined pipeline: detection -> interpolation -> visualization (real-time)

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

# If you have scipy installed, you can use it; otherwise fallback to manual linear interpolation
try:
    from scipy.interpolate import interp1d
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

# --- Your project utilities ---
# Ensure util.py provides: get_car, read_license_plate, write_csv
from util import get_car, read_license_plate, write_csv
from sort.sort import Sort


# =========================
# Detection (YOLO + SORT)
# =========================
def run_detection(video_path, coco_weights='yolov8n.pt', lp_weights='license_plate_detector.pt',
                  vehicles=(2, 3, 5, 7), target_fps=10, display_size=(640, 360), show=True):
    """
    Runs YOLO + SORT on the input video, collects raw results per frame, and optionally displays frames.
    Returns a list of dict rows compatible with DataFrame/CSV.
    """
    results_rows = []

    # Models
    coco_model = YOLO(coco_weights)
    license_plate_detector = YOLO(lp_weights)

    # Video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    frame_step = max(1, int(fps / target_fps))

    mot_tracker = Sort()

    frame_nmr = -1
    ret = True

    if show:
        cv2.namedWindow("ALPR", cv2.WINDOW_NORMAL)

    while ret:
        frame_nmr += 1
        ret, frame = cap.read()
        if not ret:
            break

        # Process only every 'frame_step' frame
        if frame_nmr % frame_step != 0:
            # Skip processing AND skip display to avoid confusion
            continue

        # -------------------------
        # 1) YOLO vehicle detection
        # -------------------------
        det = coco_model(frame)[0]
        detections_ = []
        yolo_vehicle_boxes = []  # for drawing YOLO boxes directly

        for detection in det.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])
                yolo_vehicle_boxes.append((int(x1), int(y1), int(x2), int(y2), float(score), int(class_id)))

        # -------------------------
        # 2) SORT tracking
        # -------------------------
        track_ids = mot_tracker.update(np.asarray(detections_))  # [x1,y1,x2,y2,id]

        # -------------------------
        # 3) YOLO license plates
        # -------------------------
        lp_det = license_plate_detector(frame)[0]
        yolo_lp_boxes = []
        for license_plate in lp_det.boxes.data.tolist():
            x1, y1, x2, y2, lp_score, lp_class_id = license_plate
            yolo_lp_boxes.append((int(x1), int(y1), int(x2), int(y2), float(lp_score)))

            # Assign license plate to car via IoU/overlap logic
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

            if car_id != -1:
                # Crop license plate
                lp_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
                if lp_crop.size == 0:
                    continue

                # Process license plate
                lp_gray = cv2.cvtColor(lp_crop, cv2.COLOR_BGR2GRAY)
                _, lp_thresh = cv2.threshold(lp_gray, 64, 255, cv2.THRESH_BINARY_INV)

                # Read license plate number
                lp_text, lp_text_score = read_license_plate(lp_thresh)

                # Store row
                row = {
                    'frame_nmr': frame_nmr,
                    'car_id': int(car_id),
                    'car_bbox': [float(xcar1), float(ycar1), float(xcar2), float(ycar2)],
                    'license_plate_bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'license_plate_bbox_score': float(lp_score),
                    'license_number': lp_text if lp_text is not None else '0',
                    'license_number_score': float(lp_text_score) if lp_text_score is not None else 0.0
                }
                results_rows.append(row)

        # -------------------------
        # 4) Draw everything clearly
        # -------------------------
        # Draw YOLO vehicle boxes (green)
        for (x1, y1, x2, y2, score, cls_id) in yolo_vehicle_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 3)
            cv2.putText(frame, f"YOLO {cls_id}:{score:.2f}", (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)

        # Draw SORT tracks (cyan)
        for tr in track_ids:
            tx1, ty1, tx2, ty2, tid = tr
            tx1, ty1, tx2, ty2 = int(tx1), int(ty1), int(tx2), int(ty2)
            cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), (255, 255, 0), 3)
            cv2.putText(frame, f"ID:{int(tid)}", (tx1, max(0, ty1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Draw YOLO license plate boxes (red)
        for (x1, y1, x2, y2, lp_score) in yolo_lp_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(frame, f"LP:{lp_score:.2f}", (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Debug overlay: counts
        cv2.putText(frame, f"Frames: {frame_nmr} | YOLO cars: {len(yolo_vehicle_boxes)} | LPs: {len(yolo_lp_boxes)} | Tracks: {len(track_ids)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 255, 50), 2)

        # -------------------------
        # 5) Show after drawing
        # -------------------------
        if show:
            display_frame = cv2.resize(frame, display_size)
            cv2.imshow("ALPR", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if show:
        cv2.destroyAllWindows()

    return results_rows



# =========================
# Interpolation (inline)
# =========================
def interpolate_bounding_boxes_inline(rows):
    """
    Interpolates missing frames for each car_id across car_bbox and license_plate_bbox.
    Input: list of dict rows with keys: frame_nmr, car_id, car_bbox, license_plate_bbox, license_plate_bbox_score, license_number, license_number_score
    Output: list of dict rows with imputed frames filled in (license fields set to '0' for imputed frames).
    """
    if not rows:
        return []

    # Convert to DataFrame for easier grouping
    df = pd.DataFrame(rows)
    df['frame_nmr'] = df['frame_nmr'].astype(int)
    df['car_id'] = df['car_id'].astype(int)

    # Sort by car_id and frame
    df = df.sort_values(['car_id', 'frame_nmr']).reset_index(drop=True)

    interpolated_rows = []

    for car_id, group in df.groupby('car_id'):
        frames = group['frame_nmr'].values
        car_bboxes = np.stack(group['car_bbox'].values)  # Nx4
        lp_bboxes = np.stack(group['license_plate_bbox'].values)  # Nx4

        first_frame = frames[0]
        last_frame = frames[-1]
        full_frames = np.arange(first_frame, last_frame + 1)

        # Interpolate car bbox
        if SCIPY_AVAILABLE and len(frames) >= 2:
            car_interp = interp1d(frames, car_bboxes, axis=0, kind='linear', fill_value='extrapolate')
            lp_interp = interp1d(frames, lp_bboxes, axis=0, kind='linear', fill_value='extrapolate')
            car_bboxes_full = car_interp(full_frames)
            lp_bboxes_full = lp_interp(full_frames)
        else:
            # Manual linear interpolation
            car_bboxes_full = manual_linear_interp(frames, car_bboxes, full_frames)
            lp_bboxes_full = manual_linear_interp(frames, lp_bboxes, full_frames)

        # Build rows for full frame range
        frame_set = set(frames)
        for i, f in enumerate(full_frames):
            row = {
                'frame_nmr': int(f),
                'car_id': int(car_id),
                'car_bbox': car_bboxes_full[i].tolist(),
                'license_plate_bbox': lp_bboxes_full[i].tolist()
            }

            if f in frame_set:
                # Original row: copy scores/text from df
                original = group[group['frame_nmr'] == f].iloc[0]
                row['license_plate_bbox_score'] = float(original['license_plate_bbox_score'])
                row['license_number'] = str(original['license_number'])
                row['license_number_score'] = float(original['license_number_score'])
            else:
                # Imputed row
                row['license_plate_bbox_score'] = 0.0
                row['license_number'] = '0'
                row['license_number_score'] = 0.0

            interpolated_rows.append(row)

    # Sort final rows by frame for visualization
    interpolated_rows.sort(key=lambda r: (r['frame_nmr'], r['car_id']))
    return interpolated_rows


def manual_linear_interp(frames, values, full_frames):
    """
    Manual linear interpolation for sequences without scipy.
    frames: array of known frame indices (N)
    values: array NxD of known values
    full_frames: array of target frames to fill (M)
    """
    D = values.shape[1]
    out = np.zeros((len(full_frames), D), dtype=float)

    # For each target frame, find surrounding known frames and interpolate
    for i, f in enumerate(full_frames):
        if f <= frames[0]:
            out[i] = values[0]
            continue
        if f >= frames[-1]:
            out[i] = values[-1]
            continue

        # find prev and next known frames
        idx_next = np.searchsorted(frames, f, side='right')
        idx_prev = idx_next - 1
        f0, f1 = frames[idx_prev], frames[idx_next]
        v0, v1 = values[idx_prev], values[idx_next]

        t = (f - f0) / (f1 - f0) if f1 != f0 else 0.0
        out[i] = v0 + t * (v1 - v0)

    return out


# =========================
# Visualization (live + save)
# =========================
def visualize_results(video_path, interpolated_rows, output_path="out.mp4",
                      display_size=(1280, 720), show=True):
    """
    Visualizes interpolated results: draws car borders, license plate boxes, overlays best license crop and text.
    Saves output video and optionally shows live window.
    """
    # Build DataFrame
    results = pd.DataFrame(interpolated_rows)

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Precompute best license crop per car_id
    license_plate = {}
    for car_id in np.unique(results['car_id']):
        df_car = results[results['car_id'] == car_id]
        max_score = df_car['license_number_score'].max()
        # If all zeros, still pick a frame to avoid KeyErrors
        df_best = df_car[df_car['license_number_score'] == max_score].iloc[0]

        # Seek to that frame and crop license
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(df_best['frame_nmr']))
        ret, frame = cap.read()
        if not ret:
            continue

        x1, y1, x2, y2 = df_best['license_plate_bbox']
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        license_crop = frame[y1:y2, x1:x2, :]
        if license_crop.size == 0:
            license_crop = np.zeros((100, 200, 3), dtype=np.uint8)

        # Resize crop to fixed height 400 while keeping aspect
        h = y2 - y1
        w = x2 - x1
        if h <= 0 or w <= 0:
            license_crop = np.zeros((400, 400, 3), dtype=np.uint8)
        else:
            new_h = 400
            new_w = int(w * (new_h / h))
            license_crop = cv2.resize(license_crop, (new_w, new_h))

        license_plate[car_id] = {
            'license_crop': license_crop,
            'license_plate_number': str(df_best['license_number'])
        }

    # Reset to start
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_nmr = -1
    ret = True

    if show:
        cv2.namedWindow("ALPR-Vis", cv2.WINDOW_NORMAL)

    while ret:
        ret, frame = cap.read()
        frame_nmr += 1
        if not ret:
            break

        df_frame = results[results['frame_nmr'] == frame_nmr]
        for _, row in df_frame.iterrows():
            # Draw car border (stylized)
            car_x1, car_y1, car_x2, car_y2 = row['car_bbox']
            car_x1, car_y1, car_x2, car_y2 = int(car_x1), int(car_y1), int(car_x2), int(car_y2)

            # Green rectangle for car
            cv2.rectangle(frame, (car_x1, car_y1), (car_x2, car_y2), (0, 255, 0), 4)

            # Draw license plate bbox
            x1, y1, x2, y2 = row['license_plate_bbox']
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)

            # Overlay best license crop and text
            car_id = int(row['car_id'])
            if car_id in license_plate:
                license_crop = license_plate[car_id]['license_crop']
                plate_text = license_plate[car_id]['license_plate_number']

                H, W, _ = license_crop.shape
                try:
                    # Place crop above car
                    top_y = max(0, car_y1 - H - 100)
                    left_x = int((car_x2 + car_x1 - W) / 2)
                    right_x = left_x + W
                    if top_y + H <= frame.shape[0] and left_x >= 0 and right_x <= frame.shape[1]:
                        frame[top_y:top_y + H, left_x:right_x, :] = license_crop

                        # White banner above crop for text
                        banner_top = max(0, top_y - 300)
                        banner_bottom = max(0, top_y - 100)
                        frame[banner_top:banner_bottom, left_x:right_x, :] = (255, 255, 255)

                        # Put text centered
                        (text_w, text_h), _ = cv2.getTextSize(plate_text, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 5)
                        text_x = int((car_x2 + car_x1 - text_w) / 2)
                        text_y = banner_top + int((banner_bottom - banner_top + text_h) / 2)
                        cv2.putText(frame, plate_text, (text_x, text_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 5)
                except Exception:
                    pass

        # Write to output
        out.write(frame)

        # Show live
        if show:
            display_frame = cv2.resize(frame, display_size)
            cv2.imshow("ALPR-Vis", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    out.release()
    cap.release()
    if show:
        cv2.destroyAllWindows()


# =========================
# Optional: Save CSVs
# =========================
def save_csv(rows, raw_csv_path='test.csv', interpolated_csv_path='test_interpolated.csv'):
    """
    Saves raw and interpolated CSVs for offline debugging or later analysis.
    """
    # Raw
    raw_df = pd.DataFrame(rows)
    raw_df.to_csv(raw_csv_path, index=False)

    # Interpolated
    interp_rows = interpolate_bounding_boxes_inline(rows)
    interp_df = pd.DataFrame(interp_rows)
    interp_df.to_csv(interpolated_csv_path, index=False)


# =========================
# Main
# =========================
if __name__ == "__main__":
    video_path = "Videos/sample2.mp4"

    # Step 1: Detection (process ~10 fps, show live small window)
    raw_rows = run_detection(
        video_path=video_path,
        coco_weights='yolov8n.pt',
        lp_weights='license_plate_detector.pt',
        vehicles=(2, 3, 5, 7),
        target_fps=10,
        display_size=(1920, 1080),
        show=True
    )

    # Optional: save raw CSV for debugging
    # save_csv(raw_rows, raw_csv_path='test.csv', interpolated_csv_path='test_interpolated.csv')

    # Step 2: Interpolation inline
    interpolated_rows = interpolate_bounding_boxes_inline(raw_rows)

    # Step 3: Visualization (live + save video)
    visualize_results(
        video_path=video_path,
        interpolated_rows=interpolated_rows,
        output_path="Videos/out.mp4",
        display_size=(1280, 720),
        show=True
    )
