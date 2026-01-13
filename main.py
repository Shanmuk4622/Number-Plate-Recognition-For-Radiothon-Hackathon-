from ultralytics import YOLO
import cv2
import numpy as np

import util
from sort.sort import *
from util import get_car, read_license_plate, write_csv

results = {}

mot_tracker = Sort()

# load models
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('license_plate_detector.pt')

# load video
cap = cv2.VideoCapture('Videos/sample.mp4')

vehicles = [2, 3, 5, 7]
# get video FPS
fps = int(cap.get(cv2.CAP_PROP_FPS))
print("Video FPS:", fps)

# we want 10 frames per second
target_fps = 10
frame_step = int(fps / target_fps)  # how many frames to skip
if frame_step < 1:
    frame_step = 1  # safety check

frame_nmr = -1
ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if not ret:
        break

    # process only every 'frame_step' frame
    if frame_nmr % frame_step != 0:
        continue

    results[frame_nmr] = {}

    # detect vehicles
    detections = coco_model(frame)[0]
    detections_ = []
    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        if int(class_id) in vehicles:
            detections_.append([x1, y1, x2, y2, score])

    # track vehicles
    track_ids = mot_tracker.update(np.asarray(detections_))

    # detect license plates
    license_plates = license_plate_detector(frame)[0]
    for license_plate in license_plates.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = license_plate
        xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

        if car_id != -1:
            license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]
            license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
            _, license_plate_crop_thresh = cv2.threshold(
                license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV
            )
            license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

            if license_plate_text is not None:
                results[frame_nmr][car_id] = {
                    'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                    'license_plate': {
                        'bbox': [x1, y1, x2, y2],
                        'text': license_plate_text,
                        'bbox_score': score,
                        'text_score': license_plate_text_score
                    }
                }

    # resize for display
    display_frame = cv2.resize(frame, (640, 360))
    cv2.imshow("Video", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# write results
write_csv(results, './test.csv')
