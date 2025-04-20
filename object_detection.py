import cv2
from ultralytics import YOLO

# YOLO-Modell laden
model = YOLO("yolo-Weights/yolov8n.pt")

# Klassenliste (YOLOv8 COCO)
classNames = list(model.names.values())

# --- Detection ---
def detect_obj(results, img):
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = round(float(box.conf[0]), 2)
            cls = int(box.cls[0])
            label = classNames[cls] if cls < len(classNames) else "Unknown"

            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.putText(img, f'{label} {confidence}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)