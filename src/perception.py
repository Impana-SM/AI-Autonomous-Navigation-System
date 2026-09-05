import cv2
import numpy as np
from ultralytics import YOLO

class PerceptionEngine:
    def __init__(self, model_path: str = 'yolov8n.pt'):
        self.model = YOLO(model_path)
        
    def detect_obstacles(self, frame: np.ndarray):
        results = self.model(frame, verbose=False)[0]
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            if conf > 0.3:
                detections.append({'bbox': (x1, y1, x2, y2), 'confidence': conf, 'class_id': cls_id})
        return detections

    def generate_occupancy_grid(self, frame_shape, detections, grid_size=(50, 50)):
        grid = np.zeros(grid_size, dtype=np.int32)
        h, w, _ = frame_shape
        cell_h = h / grid_size[0]
        cell_w = w / grid_size[1]
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            g_x1 = int(np.clip(x1 / cell_w, 0, grid_size[1] - 1))
            g_x2 = int(np.clip(x2 / cell_w, 0, grid_size[1] - 1))
            g_y1 = int(np.clip(y1 / cell_h, 0, grid_size[0] - 1))
            g_y2 = int(np.clip(y2 / cell_h, 0, grid_size[0] - 1))
            grid[max(0, g_y1-1):min(grid_size[0], g_y2+2), max(0, g_x1-1):min(grid_size[1], g_x2+2)] = 1
        return grid
