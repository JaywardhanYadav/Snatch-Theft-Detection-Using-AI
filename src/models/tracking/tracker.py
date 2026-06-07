from ultralytics import YOLO

class Tracker:
    def __init__(self, model_path="yolov8n.pt"):
        # Load YOLOv8 Nano model for integrated tracking
        self.model = YOLO(model_path)

    def track_frame(self, frame, conf=0.3):
        """
        Runs YOLOv8 tracking on a given frame.
        Filter classes to only 'person' (class 0).
        """
        results = self.model.track(frame, persist=True, classes=[0], conf=conf, verbose=False)
        
        detections = []
        if len(results) > 0 and results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            confidences = results[0].boxes.conf.cpu().numpy()
            
            for box, track_id, c in zip(boxes, track_ids, confidences):
                x1, y1, x2, y2 = map(int, box)
                detections.append({
                    "id": track_id,
                    "box": (x1, y1, x2, y2),
                    "conf": c
                })
        return detections
