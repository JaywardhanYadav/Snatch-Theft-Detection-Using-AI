from ultralytics import YOLO

class YOLODetector:
    def __init__(self, model_path="yolov8n.pt", conf=0.3):
        self.model = YOLO(model_path)
        self.conf = conf

    def detect(self, frame):
        """Standard detection without tracking, if needed."""
        results = self.model(frame, classes=[0], conf=self.conf, verbose=False)
        return results
