import cv2
import pandas as pd
import os
from src.models.tracking.tracker import Tracker
from src.models.motion.motion_analyzer import MotionAnalyzer
from src.models.classifier.snatch_classifier import SnatchClassifier
from src.utils.helpers import save_snapshot, draw_annotations

class InferencePipeline:
    def __init__(self, config):
        self.config = config
        self.tracker = Tracker(config['detection']['model_weights'])
        self.motion_analyzer = MotionAnalyzer(config['motion'])
        self.classifier = SnatchClassifier(config['classifier']['model_path'])
        
        # Load or Dummy train on init
        self.classifier.load_or_train_dummy()

    def process_video(self, video_path):
        video_name = os.path.basename(video_path).split('.')[0]
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ERROR] Could not open {video_path}")
            return None

        metrics = []
        unique_ids = set()
        alert_frames = []
        frames_count = 0
        
        # Reset analyzers for new video stream
        self.motion_analyzer.prev_gray = None
        self.motion_analyzer.history = {}
        
        print(f"\n[INFO] Running Inference on {video_name}...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frames_count += 1
            
            # Step 1: Tracking
            tracked_persons = self.tracker.track_frame(frame, conf=self.config['detection']['conf_threshold'])
            
            # Step 2: Feature Extraction (Motion & Kinematics)
            features = self.motion_analyzer.extract_features(frame, tracked_persons)
            
            # Step 3: ML Inference (Only run if multiple people exist or based on logic)
            is_snatch = False
            snatch_prob = 0.0
            
            # For this ML implementation, let's predict strictly based on features
            if len(tracked_persons) >= 1:
                pred = self.classifier.predict(features)
                is_snatch = bool(pred == 1)
                if hasattr(self.classifier.model, "predict_proba"):
                    import numpy as np
                    X = np.array([[features['global_motion'], features['min_distance'], features['max_speed'], features['num_persons']]])
                    probs = self.classifier.model.predict_proba(X)
                    snatch_prob = probs[0][1] # Probability of class 1

            # Step 4: Annotation and Snapshot Trigger
            current_ids = []
            thief_id = features.get('fastest_id', -1)
            annotated_frame = draw_annotations(frame, tracked_persons, is_snatch, is_snatch, snatch_prob, current_ids, unique_ids, thief_id=thief_id)
            
            if is_snatch:
                # Rate limit snapshots
                if len(alert_frames) < self.config['reporting']['max_snapshots']:
                    if len(alert_frames) == 0 or (frames_count - alert_frames[-1] > 30):
                        save_snapshot(annotated_frame, frames_count, video_name, alert=True)
                        alert_frames.append(frames_count)

            # Store metrics
            avg_conf = sum([p['conf'] for p in tracked_persons]) / len(tracked_persons) if len(tracked_persons) > 0 else 0
            
            metrics.append({
                'frame': frames_count,
                'num_persons': len(tracked_persons),
                'motion_intensity': features['global_motion'],
                'min_distance': features['min_distance'],
                'avg_confidence': avg_conf,
                'alert_triggered': is_snatch,
                'track_ids': current_ids
            })

        cap.release()
        print(f"[INFO] Finished {video_name}. Frames processed: {frames_count}")
        
        return {
            'video_name': video_name,
            'metrics_df': pd.DataFrame(metrics),
            'frames_count': frames_count,
            'unique_persons': len(unique_ids),
            'alert_frames': alert_frames,
            'alerts': len(alert_frames)
        }
