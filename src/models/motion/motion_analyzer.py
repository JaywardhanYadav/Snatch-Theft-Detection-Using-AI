import cv2
import numpy as np
import math

class MotionAnalyzer:
    def __init__(self, config=None):
        self.prev_gray = None
        self.history = {} # Store history of centers for each ID to calculate speed/direction

        if config is None:
            config = {
                'pyr_scale': 0.5, 'levels': 3, 'winsize': 15,
                'iterations': 3, 'poly_n': 5, 'poly_sigma': 1.2
            }
        self.cfg = config

    def extract_features(self, current_frame, tracked_persons):
        """
        Calculates combination of Optical Flow (background/global motion) and 
        kinematic features for tracking points (individual speed, distance).
        Returns a dictionary of features for the current frame.
        """
        # 1. Global Motion Intensity (Optical Flow)
        gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        global_motion = 0.0
        
        if self.prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray, gray, None, 
                pyr_scale=self.cfg.get('pyr_scale', 0.5), 
                levels=self.cfg.get('levels', 3), 
                winsize=self.cfg.get('winsize', 15), 
                iterations=self.cfg.get('iterations', 3), 
                poly_n=self.cfg.get('poly_n', 5), 
                poly_sigma=self.cfg.get('poly_sigma', 1.2), flags=0
            )
            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            threshold = np.percentile(magnitude, 95)
            top_motion = magnitude[magnitude >= threshold]
            if len(top_motion) > 0:
                global_motion = float(np.mean(top_motion))
                
        self.prev_gray = gray
        
        # 2. Distance and Speed Features
        min_distance = 9999.0
        max_speed = 0.0
        fastest_id = -1

        current_centers = {}

        for p in tracked_persons:
            tid = p['id']
            box = p['box']
            cx = (box[0] + box[2]) / 2.0
            cy = (box[1] + box[3]) / 2.0
            current_centers[tid] = (cx, cy)

            # Calculate speed of this person
            if tid in self.history:
                px, py = self.history[tid]
                speed = math.sqrt((cx - px)**2 + (cy - py)**2)
                if speed > max_speed:
                    max_speed = speed
                    fastest_id = tid

        # Calculate min distance between any two people
        ids = list(current_centers.keys())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                cx1, cy1 = current_centers[ids[i]]
                cx2, cy2 = current_centers[ids[j]]
                dist = math.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
                if dist < min_distance:
                    min_distance = dist

        if len(ids) < 2:
            min_distance = 0.0 # Standardize if < 2 people

        self.history = current_centers

        features = {
            'global_motion': global_motion,
            'min_distance': min_distance,
            'max_speed': max_speed,
            'num_persons': len(tracked_persons),
            'fastest_id': fastest_id
        }
        return features
