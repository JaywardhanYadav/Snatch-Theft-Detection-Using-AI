import joblib
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

class SnatchClassifier:
    def __init__(self, model_path="models/snatch_rf.pkl"):
        self.model_path = model_path
        self.model = None

    def load_or_train_dummy(self):
        """
        Loads the model if it exists, otherwise generates dummy synthetic data 
        resembling snatching interactions and trains a RandomForest on it.
        """
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            print(f"[INFO] Loaded ML classifier from {self.model_path}")
        else:
            print("[WARN] Model not found. Training a dummy RandomForest classifier...")
            self._train_dummy_model()

    def _train_dummy_model(self):
        # Generate synthetic data
        # Features: global_motion, min_distance, max_speed, num_persons
        np.random.seed(42)
        
        # Normal behavior: Low motion, distance varies, speed low
        normal_motion = np.random.uniform(0.1, 2.0, 500)
        normal_dist = np.random.uniform(50, 500, 500)
        normal_speed = np.random.uniform(0.1, 5.0, 500)
        normal_n = np.random.randint(1, 5, 500)
        normal_labels = np.zeros(500)

        # Snatching behavior: High motion, low distance, high speed
        snatch_motion = np.random.uniform(2.5, 10.0, 100)
        snatch_dist = np.random.uniform(0, 150, 100)
        snatch_speed = np.random.uniform(8.0, 30.0, 100)
        snatch_n = np.random.randint(2, 4, 100) # Usually 2 or 3 people
        snatch_labels = np.ones(100)

        X = np.column_stack((
            np.concatenate([normal_motion, snatch_motion]),
            np.concatenate([normal_dist, snatch_dist]),
            np.concatenate([normal_speed, snatch_speed]),
            np.concatenate([normal_n, snatch_n])
        ))
        y = np.concatenate([normal_labels, snatch_labels])

        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        clf.fit(X, y)
        
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(clf, self.model_path)
        self.model = clf
        print(f"[INFO] Dummy model trained and saved to {self.model_path}")

    def predict(self, features):
        """
        Given a dictionary of features, predict 0 (Normal) or 1 (Snatch).
        """
        if self.model is None:
            self.load_or_train_dummy()
            
        # Features must be in correct order: global_motion, min_distance, max_speed, num_persons
        X = np.array([[
            features.get('global_motion', 0.0),
            features.get('min_distance', 0.0),
            features.get('max_speed', 0.0),
            features.get('num_persons', 0)
        ]])
        
        pred = self.model.predict(X)
        return int(pred[0])
