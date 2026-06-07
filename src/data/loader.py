import os
import urllib.request

class VideoLoader:
    def __init__(self, raw_dir="data/raw"):
        self.raw_dir = raw_dir
        os.makedirs(self.raw_dir, exist_ok=True)

    def download_sample_videos(self):
        """Download sample videos if the raw directory is empty."""
        samples = {
            "pedestrians_1.mp4": "https://www.w3schools.com/html/mov_bbb.mp4",
            "pedestrians_2.mp4": "https://www.w3schools.com/html/mov_bbb.mp4",
            "pedestrians_3.mp4": "https://www.w3schools.com/html/mov_bbb.mp4"
        }
        
        existing_files = [f for f in os.listdir(self.raw_dir) if f.endswith('.mp4')]
        if len(existing_files) < 3:
            print("[INFO] Downloading sample videos (Bunny proxy for pedestrians)...")
            for name, url in samples.items():
                target_path = os.path.join(self.raw_dir, name)
                if not os.path.exists(target_path):
                    try:
                        urllib.request.urlretrieve(url, target_path)
                        print(f"  -> Downloaded {name}")
                    except Exception as e:
                        print(f"  -> Failed to download {name}: {e}")

    def get_video_paths(self):
        return [os.path.join(self.raw_dir, f) for f in os.listdir(self.raw_dir) if f.endswith(('.mp4', '.avi'))]
