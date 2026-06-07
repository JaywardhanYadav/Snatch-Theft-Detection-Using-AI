import matplotlib.pyplot as plt
import os
import matplotlib

# Use Agg backend for non-interactive plotting
matplotlib.use('Agg')

class Visualizer:
    def __init__(self, output_dir="output/graphs"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_graphs(self, df, video_name):
        """
        Generates 5 distinct graphs from the metrics DataFrame.
        """
        if df.empty:
            print("[WARN] DataFrame is empty. Skipping graphs generation.")
            return

        frames = df['frame']
        
        # 1. Persons per frame
        plt.figure(figsize=(10, 4))
        plt.plot(frames, df['num_persons'], color='blue', label='Persons')
        plt.title('1. Number of Persons per Frame')
        plt.xlabel('Frame')
        plt.ylabel('Count')
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, f"{video_name}_1_persons.png"))
        plt.close()

        # 2. Motion Intensity
        plt.figure(figsize=(10, 4))
        plt.plot(frames, df['motion_intensity'], color='red', label='Motion Intensity')
        plt.title('2. Global Motion Intensity')
        plt.xlabel('Frame')
        plt.ylabel('Intensity (Optical Flow)')
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, f"{video_name}_2_motion.png"))
        plt.close()

        # 3. Detection confidence
        plt.figure(figsize=(10, 4))
        plt.plot(frames, df['avg_confidence'], color='green', label='Avg Confidence')
        plt.title('3. YOLOv8 Average Detection Confidence')
        plt.xlabel('Frame')
        plt.ylabel('Confidence')
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, f"{video_name}_3_confidence.png"))
        plt.close()

        # 4. Interaction Duration / Min Distance
        plt.figure(figsize=(10, 4))
        plt.plot(frames, df['min_distance'], color='purple', label='Minimum Distance')
        plt.title('4. Interaction Proximity (Min Distance between persons)')
        plt.xlabel('Frame')
        plt.ylabel('Pixels')
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, f"{video_name}_4_distance.png"))
        plt.close()

        # 5. Alert Timeline
        plt.figure(figsize=(10, 4))
        alerts = df[df['alert_triggered'] == True]
        plt.scatter(alerts['frame'], [1] * len(alerts), color='red', marker='x', s=100)
        plt.title('5. Snatching Alert Timeline')
        plt.xlabel('Frame')
        plt.yticks([])
        plt.ylim(0, 2)
        plt.savefig(os.path.join(self.output_dir, f"{video_name}_5_alerts.png"))
        plt.close()

        print(f"[INFO] Graphical plots saved to {self.output_dir} for {video_name}")
