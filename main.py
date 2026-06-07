import yaml
from src.data.loader import VideoLoader
from src.pipeline.inference_pipeline import InferencePipeline
from src.visualization.plots import Visualizer
from src.reporting.report_generator import ReportGenerator

def load_config(path="src/config/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    print("==================================================")
    print("   Real-Time Chain Snatching Detection System   ")
    print("==================================================")

    # 1. Load Configurations
    config = load_config()

    # 2. Data Preparation
    loader = VideoLoader(config['pipeline']['videos_dir'])
    loader.download_sample_videos()
    video_paths = loader.get_video_paths()

    if not video_paths:
        print("[ERROR] No videos found in the raw data directory.")
        return

    # 3. Initialize Pipeline and Output Modules
    pipeline = InferencePipeline(config)
    visualizer = Visualizer(output_dir=f"{config['pipeline']['output_dir']}/graphs")
    report_gen = ReportGenerator(output_dir=config['pipeline']['output_dir'])

    summary_data = {}

    # Limit execution to only 'alternate_snatch' video if it implies testing specific interactions
    video_paths = [vp for vp in video_paths if "alternate_snatch" in vp.lower()]
    if not video_paths:
        print("[ERROR] alternate_snatch video not found.")
        return

    # 4. Process all videos
    for vp in video_paths:
        result = pipeline.process_video(vp)
        if result:
            v_name = result['video_name']
            # Generate visual outputs
            visualizer.generate_graphs(result['metrics_df'], v_name)
            
            # Store summary
            summary_data[v_name] = {
                'frames': result['frames_count'],
                'unique_persons': result['unique_persons'],
                'alerts': result['alerts'],
                'alert_frames': result['alert_frames'],
                'df': result['metrics_df']
            }

    # 5. Generate Final Report
    report_gen.generate(summary_data)

    print("\n[SUCCESS] Pipeline Execution Complete!")
    print("Check output/graphs, output/snapshots, and output/report.txt")

if __name__ == "__main__":
    main()
