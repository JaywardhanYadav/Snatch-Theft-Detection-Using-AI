import os
import datetime

class ReportGenerator:
    def __init__(self, output_dir="output"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def generate(self, summary_data):
        report_path = os.path.join(self.output_dir, "report.txt")
        
        with open(report_path, "w") as f:
            f.write("==================================================\n")
            f.write("      CHAIN SNATCHING DETECTION - FINAL REPORT    \n")
            f.write("==================================================\n")
            f.write(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            total_alerts = 0
            for vname, stats in summary_data.items():
                f.write(f"--- Video: {vname} ---\n")
                f.write(f"  Frames processed: {stats['frames']}\n")
                f.write(f"  Unique persons detected: {stats['unique_persons']}\n")
                f.write(f"  Alerts triggered: {stats['alerts']}\n")
                if stats['alerts'] > 0:
                    f.write(f"  Alert Frames: {stats['alert_frames']}\n")
                    
                    df = stats.get('df')
                    if df is not None:
                        f.write("\n  [Narrative Timeline]\n")
                        # Basic timeline: Start -> First Alert -> Max Speed event -> End
                        first_alert_frame = df[df['alert_triggered'] == True]['frame'].min()
                        max_motion_row = df.loc[df['motion_intensity'].idxmax()]
                        max_motion_frame = max_motion_row['frame']
                        
                        f.write(f"  - Frame 0: Scene begins, tracking {stats['unique_persons']} total unique individuals over time.\n")
                        f.write(f"  - Frame {int(first_alert_frame)}: Interaction detected. Persons come into close proximity and system triggers snatching alert.\n")
                        f.write(f"  - Frame {int(max_motion_frame)}: Sudden spike in velocity detected (Motion Intensity: {max_motion_row['motion_intensity']:.2f}). Identified thief flees the scene.\n")
                        f.write(f"  - Frame {stats['frames']}: Tracking module loses track of moving subjects or video ends.\n")
                f.write("\n")
                total_alerts += stats['alerts']
                
            f.write("==================================================\n")
            f.write(f"TOTAL SYSTEM ALERTS: {total_alerts}\n")
            f.write("==================================================\n")
            
            # Optional GenAI step
            api_key = os.environ.get("GEMINI_API_KEY", "")
            if api_key:
                try:
                    import google.generativeai as genai
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel('gemini-pro')
                    prompt = f"Write a 3 sentence professional security summary for a system that processed CCTV footage and found {total_alerts} chain snatching alerts across {len(summary_data)} videos."
                    response = model.generate_content(prompt)
                    f.write("\n[Generative AI Summary]\n")
                    f.write(response.text + "\n")
                except Exception as e:
                    f.write("\n[Generative AI Summary Failed]\n")
                    f.write(str(e) + "\n")
            else:
                f.write("\n[Generative AI Summary Skipped: No API Key provided]\n")

        print(f"[INFO] Final report generated at {report_path}")
