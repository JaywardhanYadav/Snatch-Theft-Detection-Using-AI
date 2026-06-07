# Real-Time Chain Snatching Detection System

A modular, scalable, production-grade AI/ML system to detect chain-snatching events in real-time from CCTV videos using Computer Vision and Machine Learning.

## Architecture Layering
The project is built around a robust layered architecture for maximum module separation and logical flow:

1. **Data Layer**: Downloading and Loading videos.
2. **Processing Layer**: Tracking individual movement via YOLO + ByteTrack/DeepSORT.
3. **Feature Layer**: Kinematics (Speed, acceleration, relative distances) are extracted via `MotionAnalyzer`.
4. **Model Layer**: Random Forest structured machine learning binary classification.
5. **Inference Pipeline**: Orchestrates ingestion through visualization without clutter.
6. **Reporting Layer**: Automated plotting and textual reporting, possibly with GenAI integrations.

## System Workflow Pipeline
Video Source ➔ `YOLODetector` / `Tracker` ➔ `MotionAnalyzer` (feature extraction) ➔ `SnatchClassifier` (predicts 0/1) ➔ `Visualizer` & `ReportGenerator`.

## 📁 Directory Structure
```
project_root/
│
├── data/
│   ├── raw/                  # Downloaded input mp4s
│   ├── processed/            # Intermediates
│   └── annotations/          
│
├── notebooks/
│   ├── eda.ipynb             
│   ├── model_experiments.ipynb
│
├── src/
│   ├── config/               # config.yaml (Centralized system hyperparams)
│   ├── data/                 # Video Loaders
│   ├── models/
│   │   ├── detection/        # YOLO Wrappers
│   │   ├── tracking/         # BoT-SORT / Tracker abstractions
│   │   ├── motion/           # Feature Extraction (velocity, proximity, flow)
│   │   ├── classifier/       # ML Scikit Classifier
│   ├── pipeline/             # inference_pipeline.py uniting steps
│   ├── visualization/        # Matplotlib graphs
│   ├── reporting/            # Text summary generator
│   └── utils/                # BBox rendering, output snapshotting
│
├── output/                   # /graphs, /snapshots, report.txt
```

## Quick Start
1. Generate / Configure Environment:
   ```bash
   pip install -r requirements.txt
   ```
2. Run Execution:
   ```bash
   python main.py
   ```
3. Check outputs inside the `output/` folder!
