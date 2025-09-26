# Scripts Folder

This folder contains runnable scripts for wire detection and analysis.

## run_wire_detection.py

Runs the main wire detection and template matching pipeline.
Usage:
```sh
python run_wire_detection.py
```

## run_wire_detection_intersections.py

Runs the wire detection pipeline **with intersection detection**.
Usage:
```sh
python run_wire_detection_intersections.py
```

## calibrate_cameras.py

Calibrates cameras and saves calibration/homography to config.
Usage:
```sh
python calibrate_cameras.py
```

## evaluate_results.py

Evaluates detection results using saved screenshots and templates.
Usage:
```sh
python evaluate_results.py
```
