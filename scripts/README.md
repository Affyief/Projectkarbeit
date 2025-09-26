# Scripts Folder

This folder contains runnable scripts for wire detection and analysis.

## run_wire_detection.py

Runs the main wire detection and template matching pipeline for checking the correct/incorrect layout against the saved master layout.
Usage:
```sh
python run_wire_detection.py
```

## run_wire_detection_intersections.py

Runs the existing wire detection pipeline **with additional intersection detection**.
Note: Is computationally heavy for a non-GPU /CPU only system.
Usage:
```sh
python run_wire_detection_intersections.py
```

## calibrate_cameras.py

Calibrates cameras and saves calibration/homography matrix to config.
Usage:
```sh
python calibrate_cameras.py
```
