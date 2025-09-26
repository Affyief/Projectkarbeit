<img width="1920" height="1080" alt="Screenshot from 2025-09-25 14-41-03" src="https://github.com/user-attachments/assets/dbf83941-b129-4cf6-81cd-7d64adb4d20b" />

**Low-Latency Multimodal Perception for Robotic Assembly and Real-Time Inspection of Automotive Wiring Systems using Frame-Based and Event-Based Vision**

A masters research project for **deformable linear object (wire) detection** using two complementary vision sensors:
- **Basler frame camera** (grayscale, high-resolution frames)
- **iniVation DVXplorer event camera** (asynchronous events, high dynamic range)

The system combines **AprilTag-based motion ROI detection**, **homography calibration**, and **real-time wire tracking** with spline fitting.

---

## Features
- 📷 **Dual-camera acquisition** (Basler + DVX)
- 🏷️ **AprilTag ROI & motion detection** inside a defined region
- 🔄 **Homography calibration** (DVX → Basler alignment using ORB features)
- 🎥 **Real-time overlay** of event data on frame-based images
- 🧵 **Spline-based wire detection** from masked regions
- 💾 **Screenshot saving & ROI cropping** with keyboard shortcuts
- 🖥️ **Interactive UI** with visualization windows

---

## Repository Structure

Projectkarbeit/
│
├── src/ # core modules
│ ├── cameras/ # camera drivers (Basler, DVX)
│ ├── calibration/ # homography estimation
│ ├── detection/ # AprilTag motion & wire detection
│ ├── visualization/ # overlay, labeling, display
│ ├── utils/ # IO, geometry helpers
│ └── main.py # orchestrator (dual-camera pipeline)
│
├── scripts/ # runnable entry points
│ ├── run_dual_camera.py # full pipeline (Basler + DVX + wire detection)
│ ├── april_motion_demo.py # AprilTag + motion ROI
│ └── homography_demo.py # offline homography alignment
│
├── docs/ # documentation
│ ├── INSTALL.md # detailed install instructions
│ ├── USAGE.md # how to run scripts & shortcuts
│ └── ARCHITECTURE.md # system description
│
├── requirements.txt # dependencies
├── LICENSE
└── README.md # (this file)



**Requirements**
- Python 3.8+
- [Basler Pylon SDK](https://www.baslerweb.com/en/products/software/basler-pylon-camera-software-suite/)  
- [DVXplorer SDK](https://inivation.com/support/software/)  


**Core dependencies:**
- opencv-python
- numpy
- scipy
- pypylon (Basler)
- dv-processing (DVXplorer)
- pupil-apriltags

**Keyboard Shortcuts (wire detection window)**
- ESC → exit
- p → save overlay screenshot
- m → save cropped ROI
- q/w → adjust low threshold
- a/s → adjust high threshold
- r/e → decrease/increase ROI width
- f/t → decrease/increase ROI height
- i/j/k/l → move ROI rectangle (up/left/down/right)

**Example Output**
<img width="1920" height="1080" alt="Screenshot from 2025-09-17 15-59-26" src="https://github.com/user-attachments/assets/962e35b1-b202-4db6-ac96-678bb1082214" />
<img width="1920" height="1080" alt="Screenshot from 2025-09-22 11-39-03" src="https://github.com/user-attachments/assets/de32bea7-a694-46f8-8322-ea059e7f15eb" />

<img width="1920" height="1080" alt="Screenshot from 2025-09-25 14-41-03" src="https://github.com/user-attachments/assets/a6ab5144-d4d9-4165-9347-e111a95c5062" />


**License**
This project is licensed under the MIT License. See LICENSE for details.

**Acknowledgements**
- Basler Pylon SDK
- iniVation DVXplorer SDK
- pupil-apriltags
 -OpenCV community
- [FAU-FAPS](https://github.com/FAU-FAPS)
