![ezgif-7f6b9b4e4c7d57](https://github.com/user-attachments/assets/124b19ae-de76-40ba-a448-d71e75157342)


**Low-Latency Multimodal Perception for Robotic Assembly and Real-Time Inspection of Automotive Wiring Systems using Frame-Based and Event-Based Vision**

A masters research project for **deformable linear object (wire) detection** using two complementary vision sensors:
- **Basler frame camera** (grayscale, high-resolution frames)
- **iniVation DVXplorer event camera** (asynchronous events, high dynamic range)

The system combines **AprilTag-based ROI motion detection**, **homography calibration**, and **real-time wire tracking** with spline fitting.

---

## Features
- **Dual-camera acquisition** (Basler + DVX)
- **AprilTag ROI & motion detection** inside a defined region
- **Homography calibration** (DVX → Basler alignment using ORB features)
- **Real-time overlay** of event data on frame-based images
- **Spline-based wire detection** from masked regions
- **Screenshot saving & ROI cropping** with keyboard shortcuts
- **Interactive UI** with visualization windows

---

**Requirements**
- Python 3.8+
- [Basler Pylon SDK](https://www.baslerweb.com/en/products/software/basler-pylon-camera-software-suite/)  
- [DVXplorer SDK](https://gitlab.com/inivation/dv/dv-processing)  
---

**Core dependencies:**
- opencv-python
- numpy
- scipy
- pypylon (Basler)
- dv-processing (DVXplorer)
- pupil-apriltags
---

**Keyboard Shortcuts (wire detection window)**
- ESC → exit
- p → save overlay screenshot
- m → save cropped ROI
- q/w → adjust low threshold
- a/s → adjust high threshold
- r/e → decrease/increase ROI width
- f/t → decrease/increase ROI height
- i/j/k/l → move ROI rectangle (up/left/down/right)
---

**Example Output**
<img width="1920" height="1080" alt="Screenshot from 2025-09-17 15-59-26" src="https://github.com/user-attachments/assets/962e35b1-b202-4db6-ac96-678bb1082214" />
<img width="1920" height="1080" alt="Screenshot from 2025-09-22 11-39-03" src="https://github.com/user-attachments/assets/de32bea7-a694-46f8-8322-ea059e7f15eb" />
<img width="1920" height="1080" alt="Screenshot from 2025-09-22 11-41-13" src="https://github.com/user-attachments/assets/3c0941d5-61bd-4c93-8563-e85064289359" />
<img width="1920" height="1080" alt="Screenshot from 2025-09-25 14-41-03" src="https://github.com/user-attachments/assets/dbf83941-b129-4cf6-81cd-7d64adb4d20b" />
---

**License**

This project is licensed under the MIT License. See LICENSE for details.

---

**Acknowledgements**
- Basler Pylon SDK
- iniVation DVXplorer SDK
- pupil-apriltags
 -OpenCV community
- [FAU-FAPS](https://github.com/FAU-FAPS)
---

**Citation / Bibtex**
If you use this project in your research, please cite:

```bibtex
@projekt{chaudhry2025lowlatency,
  title = {Low-Latency Multimodal Perception for Robotic Assembly and Real-Time Inspection of Automotive Wiring Systems using Frame-Based and Event-Based Vision},
  author = {Chaudhry, Affyief},
  supervisor = {Hartmann, Annalena},
  institution = {FAU-FAPS},
  year = {2025},
  note = {Project Arbeit},
  url = {https://github.com/Affyief/Project-arbeit---Event-based-vision}
}
```
