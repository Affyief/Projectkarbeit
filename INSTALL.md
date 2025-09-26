# Installation Guide

This project requires **Python 3.8+** and access to two hardware cameras:

- **Basler area-scan camera** (via Pylon SDK)
- **iniVation DVXplorer event camera**

---

## 1. System Requirements
- Linux (recommended) or Windows
- Python 3.8 or higher
- Basler Pylon SDK installed
- DVXplorer SDK / dv-processing library installed
- OpenCV-compatible GPU optional, but not required

---

## 2. Install Camera Drivers

### Basler
Download and install the **Basler Pylon SDK**:  
👉 [https://www.baslerweb.com/en/products/software/basler-pylon-camera-software-suite/](https://www.baslerweb.com/en/products/software/basler-pylon-camera-software-suite/)

Make sure the `pypylon` Python bindings are available:
```bash
pip install pypylon
