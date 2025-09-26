"""
Camera calibration utilities for Basler and DVXplorer.
"""

import cv2
import numpy as np
from pypylon import pylon

def calibrate_basler_camera(index=0, save_path=None):
    """
    Example stub for calibrating a Basler camera.
    Replace with actual calibration logic as needed.
    """
    cam, conv = None, None
    try:
        from .camera_manager import init_basler
        cam, conv = init_basler(index)
    except Exception as e:
        print(f"Calibration failed: {e}")
        return None

    # Example: grab a calibration frame and save it
    if cam.IsGrabbing():
        res = cam.RetrieveResult(1000, pylon.TimeoutHandling_ThrowException)
        if res.GrabSucceeded():
            frame = conv.Convert(res).GetArray()
            if save_path:
                cv2.imwrite(save_path, frame)
            print("Calibration frame grabbed and saved.")
        res.Release()
    cam.StopGrabbing()
    cam.Close()
    return True

def calibrate_dvxplorer(serial="DVXplorer_DXAS0024", save_path=None):
    """
    Example stub for calibrating a DVXplorer camera.
    Replace with actual calibration logic as needed.
    """
    from dv_processing.io import CameraCapture
    camera = CameraCapture(serial)
    events = None
    if camera.isEventStreamAvailable():
        events = camera.getNextEventBatch()
        if events is not None and events.size() > 0:
            res = camera.getEventResolution()
            image = np.zeros((res[1], res[0]), dtype=np.uint8)  # Placeholder
            if save_path:
                cv2.imwrite(save_path, image)
            print("DVXplorer calibration image saved.")
    return True
