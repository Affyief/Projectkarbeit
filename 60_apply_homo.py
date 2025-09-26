import numpy as np
import cv2
import threading
import time
from pypylon import pylon
import dv_processing as dv
import os

# ---------- Shared globals ----------
basler_frame = None
dvx_event_frame = None
frame_lock = threading.Lock()

DVX_SERIAL = "DVXplorer_DXAS0024"
screenshot_count = 0

# Resized dimensions (matching homography calculation)
RESIZED_W, RESIZED_H = 940, 540
ALPHA = 0.5  # overlay transparency

# ------------------------
# Homography (from 940x540 resized screenshots)
# ------------------------
H_resized = np.array([
    [ 7.81740620e-01, -7.77607070e-02,  1.59554279e+02],
    [-9.05046835e-04,  9.22111603e-01,  1.51813399e+01],
    [-4.14179057e-05, -1.45666882e-04,  1.00000000e+00]

    
    
])

# ------------------------
# Basler initialization
# ------------------------
def init_basler():
    factory = pylon.TlFactory.GetInstance()
    devices = factory.EnumerateDevices()
    if not devices:
        raise RuntimeError("No Basler camera found")
    cam = pylon.InstantCamera(factory.CreateDevice(devices[0]))
    cam.Open()
    cam.OffsetX.SetValue(0)
    cam.OffsetY.SetValue(0)
    cam.Width.SetValue(cam.Width.GetMax())
    cam.Height.SetValue(cam.Height.GetMax())
    cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    conv = pylon.ImageFormatConverter()
    conv.OutputPixelFormat = pylon.PixelType_Mono8
    return cam, conv

def basler_worker(cam, conv):
    global basler_frame
    while cam.IsGrabbing():
        res = cam.RetrieveResult(1000, pylon.TimeoutHandling_ThrowException)
        if res.GrabSucceeded():
            gray = conv.Convert(res).GetArray()
            with frame_lock:
                basler_frame = gray.copy()
        res.Release()

# ------------------------
# DVX initialization
# ------------------------
def init_dvx():
    cam = dv.io.CameraCapture(DVX_SERIAL)
    res = cam.getEventResolution()
    vis = dv.visualization.EventVisualizer(res)
    vis.setBackgroundColor((0,0,0))
    vis.setPositiveColor((0,255,0))
    vis.setNegativeColor((0,0,255))
    return cam, vis

def dvx_worker(cam, vis):
    global dvx_event_frame
    while True:
        if not cam.isEventStreamAvailable():
            time.sleep(0.005)
            continue
        events = cam.getNextEventBatch()
        if events is None or events.size() == 0:
            continue
        ev_img = vis.generateImage(events)
        if ev_img.ndim == 3:
            ev_img = cv2.cvtColor(ev_img, cv2.COLOR_BGR2GRAY)
        with frame_lock:
            dvx_event_frame = ev_img

# ------------------------
# Overlay function (resized streams)
# ------------------------
def overlay_dvx_on_basler_resized(b_frame, d_frame):
    # Default frames if None
    if b_frame is None:
        b_frame = np.zeros((RESIZED_H, RESIZED_W), dtype=np.uint8)
    if d_frame is None:
        d_frame = np.zeros((RESIZED_H, RESIZED_W), dtype=np.uint8)

    # Resize both streams to 940x540
    b_resized = cv2.resize(b_frame, (RESIZED_W, RESIZED_H))
    d_resized = cv2.resize(d_frame, (RESIZED_W, RESIZED_H))

    # Apply homography to DVX
    warped_dvx = cv2.warpPerspective(d_resized, H_resized, (RESIZED_W, RESIZED_H))

    # Convert to BGR for overlay
    b_color = cv2.cvtColor(b_resized, cv2.COLOR_GRAY2BGR)
    d_color = np.zeros((RESIZED_H, RESIZED_W, 3), dtype=np.uint8)    #
    d_color[:, :, 2] = warped_dvx  # assign grayscale to red channel #


    # Overlay
    overlay = cv2.addWeighted(b_color, 1-ALPHA, d_color, ALPHA, 0)
    return overlay

# ------------------------
# Screenshot helper
# ------------------------
def save_screenshot(img):
    global screenshot_count
    os.makedirs("screenshots", exist_ok=True)
    path = f"screenshots/overlay_{screenshot_count:03d}.png"
    cv2.imwrite(path, img)
    print(f"[INFO] Saved screenshot {screenshot_count:03d}")
    screenshot_count += 1

# ------------------------
# Main
# ------------------------
def main():
    global basler_frame, dvx_event_frame
    bas_cam, bas_conv = init_basler()
    dvx_cam, dvx_vis = init_dvx()

    threading.Thread(target=basler_worker, args=(bas_cam, bas_conv), daemon=True).start()
    threading.Thread(target=dvx_worker, args=(dvx_cam, dvx_vis), daemon=True).start()

    print("Press ESC to quit. Press 's' to save screenshot.")

    while True:
        with frame_lock:
            b_frame = basler_frame.copy() if basler_frame is not None else None
            d_frame = dvx_event_frame.copy() if dvx_event_frame is not None else None

        # Overlay DVX on Basler (both resized to 940x540)
        overlay = overlay_dvx_on_basler_resized(b_frame, d_frame)

        # Display
        cv2.imshow("DVX over Basler (resized)", overlay)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('s'):
            save_screenshot(overlay)

    bas_cam.StopGrabbing()
    bas_cam.Close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


'''[ 8.94508654e-01, -8.54891897e-02,  2.71994407e+02],
    [ 2.97664754e-02,  9.66133796e-01,  1.43606263e+01],
    [ 6.52469578e-05, -1.09108111e-04,  1.00000000e+00]'''
