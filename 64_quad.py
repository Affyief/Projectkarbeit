import numpy as np
import cv2
import threading
import time
from pypylon import pylon
import dv_processing as dv
import os
from scipy.interpolate import splprep, splev

# ---------- Shared globals ----------
basler_frame = None
dvx_event_frame = None
frame_lock = threading.Lock()

DVX_SERIAL = "DVXplorer_DXAS0024"
screenshot_count = 0
last_screenshot_msg = ""  # message for legend

RESIZED_W, RESIZED_H = 940, 540
ALPHA = 0.5  # overlay transparency

# Homography for overlay
H_resized = np.array([
    [ 7.81740620e-01, -7.77607070e-02,  1.59554279e+02],
    [-9.05046835e-04,  9.22111603e-01,  1.51813399e+01],
    [-4.14179057e-05, -1.45666882e-04,  1.00000000e+00]
])

# ---------------- Basler ----------------
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

# ---------------- DVX ----------------
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

# ---------------- Overlay ----------------
def overlay_dvx_on_basler_resized(b_frame, d_frame):
    if b_frame is None:
        b_frame = np.zeros((RESIZED_H, RESIZED_W), dtype=np.uint8)
    if d_frame is None:
        d_frame = np.zeros((RESIZED_H, RESIZED_W), dtype=np.uint8)
    b_resized = cv2.resize(b_frame, (RESIZED_W, RESIZED_H))
    d_resized = cv2.resize(d_frame, (RESIZED_W, RESIZED_H))
    warped_dvx = cv2.warpPerspective(d_resized, H_resized, (RESIZED_W, RESIZED_H))
    b_color = cv2.cvtColor(b_resized, cv2.COLOR_GRAY2BGR)
    d_color = np.zeros((RESIZED_H, RESIZED_W, 3), dtype=np.uint8)
    d_color[:, :, 2] = warped_dvx
    overlay = cv2.addWeighted(b_color, 1-ALPHA, d_color, ALPHA, 0)
    return overlay

# ---------------- Mask and Wire ----------------
def detect_wire(masked_gray):
    blurred = cv2.GaussianBlur(masked_gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = [c for c in contours if cv2.contourArea(c) > 500]
    if not contours:
        return None, None, None
    wire = max(contours, key=cv2.contourArea).squeeze()
    if wire.ndim != 2 or len(wire) < 10:
        return None, None, None
    try:
        tck, _ = splprep(wire.T, s=0.001 * len(wire))
        u_new = np.linspace(0, 1, 1000)
        x_new, y_new = splev(u_new, tck)
        if np.any(np.isnan(x_new)) or np.any(np.isnan(y_new)):
            return None, None, None
        return wire, x_new, y_new
    except:
        return None, None, None

def draw_wire(img, wire_pts, x_spline, y_spline):
    if wire_pts is not None:
        for pt in wire_pts:
            cv2.circle(img, tuple(pt), 2, (0, 0, 255), -1)
    if x_spline is not None and y_spline is not None:
        for i in range(len(x_spline)-1):
            pt1 = (int(x_spline[i]), int(y_spline[i]))
            pt2 = (int(x_spline[i+1]), int(y_spline[i+1]))
            cv2.line(img, pt1, pt2, (255, 0, 0), 2)
    return img

def intensity_mask_rectangle(frame, low, high, rect_w, rect_h, rect_cx, rect_cy, msg=""):
    if frame is None:
        return np.zeros((RESIZED_H, RESIZED_W, 3), dtype=np.uint8)
    gray = cv2.resize(frame, (RESIZED_W, RESIZED_H))
    mask = np.zeros((RESIZED_H, RESIZED_W), dtype=np.uint8)
    x1 = max(rect_cx - rect_w//2, 0)
    y1 = max(rect_cy - rect_h//2, 0)
    x2 = min(rect_cx + rect_w//2, RESIZED_W)
    y2 = min(rect_cy + rect_h//2, RESIZED_H)
    roi = gray[y1:y2, x1:x2]
    mask_roi = np.where(roi <= high, 0, 255).astype(np.uint8)
    mask[y1:y2, x1:x2] = mask_roi
    mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(mask_color, (x1,y1), (x2,y2), (0,0,255), 2)

    # Wire detection
    wire_pts, x_spline, y_spline = detect_wire(mask[y1:y2, x1:x2])
    # Shift spline and points to global coords
    if wire_pts is not None:
        wire_pts_global = wire_pts + np.array([x1, y1])
        x_spline_global = x_spline + x1
        y_spline_global = y_spline + y1
        mask_color = draw_wire(mask_color, wire_pts_global, x_spline_global, y_spline_global)

    # Legend
    legend_lines = [
        "Low threshold: q/w",
        "High threshold: a/s",
        "Width: r/e",
        "Height: f/t",
        "Move: i/j/k/l",
        msg
    ]
    for i, line in enumerate(legend_lines):
        cv2.putText(mask_color, line, (10, 80 + i*25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    return mask_color

# ---------------- Screenshot ----------------
def save_screenshot(img, name_prefix="overlay"):
    global screenshot_count, last_screenshot_msg
    os.makedirs("screenshots", exist_ok=True)
    path = f"screenshots/{name_prefix}_{screenshot_count:03d}.png"
    cv2.imwrite(path, img)
    last_screenshot_msg = f"Saved screenshot {name_prefix}_{screenshot_count:03d}.png"
    print(f"[INFO] {last_screenshot_msg}")
    screenshot_count += 1

# ---------------- Main ----------------
def main():
    global basler_frame, dvx_event_frame, last_screenshot_msg
    bas_cam, bas_conv = init_basler()
    dvx_cam, dvx_vis = init_dvx()
    threading.Thread(target=basler_worker, args=(bas_cam, bas_conv), daemon=True).start()
    threading.Thread(target=dvx_worker, args=(dvx_cam, dvx_vis), daemon=True).start()

    mask_low, mask_high = 0, 100
    rect_w, rect_h = 500, 300
    rect_min, rect_max = 50, min(RESIZED_W, RESIZED_H)
    rect_cx, rect_cy = RESIZED_W//2, RESIZED_H//2

    while True:
        with frame_lock:
            b_frame = basler_frame.copy() if basler_frame is not None else None
            d_frame = dvx_event_frame.copy() if dvx_event_frame is not None else None

        b_resized = cv2.resize(b_frame, (RESIZED_W, RESIZED_H)) if b_frame is not None else np.zeros((RESIZED_H, RESIZED_W), dtype=np.uint8)
        d_resized = cv2.resize(d_frame, (RESIZED_W, RESIZED_H)) if d_frame is not None else np.zeros((RESIZED_H, RESIZED_W), dtype=np.uint8)

        overlay = overlay_dvx_on_basler_resized(b_resized, d_resized)
        overlay_brighter = cv2.convertScaleAbs(overlay, alpha=1.5, beta=30)

        b_color = cv2.cvtColor(b_resized, cv2.COLOR_GRAY2BGR)
        d_color = cv2.cvtColor(d_resized, cv2.COLOR_GRAY2BGR)

        window4_mask = intensity_mask_rectangle(b_frame, mask_low, mask_high,
                                                rect_w, rect_h, rect_cx, rect_cy,
                                                last_screenshot_msg)

        top_row = np.hstack([b_color, d_color])
        bottom_row = np.hstack([overlay_brighter, window4_mask])
        master = np.vstack([top_row, bottom_row])

        cv2.putText(master, "Basler Only", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
        cv2.putText(master, "DVX Only", (RESIZED_W+10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
        cv2.putText(master, "Overlay", (10,RESIZED_H+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
        cv2.putText(master, f"Mask {mask_low}-{mask_high}", (RESIZED_W+10,RESIZED_H+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

        cv2.imshow("Master Window", master)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('p'):
            save_screenshot(master, "overlay")
        elif key == ord('m'):
            # Crop rectangle only
            x1 = max(rect_cx - rect_w//2, 0)
            y1 = max(rect_cy - rect_h//2, 0)
            x2 = min(rect_cx + rect_w//2, RESIZED_W)
            y2 = min(rect_cy + rect_h//2, RESIZED_H)
            if window4_mask is not None:
                cropped = window4_mask[y1:y2, x1:x2].copy()
                save_screenshot(cropped, "rect")

        # Threshold adjustments
        elif key == ord('q'): mask_low = max(0, mask_low-1)
        elif key == ord('w'): mask_low = min(mask_high, mask_low+1)
        elif key == ord('a'): mask_high = max(mask_low, mask_high-1)
        elif key == ord('s'): mask_high = min(255, mask_high+1)
        # Width/height adjustments
        elif key == ord('r'): rect_w = max(rect_min, rect_w-10)
        elif key == ord('e'): rect_w = min(rect_max, rect_w+10)
        elif key == ord('f'): rect_h = max(rect_min, rect_h-10)
        elif key == ord('t'): rect_h = min(rect_max, rect_h+10)
        # Movement
        elif key == ord('i'): rect_cy = max(0, rect_cy-10)
        elif key == ord('k'): rect_cy = min(RESIZED_H, rect_cy+10)
        elif key == ord('j'): rect_cx = max(0, rect_cx-10)
        elif key == ord('l'): rect_cx = min(RESIZED_W, rect_cx+10)

    bas_cam.StopGrabbing()
    bas_cam.Close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

