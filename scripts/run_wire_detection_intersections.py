"""
Basler + DVX Wire Detection Overlay with additional intersection detection

This script captures frames from a Basler camera and a DVXplorer event camera,
detects wire contours in a defined region, overlays DVX events onto the Basler
image using a homography, computes spline-based wire intersections, compares
to a saved wire template, and displays a 2x2 master window with all views.

Features:
- Live Basler and DVX display
- Overlay of DVX events on Basler frame
- Wire detection with spline interpolation
- Intersection highlighting
- Wire template matching and similarity calculation
- Adjustable mask rectangle for intensity-based wire detection
- Screenshot saving of overlay, mask, and cropped wire region
- Interactive keyboard controls for thresholds, rectangle size, and movement

Keyboard Controls:
- q/w : Decrease/Increase low threshold
- a/s : Decrease/Increase high threshold
- r/e : Decrease/Increase rectangle width
- f/t : Decrease/Increase rectangle height
- i/j/k/l : Move rectangle up/left/down/right
- m : Save rectangle crop as screenshot
- g : Save wire template
- p : Save full overlay screenshot
- ESC : Exit

Author: Affyief
Date: 2025-09-11
"""

import numpy as np
import cv2
import threading
import time
from pypylon import pylon
import dv_processing as dv
import os
from scipy.interpolate import splprep, splev
from scipy.spatial.distance import directed_hausdorff

# ---------- Shared globals ----------
basler_frame = None  # Latest frame from Basler camera
dvx_event_frame = None  # Latest frame from DVX event camera
frame_lock = threading.Lock()  # Thread-safe access to frames

DVX_SERIAL = "DVXplorer_DXAS0024"  # DVX camera serial number
screenshot_count = 0  # Counter for saved screenshots
last_screenshot_msg = ""  # Message to display in legend
similarity_msg = ""  # Shape similarity message

RESIZED_W, RESIZED_H = 940, 540  # Resize dimensions for display
ALPHA = 0.5  # Transparency factor for overlay

# Homography matrix to warp DVX onto Basler frame
H_resized = np.array([
    [7.81740620e-01, -7.77607070e-02, 1.59554279e+02],
    [-9.05046835e-04, 9.22111603e-01, 1.51813399e+01],
    [-4.14179057e-05, -1.45666882e-04, 1.00000000e+00]
])

# Directory for wire templates
TEMPLATE_DIR = "wire_templates"
os.makedirs(TEMPLATE_DIR, exist_ok=True)
TEMPLATE_PATH = os.path.join(TEMPLATE_DIR, "wire_template1.npy")

flash_state = True  # Toggle for flashing overlay
flash_counter = 0  # Counter for flashing

WIRE_THICKNESS = 8  # Approximate wire thickness in pixels
SPLINE_POINTS = 100  # Number of spline points for interpolation

# ---------------- Basler Camera ----------------
def init_basler():
    """Initialize Basler camera and return camera and converter objects."""
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
    """Continuously grab frames from Basler camera."""
    global basler_frame
    while cam.IsGrabbing():
        res = cam.RetrieveResult(1000, pylon.TimeoutHandling_ThrowException)
        if res.GrabSucceeded():
            gray = conv.Convert(res).GetArray()
            with frame_lock:
                basler_frame = gray.copy()
        res.Release()

# ---------------- DVX Camera ----------------
def init_dvx():
    """Initialize DVX camera and visualizer."""
    cam = dv.io.CameraCapture(DVX_SERIAL)
    res = cam.getEventResolution()
    vis = dv.visualization.EventVisualizer(res)
    vis.setBackgroundColor((0,0,0))
    vis.setPositiveColor((0,255,0))
    vis.setNegativeColor((0,0,255))
    return cam, vis

def dvx_worker(cam, vis):
    """Continuously grab events from DVX camera and generate images."""
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
    """Overlay DVX image onto resized Basler frame with transparency."""
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

# ---------------- Wire Detection ----------------
def detect_wire(masked_gray):
    """Detect wire in masked grayscale image using contours and spline."""
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
        u_new = np.linspace(0, 1, SPLINE_POINTS)
        x_new, y_new = splev(u_new, tck)
        if np.any(np.isnan(x_new)) or np.any(np.isnan(y_new)):
            return None, None, None
        return wire, x_new, y_new
    except:
        return None, None, None

def draw_wire(img, wire_pts, x_spline, y_spline):
    """Draw detected wire points and spline; highlight intersections."""
    # Draw wire points
    if wire_pts is not None:
        for pt in wire_pts:
            cv2.circle(img, tuple(pt), 2, (0, 0, 255), -1)
    # Draw spline line
    if x_spline is not None and y_spline is not None:
        for i in range(len(x_spline)-1):
            pt1 = (int(x_spline[i]), int(y_spline[i]))
            pt2 = (int(x_spline[i+1]), int(y_spline[i+1]))
            cv2.line(img, pt1, pt2, (0, 255, 0), 2)
        # Highlight biggest intersection
        intersections = find_largest_spline_intersection(x_spline, y_spline, WIRE_THICKNESS)
        if intersections is not None:
            cv2.circle(img, intersections, 15, (0,0,255), 3)  # big red circle
    return img

# ---------------- Intersection Detection ----------------
def find_largest_spline_intersection(x, y, thickness):
    """Find largest intersection in spline segments."""
    max_dist = 0
    intersection_point = None
    for i in range(len(x)-1):
        for j in range(i+2, len(x)-1):  # skip consecutive segments
            pt = segment_proximity_intersection((x[i], y[i]), (x[i+1], y[i+1]),
                                                (x[j], y[j]), (x[j+1], y[j+1]), thickness)
            if pt is not None:
                d = np.hypot(pt[0]-x[i], pt[1]-y[i])
                if d > max_dist:
                    max_dist = d
                    intersection_point = pt
    return intersection_point

def segment_proximity_intersection(p1, p2, q1, q2, threshold):
    """Check if two line segments are within a threshold and return approximate intersection."""
    def dist2_segments(a1,a2,b1,b2):
        a1 = np.array(a1); a2 = np.array(a2)
        b1 = np.array(b1); b2 = np.array(b2)
        def seg_dist(p,r,q,s):
            u = r-p
            v = s-q
            w0 = p-q
            a = np.dot(u,u)
            b = np.dot(u,v)
            c = np.dot(v,v)
            d = np.dot(u,w0)
            e = np.dot(v,w0)
            denom = a*c - b*b
            if denom == 0:
                sc, tc = 0, d/b if b!=0 else 0
            else:
                sc = (b*e - c*d)/denom
                tc = (a*e - b*d)/denom
            sc = np.clip(sc,0,1)
            tc = np.clip(tc,0,1)
            pt1 = p + sc*u
            pt2 = q + tc*v
            return np.linalg.norm(pt1 - pt2)
        return seg_dist(a1,a2,b1,b2)
    d = dist2_segments(p1,p2,q1,q2)
    if d <= threshold:
        return (int((p1[0]+p2[0]+q1[0]+q2[0])/4),
                int((p1[1]+p2[1]+q1[1]+q2[1])/4))
    return None

# ---------------- Similarity ----------------
def compute_similarity(x_new, y_new, template):
    """Compute shape similarity to template using directed Hausdorff distance."""
    if template is None:
        return None
    curve = np.column_stack([x_new, y_new])
    d1 = directed_hausdorff(curve, template)[0]
    d2 = directed_hausdorff(template, curve)[0]
    dist = max(d1, d2)
    norm_dist = dist / np.hypot(RESIZED_W, RESIZED_H)
    similarity = max(0, 100 * (1 - norm_dist))
    return round(similarity, 1)

# ---------------- Mask with Rectangle ----------------
def intensity_mask_rectangle(frame, low, high, rect_w, rect_h, rect_cx, rect_cy, msg=""):
    """Create intensity mask using rectangle; detect and draw wire."""
    global similarity_msg, flash_state, flash_counter
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
    if wire_pts is not None and x_spline is not None and len(x_spline) > 20:
        wire_pts_global = wire_pts + np.array([x1, y1])
        x_spline_global = x_spline + x1
        y_spline_global = y_spline + y1
        mask_color = draw_wire(mask_color, wire_pts_global, x_spline_global, y_spline_global)

        if os.path.exists(TEMPLATE_PATH):
            template = np.load(TEMPLATE_PATH)
            sim = compute_similarity(x_spline_global, y_spline_global, template)
            similarity_msg = f"Shape match: {sim}%"

            # Layout status
            if sim >= 90:
                flash_counter += 1
                if flash_counter % 5 == 0:  # toggle every 5 frames
                    flash_state = not flash_state
                if flash_state:
                    cv2.putText(mask_color, "CORRECT LAYOUT", (400, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 3)
            else:
                cv2.putText(mask_color, "INCORRECT LAYOUT", (400, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
        else:
            similarity_msg = "No template saved"
    else:
        similarity_msg = "No wire detected"

    # Legend display
    legend_lines = [
        "Low threshold: q/w",
        "High threshold: a/s",
        "Width: r/e",
        "Height: f/t",
        "Move: i/j/k/l",
        "Save rect: m | Save wire: g",
        msg,
        similarity_msg
    ]
    for i, line in enumerate(legend_lines):
        cv2.putText(mask_color, line, (10, 80 + i*25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,255), 2)
    return mask_color

# ---------------- Screenshot ----------------
def save_screenshot(img, name_prefix="overlay"):
    """Save screenshot of given image with counter."""
    global screenshot_count, last_screenshot_msg
    os.makedirs("screenshots", exist_ok=True)
    path = f"screenshots/{name_prefix}_{screenshot_count:03d}.png"
    cv2.imwrite(path, img)
    last_screenshot_msg = f"Saved screenshot {name_prefix}_{screenshot_count:03d}.png"
    print(f"[INFO] {last_screenshot_msg}")
    screenshot_count += 1

# ---------------- Main Loop ----------------
def main():
    """Main application loop for capturing, overlay, wire detection, and display."""
    global basler_frame, dvx_event_frame, last_screenshot_msg

    bas_cam, bas_conv = init_basler()
    dvx_cam, dvx_vis = init_dvx()
    threading.Thread(target=basler_worker, args=(bas_cam, bas_conv), daemon=True).start()
    threading.Thread(target=dvx_worker, args=(dvx_cam, dvx_vis), daemon=True).start()

    mask_low, mask_high = 0, 130
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
        elif key == ord('g'):
            # Save wire template
            gray = cv2.resize(b_frame, (RESIZED_W, RESIZED_H))
            mask = np.zeros((RESIZED_H, RESIZED_W), dtype=np.uint8)
            x1 = max(rect_cx - rect_w//2, 0)
            y1 = max(rect_cy - rect_h//2, 0)
            x2 = min(rect_cx + rect_w//2, RESIZED_W)
            y2 = min(rect_cy + rect_h//2, RESIZED_H)
            roi = gray[y1:y2, x1:x2]
            mask_roi = np.where(roi <= mask_high, 0, 255).astype(np.uint8)
            mask[y1:y2, x1:x2] = mask_roi
            wire_pts, x_spline, y_spline = detect_wire(mask[y1:y2, x1:x2])
            if wire_pts is not None:
                x_spline_global = x_spline + x1
                y_spline_global = y_spline + y1
                template = np.column_stack([x_spline_global, y_spline_global])
                np.save(TEMPLATE_PATH, template)
                print(f"[INFO] Saved wire template to {TEMPLATE_PATH}")
                last_screenshot_msg = "Wire template saved!"

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
