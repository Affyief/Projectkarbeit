"""
Basler + DVXplorer live event visualization and 1-pixel motion edges in QUAD window.

This script:
1. Initializes a Basler camera and a DVXplorer event camera.
2. Computes a 1-pixel-wide motion mask from Basler grayscale frames.
3. Generates event images and accumulated grayscale images from the DVXplorer.
4. Displays a 2x2 grid:
       - Basler motion B&W (1-pixel edges)
       - DVX events B&W
       - DVX accumulator B&W
       - Overlay of Basler + DVX events
5. Shows a note in the center of the quad:
       - Press 's' to save a screenshot.
       - Screenshots are stored in the 'screenshots' folder at 960x540 pixels.
6. Allows saving screenshots at display resolution (960x540):
       - basler_###.png and dvx_###.png
       - Incrementing numbers prevent overwrites.

Requirements:
    - OpenCV (cv2)
    - NumPy
    - pypylon
    - dv_processing (DVXplorer Python wrapper)

Author: Affyief
Date: 2025-09-11
"""

import os
import cv2
import time
import threading
import numpy as np
from pypylon import pylon
import dv_processing as dv

# ---------------------------- Shared Globals ---------------------------- #
basler_motion_mask = None
dvx_event_mask = None
dvx_accum_gray = None
frame_lock = threading.Lock()

DVX_SERIAL = "DVXplorer_DXAS0024"
TILE_W, TILE_H = 960, 540
screenshot_count = 0
# ------------------------------------------------------------------------ #


# -------------------------- Basler Functions ---------------------------- #
def init_basler():
    """Initialize Basler camera and return camera and converter."""
    factory = pylon.TlFactory.GetInstance()
    devices = factory.EnumerateDevices()
    if not devices:
        raise RuntimeError("No Basler camera found")

    cam = pylon.InstantCamera(factory.CreateDevice(devices[0]))
    cam.Open()
    cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    conv = pylon.ImageFormatConverter()
    conv.OutputPixelFormat = pylon.PixelType_Mono8  # force grayscale

    print(f"Basler resolution: {cam.Width.GetValue()}x{cam.Height.GetValue()}")
    print("Basler camera started.")
    return cam, conv


def basler_worker(cam, conv):
    """Continuously grab frames and compute 1-pixel-wide motion mask."""
    global basler_motion_mask
    prev_gray = None

    while cam.IsGrabbing():
        res = cam.RetrieveResult(1000, pylon.TimeoutHandling_ThrowException)
        if res.GrabSucceeded():
            gray = conv.Convert(res).GetArray()

            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                _, motion_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

                # Keep 1-pixel edges (no thickening)
                # Optionally remove isolated noise with small erosion
                # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
                # motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_ERODE, kernel)

                with frame_lock:
                    basler_motion_mask = motion_mask

            prev_gray = gray
        res.Release()
# ------------------------------------------------------------------------ #


# -------------------------- DVX Functions ------------------------------- #
def init_dvx():
    """Initialize DVXplorer camera, visualizer, and accumulator."""
    cam = dv.io.CameraCapture(DVX_SERIAL)
    res = cam.getEventResolution()
    print(f"DVXplorer {DVX_SERIAL} started with resolution {res}")

    vis = dv.visualization.EventVisualizer(res)
    vis.setBackgroundColor((0, 0, 0))
    vis.setPositiveColor((255, 255, 255))
    vis.setNegativeColor((255, 255, 255))

    acc = dv.Accumulator(res)
    acc.setMinPotential(0.0)
    acc.setMaxPotential(1.0)
    acc.setNeutralPotential(0.5)
    acc.setEventContribution(0.15)
    acc.setDecayFunction(dv.Accumulator.Decay.EXPONENTIAL)
    acc.setDecayParam(1e6)

    return cam, vis, acc


def dvx_worker(cam, vis, acc):
    """Continuously grab DVX events and update event/accumulator images."""
    global dvx_event_mask, dvx_accum_gray

    while True:
        if not cam.isEventStreamAvailable():
            time.sleep(0.005)
            continue

        events = cam.getNextEventBatch()
        if events is None or events.size() == 0:
            continue

        ev_img = vis.generateImage(events)
        ev_gray = cv2.cvtColor(ev_img, cv2.COLOR_BGR2GRAY)

        acc.accept(events)
        acc_frame = acc.generateFrame()
        acc_gray = acc_frame.image.copy() if acc_frame and acc_frame.image is not None else None

        with frame_lock:
            dvx_event_mask = ev_gray
            dvx_accum_gray = acc_gray
# ------------------------------------------------------------------------ #


# ---------------------------- Helper Functions -------------------------- #
def resize_to_tile(img):
    """Resize image to tile size or return blank tile if None."""
    if img is None:
        return np.zeros((TILE_H, TILE_W), dtype=np.uint8)
    return cv2.resize(img, (TILE_W, TILE_H))


def label_image(img, label):
    """Resize and overlay text label on grayscale image."""
    if img is None:
        img = np.zeros((TILE_H, TILE_W), dtype=np.uint8)
    img = resize_to_tile(img)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.putText(img_bgr, label, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
    return img_bgr


def save_screenshot(bas_img, dvx_img):
    """Save Basler and DVX images to screenshots folder with unique names."""
    global screenshot_count
    os.makedirs("screenshots", exist_ok=True)
    bas_path = f"screenshots/basler_{screenshot_count:03d}.png"
    dvx_path = f"screenshots/dvx_{screenshot_count:03d}.png"
    cv2.imwrite(bas_path, bas_img)
    cv2.imwrite(dvx_path, dvx_img)
    print(f"[INFO] Saved screenshot {screenshot_count:03d} at {TILE_W}x{TILE_H}")
    screenshot_count += 1
# ------------------------------------------------------------------------ #


# ------------------------------ Main Loop -------------------------------- #
def main():
    """Start Basler and DVX workers, display 2x2 grid, show central note."""
    global basler_motion_mask, dvx_event_mask, dvx_accum_gray

    bas_cam, bas_conv = init_basler()
    dvx_cam, dvx_vis, dvx_acc = init_dvx()

    threading.Thread(target=basler_worker, args=(bas_cam, bas_conv), daemon=True).start()
    threading.Thread(target=dvx_worker, args=(dvx_cam, dvx_vis, dvx_acc), daemon=True).start()

    while True:
        with frame_lock:
            b = basler_motion_mask.copy() if basler_motion_mask is not None else None
            ev = dvx_event_mask.copy() if dvx_event_mask is not None else None
            ac = dvx_accum_gray.copy() if dvx_accum_gray is not None else None

        # Label each tile
        tile_b = label_image(b, "Basler Motion B&W")
        tile_ev = label_image(ev, "DVX Events B&W")
        tile_ac = label_image(ac, "DVX Accumulator B&W")
        overlay = cv2.bitwise_or(resize_to_tile(b), resize_to_tile(ev)) if b is not None and ev is not None else np.zeros((TILE_H, TILE_W), dtype=np.uint8)
        tile_ol = label_image(overlay, "Overlay B&W")

        # 2x2 quad
        top_row = np.hstack((tile_b, tile_ev))
        bot_row = np.hstack((tile_ac, tile_ol))
        quad = np.vstack((top_row, bot_row))

        # Add central note
        note = "Press 's' to save screenshots (960x540) in 'screenshots' folder"
        text_size = cv2.getTextSize(note, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        x = (quad.shape[1] - text_size[0]) // 2
        y = (quad.shape[0] + text_size[1]) // 2
        cv2.putText(quad, note, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Basler | DVX Events | Accumulator | Overlay", quad)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            break
        elif key == ord('s'):
            with frame_lock:
                if b is not None and ev is not None:
                    save_screenshot(resize_to_tile(b), resize_to_tile(ev))

    bas_cam.StopGrabbing()
    bas_cam.Close()
    cv2.destroyAllWindows()
# ------------------------------------------------------------------------ #


if __name__ == "__main__":
    main()
