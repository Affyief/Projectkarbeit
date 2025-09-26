import cv2
import numpy as np
import os

def overlay_dvx_on_basler_resized(b_frame, d_frame, H_resized, alpha=0.5, width=940, height=540):
    """
    Overlay DVX event data (d_frame) onto Basler camera data (b_frame) using homography.
    Returns color overlay image.
    """
    if b_frame is None:
        b_frame = np.zeros((height, width), dtype=np.uint8)
    if d_frame is None:
        d_frame = np.zeros((height, width), dtype=np.uint8)
    b_resized = cv2.resize(b_frame, (width, height))
    d_resized = cv2.resize(d_frame, (width, height))
    warped_dvx = cv2.warpPerspective(d_resized, H_resized, (width, height))
    b_color = cv2.cvtColor(b_resized, cv2.COLOR_GRAY2BGR)
    d_color = np.zeros((height, width, 3), dtype=np.uint8)
    d_color[:, :, 2] = warped_dvx
    overlay = cv2.addWeighted(b_color, 1-alpha, d_color, alpha, 0)
    return overlay

def save_screenshot(img, out_dir="screenshots", name_prefix="overlay", count=0):
    """
    Save an image to the screenshots directory, auto-numbered.
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name_prefix}_{count:03d}.png")
    cv2.imwrite(path, img)
    print(f"[INFO] Saved screenshot {name_prefix}_{count:03d}.png")
    return path

def draw_legend(img, legend_lines, start_y=80, color=(0,255,255)):
    """
    Draws legend instructions/messages onto an OpenCV image.
    """
    for i, line in enumerate(legend_lines):
        cv2.putText(img, line, (10, start_y + i*25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    return img

def intensity_mask_rectangle(frame, low, high, rect_w, rect_h, rect_cx, rect_cy):
    """
    Create a masked rectangle on the image for ROI-based wire detection.
    Returns color mask image.
    """
    RESIZED_W, RESIZED_H = 940, 540
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
    return mask_color
