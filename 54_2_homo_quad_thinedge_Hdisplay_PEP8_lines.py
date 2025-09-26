"""
Homography alignment between Basler and DVX event camera screenshots.

This script:
1. Loads resized 960x540 grayscale screenshots from disk.
2. Detects and matches ORB features.
3. Computes a homography mapping DVX → Basler perspective.
4. Warps the DVX image and overlays with the Basler.
5. Displays both original and aligned overlays, along with feature matches.

Author: FAPS
Date: 2025-09-11
"""

import os
import cv2
import numpy as np

# ------------------------------ Constants ------------------------------- #
FOLDER = "/home/faps/Desktop/projekt/screenshots"
BASLER_PATH = os.path.join(FOLDER, "b1.png")
DVX_PATH = os.path.join(FOLDER, "d1.png")
TILE_SIZE = (960, 540)
# ------------------------------------------------------------------------ #

# Load images in grayscale
img_b = cv2.imread(BASLER_PATH, 0)
img_d = cv2.imread(DVX_PATH, 0)

if img_b is None or img_d is None:
    raise FileNotFoundError("One of the images was not found!")


# -------------------------- Helper Functions ---------------------------- #
def preprocess_for_features(img):
    """Apply Gaussian blur and morphological thinning for feature detection."""
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thin = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, kernel)
    return thin


def label_image(img, label, homo=None, tile_size=TILE_SIZE):
    """Convert grayscale to BGR, add label, resolutions, and optional homography."""
    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img_resized = cv2.resize(img_bgr, tile_size)
    h_orig, w_orig = img.shape[:2]
    h_resized, w_resized = tile_size

    # Main label
    cv2.putText(
        img_resized,
        label,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Show resolutions
    cv2.putText(
        img_resized,
        f"Original: {w_orig}x{h_orig}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        img_resized,
        f"Resized: {w_resized}x{h_resized}",
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )

    # Overlay homography matrix if provided
    if homo is not None:
        H_text = [f"{homo[i, j]:.2f}" for i in range(3) for j in range(3)]
        y0 = 130
        for i in range(3):
            row_text = " ".join(H_text[i * 3 : (i + 1) * 3])
            cv2.putText(
                img_resized,
                row_text,
                (20, y0 + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

    return img_resized


def draw_feature_correspondences(img1, kp1, img2, kp2, matches, max_matches=50):
    """
    Draw colored lines between corresponding keypoints of two images.
    Returns a single concatenated image.
    """
    # Resize images to same height
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    H = max(h1, h2)
    scale1 = H / h1
    scale2 = H / h2
    img1_r = cv2.resize(img1, (int(w1*scale1), H))
    img2_r = cv2.resize(img2, (int(w2*scale2), H))

    # Convert to BGR
    if len(img1_r.shape) == 2:
        img1_r = cv2.cvtColor(img1_r, cv2.COLOR_GRAY2BGR)
    if len(img2_r.shape) == 2:
        img2_r = cv2.cvtColor(img2_r, cv2.COLOR_GRAY2BGR)

    # Concatenate horizontally
    vis = np.hstack((img1_r, img2_r))
    offset = img1_r.shape[1]

    colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (0,255,255), (255,0,255)]
    for i, m in enumerate(matches[:max_matches]):
        color = colors[i % len(colors)]
        pt1 = tuple(np.round(np.array(kp1[m.queryIdx].pt) * scale1).astype(int))
        pt2 = tuple(np.round(np.array(kp2[m.trainIdx].pt) * scale2).astype(int) + np.array([offset,0]))
        cv2.line(vis, pt1, pt2, color, 2)
        cv2.circle(vis, pt1, 4, color, -1)
        cv2.circle(vis, pt2, 4, color, -1)

    return vis
# ------------------------------------------------------------------------ #


def main():
    """Main routine to process images and display homography overlays."""
    # Preprocess images
    img_b_proc = preprocess_for_features(img_b)
    img_d_proc = preprocess_for_features(img_d)

    # ORB feature detection
    orb = cv2.ORB_create(1000)
    kp_b, des_b = orb.detectAndCompute(img_b_proc, None)
    kp_d, des_d = orb.detectAndCompute(img_d_proc, None)

    # Match features
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des_d, des_b)
    matches = sorted(matches, key=lambda x: x.distance)

    pts_d = np.float32([kp_d[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts_b = np.float32([kp_b[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Compute homography
    H, mask = cv2.findHomography(pts_d, pts_b, cv2.RANSAC, 5.0)
    print("Calculated Homography Matrix (DVX -> Basler):")
    print(H)

    # Warp DVX to Basler perspective
    h, w = img_b.shape
    dvx_aligned = cv2.warpPerspective(img_d, H, (w, h))

    # Overlays
    overlay_no_tfr = cv2.bitwise_or(img_b, img_d)
    overlay_tfr = cv2.bitwise_or(img_b, dvx_aligned)

    # Label each tile
    tile_tl = label_image(img_b, "Original B1")
    tile_tr = label_image(img_d, "Original D1")
    tile_bl = label_image(overlay_no_tfr, "Overlay No Transform")
    tile_br = label_image(overlay_tfr, "Overlay With Transform", H)

    # 2x2 quad layout
    top_row = np.hstack((tile_tl, tile_tr))
    bottom_row = np.hstack((tile_bl, tile_br))
    quad = np.vstack((top_row, bottom_row))

    # Draw feature correspondences for visual feedback
    corr_vis = draw_feature_correspondences(img_d, kp_d, img_b, kp_b, matches, max_matches=50)

    # Display
    cv2.imshow("Quad Layout: Original & Overlay", quad)
    cv2.imshow("Feature Correspondences", corr_vis)

    print("Press ESC to exit.")
    while True:
        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

