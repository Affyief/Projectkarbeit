import cv2
import numpy as np
from pypylon import pylon
from pupil_apriltags import Detector

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect

def main():
    # Initialize Basler camera
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    camera.Open()
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    # Convert to grayscale
    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_Mono8
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    prev_gray = None
    tag_detector = Detector(families='tag36h11')

    print("Press ESC to quit.")

    while camera.IsGrabbing():
        grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        if grab_result.GrabSucceeded():
            gray = converter.Convert(grab_result).GetArray()
            display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            # === AprilTag detection ===
            tags = tag_detector.detect(gray)
            for tag in tags:
                pts = tag.corners.reshape((-1, 1, 2)).astype(int)
                cv2.polylines(display, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
                cx, cy = int(tag.center[0]), int(tag.center[1])
                cv2.putText(display, str(tag.tag_id), (cx - 10, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # === Region Mask: Only if 4 tags ===
            if len(tags) >= 4:
                tag_pts = np.array([tag.center for tag in tags[:4]], dtype=np.float32)
                ordered_pts = order_points(tag_pts)

                # Make region mask from tag polygon
                region_mask = np.zeros_like(gray, dtype=np.uint8)
                cv2.fillConvexPoly(region_mask, ordered_pts.astype(np.int32), 255)

                # Draw the detection region outline
                cv2.polylines(display, [ordered_pts.astype(int)], isClosed=True, color=(255, 255, 0), thickness=2)

                # === Motion detection inside masked region ===
                if prev_gray is not None:
                    diff = cv2.absdiff(gray, prev_gray)
                    _, motion = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
                    motion = cv2.bitwise_and(motion, region_mask)

		

                    # Clean noise
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                    motion = cv2.morphologyEx(motion, cv2.MORPH_OPEN, kernel)

                    contours, _ = cv2.findContours(motion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for cnt in contours:
                        if cv2.contourArea(cnt) > 500:
                            cv2.drawContours(display, [cnt], -1, (0, 0, 255), 2)

            # Update previous frame
            prev_gray = gray.copy()

            # === Show final image ===
            cv2.imshow("Motion in AprilTag Region", display)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        grab_result.Release()

    camera.StopGrabbing()
    camera.Close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

