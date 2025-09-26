"""
Clean, concise DVXplorer stream viewer
- Detects DVXplorer by serial (manual input)
- Displays its event stream with positive/negative colors
- Exit on ESC key or after 5 seconds
"""

import time
import cv2
from dv_processing import io
from dv_processing.visualization import EventVisualizer

def main():
    # --- Specify your DVXplorer serial here ---
    serial = "DVXplorer_DXAS0024"

    # --- Initialize camera capture ---
    camera = io.CameraCapture(serial)

    # --- Get event resolution ---
    resolution = camera.getEventResolution()

    # --- Setup event visualizer ---
    visualizer = EventVisualizer(resolution)
    visualizer.setPositiveColor((0, 255, 0))  # Green for positive events
    visualizer.setNegativeColor((0, 0, 255))  # Red for negative events

    print(f"Streaming events from {serial}... Press ESC to exit.")

    start = time.time()
    while True:
        if camera.isEventStreamAvailable():
            events = camera.getNextEventBatch()
            if events is not None and events.size() > 0:
                img = visualizer.generateImage(events)
                cv2.imshow("DVXplorer Events", img)

        # Exit on ESC key
        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

