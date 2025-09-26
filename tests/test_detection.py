import pytest
import numpy as np
import src.detection.wire_detection as wire_detection

def test_wire_detection_basic():
    # Create a synthetic image with a straight wire-like line
    img = np.zeros((540, 940), dtype=np.uint8)
    cv2 = __import__('cv2')
    cv2.line(img, (100, 270), (840, 270), 255, 8)
    wire_pts, x_spline, y_spline = wire_detection.detect_wire(img)
    assert wire_pts is not None
    assert x_spline is not None
    assert y_spline is not None

def test_wire_similarity():
    # Simulate two similar splines and compute similarity
    x = np.linspace(100, 800, 100)
    y = np.linspace(270, 270, 100)
    template = np.column_stack([x, y])
    sim = wire_detection.compute_similarity(x, y, template)
    assert sim == pytest.approx(100.0, abs=1.0)

def test_intersection_detection():
    # Create two crossing lines and test intersection logic
    x = np.concatenate([np.linspace(100, 800, 50), np.linspace(800, 100, 50)])
    y = np.concatenate([np.linspace(100, 800, 50), np.linspace(800, 100, 50)])
    result = wire_detection.find_largest_spline_intersection(x, y, thickness=8)
    assert result is not None
