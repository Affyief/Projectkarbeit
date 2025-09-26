import pytest
import numpy as np
import src.utils.helpers as helpers

def test_overlay_function():
    # Test overlay function with dummy frames
    b_frame = np.zeros((540, 940), dtype=np.uint8)
    d_frame = np.zeros((540, 940), dtype=np.uint8)
    overlay = helpers.overlay_dvx_on_basler_resized(b_frame, d_frame)
    assert overlay.shape == (540, 940, 3)
    assert overlay.dtype == np.uint8

def test_mask_rectangle():
    b_frame = np.zeros((540, 940), dtype=np.uint8)
    mask_img = helpers.intensity_mask_rectangle(b_frame, 0, 100, 500, 300, 470, 270)
    assert mask_img.shape == (540, 940, 3)
    assert mask_img.dtype == np.uint8
