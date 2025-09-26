# API Reference

## Main Functions

- `init_basler()`: Initializes Basler camera.
- `basler_worker(cam, conv)`: Streams frames from Basler.
- `init_dvx()`: Initializes DVXplorer camera.
- `dvx_worker(cam, vis)`: Streams event frames from DVXplorer.
- `overlay_dvx_on_basler_resized(b_frame, d_frame)`: Overlays event data on camera frames.
- `detect_wire(masked_gray)`: Detects wire in ROI and fits spline.
- `draw_wire(img, wire_pts, x_spline, y_spline)`: Draws wire and spline on image.
- `find_largest_spline_intersection(x, y, thickness)`: Detects intersections in spline.
- `compute_similarity(x_new, y_new, template)`: Compares wire shape to template.
- `intensity_mask_rectangle(...)`: Applies rectangle mask and legend overlay.
- `save_screenshot(img, name_prefix)`: Saves an image to disk.

## Global Variables
- `RESIZED_W`, `RESIZED_H`: Output dimensions.
- `ALPHA`: Overlay transparency.
- `H_resized`: Homography matrix.
- `TEMPLATE_PATH`: Path for wire template.
