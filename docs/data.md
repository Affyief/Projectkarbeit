# Data Formats & Files

## 1. Screenshots

- **Location:** `screenshots/`
- **Format:** `.png`
- **Naming:** `overlay_XXX.png`, `rect_XXX.png`
- **Description:**  
  Saved images of the current pipeline view or cropped ROI.  
  Used for analysis, presentation, or debugging.

## 2. Wire Templates

- **Location:** `wire_templates/`
- **Format:** `.npy` (NumPy arrays)
- **Naming:** `wire_template1.npy`, etc.
- **Description:**  
  Contains spline points (`(N, 2)` array of x/y coordinates) representing wire shapes for template matching.
  Used to compare live wire detection against saved reference shapes.

## 3. Homography Matrix

- **Location:** Hardcoded in code (can optionally save as `.npy`)
- **Format:** NumPy array (`(3, 3)`)
- **Description:**  
  Used to align DVXplorer events with Basler camera images for overlay.



## Example Screenshot File

- `screenshots/overlay_005.png`
    - Overlay of Basler and DVXplorer camera feeds with wire mask, 940x540 resolution.
<img width="1920" height="1080" alt="Screenshot from 2025-09-12 14-19-03" src="https://github.com/user-attachments/assets/8bf8629f-5480-4ef2-823a-8e3964615099" />

## Example Wire Template File

- `wire_templates/wire_template1.npy`
    - NumPy array, shape `(100, 2)`, spline points of a correctly routed wire.

## Data Usage

- **Screenshots**: For manual inspection, reporting, or debugging.
- **Wire Templates**: Loaded in code for wire shape similarity check and layout validation.

---

*Note: If you add new types of data, please update this file to document format, use, and location. :)*
