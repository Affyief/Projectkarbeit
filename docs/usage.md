# Usage Guide

## Running the Pipeline
- Start the pipeline with:
    ```sh
    python main.py
    ```
- The main window will display Basler, DVXplorer, Overlay, and Masked views.

## Controls
- `ESC`: Quit
- `p`: Save screenshot of all views
- `m`: Save cropped rectangle screenshot
- `g`: Save wire template (for matching)
- Arrow keys and letter keys adjust mask/ROI parameters:
    - `q/w`: Low threshold down/up
    - `a/s`: High threshold down/up
    - `r/e`: ROI width down/up
    - `f/t`: ROI height down/up
    - `i/j/k/l`: Move ROI up/left/down/right

## Output Files
- **Screenshots:** Saved in `screenshots/` as `.png`
- **Wire Templates:** Saved in `wire_templates/` as `.npy`

## Template Matching
- If a wire template exists, wire shape similarity will be shown.
