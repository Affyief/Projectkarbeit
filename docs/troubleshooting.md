# Troubleshooting

## No Basler Camera Found
- Ensure Basler camera is connected, powered, and drivers are installed.
- Check `pypylon` installation.

## DVXplorer Not Found
- Ensure correct serial number is set in `DVX_SERIAL`.
- Check `dv-processing` installation.
- Disconnect / Reconnect camera.

## Window Not Displaying Properly
- Check OpenCV installation.
- Ensure your system supports GUI windows.

## Homography Misalignment
- Check homography matrix values in code.
- Re-calibrate with correct frame size.
- Use different calibration pattern.

## Screenshots/Template Not Saving
- Ensure `screenshots/` and `wire_templates/` folders exist or are creatable.
- Check file permissions.

## General Tips
- Restart the script if cameras disconnect.
- Confirm all dependencies are installed.
- Review console logs for Python exceptions.
