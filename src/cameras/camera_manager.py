"""
Camera Manager module for listing and initializing Basler and DVXplorer cameras.
"""

from pypylon import pylon
from dv_processing.io import CameraCapture
from dv_processing.visualization import EventVisualizer

def list_basler_cameras():
    """Lists all connected Basler cameras and prints their details."""
    factory = pylon.TlFactory.GetInstance()
    devices = factory.EnumerateDevices()
    if not devices:
        print("❌ No Basler cameras found.")
        return []
    print(f"✅ Found {len(devices)} Basler camera(s):")
    for i, dev in enumerate(devices):
        print(f"\nCamera {i}")
        print(f"  Model:         {dev.GetModelName()}")
        print(f"  Serial:        {dev.GetSerialNumber()}")
        print(f"  Vendor:        {dev.GetVendorName()}")
        print(f"  Device Class:  {dev.GetDeviceClass()}")
    return devices

def init_basler(index=0):
    """Initializes a Basler camera by index and returns camera, converter objects."""
    factory = pylon.TlFactory.GetInstance()
    devices = factory.EnumerateDevices()
    if not devices or index >= len(devices):
        raise RuntimeError("No Basler camera found.")
    cam = pylon.InstantCamera(factory.CreateDevice(devices[index]))
    cam.Open()
    conv = pylon.ImageFormatConverter()
    conv.OutputPixelFormat = pylon.PixelType_Mono8
    return cam, conv

def init_dvxplorer(serial="DVXplorer_DXAS0024"):
    """Initializes a DVXplorer camera and returns camera & event visualizer."""
    camera = CameraCapture(serial)
    res = camera.getEventResolution()
    if res is None:
        raise RuntimeError("Cannot get resolution from DVXplorer!")
    visualizer = EventVisualizer(res)
    visualizer.setPositiveColor((0, 255, 0))
    visualizer.setNegativeColor((0, 0, 255))
    visualizer.setBackgroundColor((0, 0, 0))
    return camera, visualizer
