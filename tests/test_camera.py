import pytest
import numpy as np

def test_basler_init(monkeypatch):
    # Mock pypylon.TlFactory and InstantCamera for test
    import src.camera.camera_manager as camera_manager
    class DummyFactory:
        def EnumerateDevices(self): return ['dummy']
        def CreateDevice(self, device): return 'camera'
    class DummyCamera:
        def Open(self): pass
        def OffsetX(self): return type('obj', (), {'SetValue': lambda self, x: None})()
        def OffsetY(self): return type('obj', (), {'SetValue': lambda self, x: None})()
        def Width(self): return type('obj', (), {'SetValue': lambda self, x: None, 'GetMax': lambda self: 100})()
        def Height(self): return type('obj', (), {'SetValue': lambda self, x: None, 'GetMax': lambda self: 100})()
        def StartGrabbing(self, strategy): pass
        def IsGrabbing(self): return False
        def Close(self): pass
    monkeypatch.setattr(camera_manager.pylon, 'TlFactory', lambda : DummyFactory())
    monkeypatch.setattr(camera_manager.pylon, 'InstantCamera', lambda cam: DummyCamera())
    cam, conv = camera_manager.init_basler()
    assert cam is not None

def test_dvx_init(monkeypatch):
    import src.camera.camera_manager as camera_manager
    class DummyCam:
        def getEventResolution(self): return (100,100)
    class DummyVis:
        def setBackgroundColor(self, c): pass
        def setPositiveColor(self, c): pass
        def setNegativeColor(self, c): pass
    monkeypatch.setattr(camera_manager.dv.io, 'CameraCapture', lambda serial: DummyCam())
    monkeypatch.setattr(camera_manager.dv.visualization, 'EventVisualizer', lambda res: DummyVis())
    cam, vis = camera_manager.init_dvx()
    assert cam is not None
    assert vis is not None
