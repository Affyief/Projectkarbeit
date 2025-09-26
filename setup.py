from setuptools import setup, find_packages

setup(
    name="wire_detection_pipeline",
    version="0.1.0",
    description="Dual camera wire detection pipeline (Basler + DVXplorer)",
    author="Affyief",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "opencv-python",
        "pypylon",
        "dv-processing",
        "scipy",
    ],
    python_requires=">=3.8",
)
