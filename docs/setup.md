# Setup Instructions

## Hardware Requirements
- Basler 1920gm Camera (compatible with pypylon)
- Inivation DVXplorer Camera (serial# required to be listed/changed in code)
- Sufficient USB ports and power
- Computer running Python 3.8+

## Software Requirements
- Python 3.8+
- [requirements.txt](../requirements.txt) or [environment.yml](../environment.yml)
- pypylon SDK
- DVXplorer SDK (dv-processing)
- OpenCV, NumPy, SciPy

## Installation Steps
1. Clone the repository:
    ```sh
    git clone https://github.com/Affyief/Projectkarbeit.git
    cd Projectkarbeit
    ```
2. Install dependencies:
    ```sh
    pip install -r requirements.txt
    ```
    or (with conda):
    ```sh
    conda env create -f environment.yml
    conda activate wire-detection-env
    ```
3. Connect both cameras to the computer.
4. Make sure `DVX_SERIAL` in the code matches your DVXplorer serial number.
5. Run the main script:
    ```sh
    python main.py
    ```
