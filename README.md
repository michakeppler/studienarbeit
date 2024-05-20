# Official repository for the "Studienarbeit"
##### Here we need some text, that desc the project in one short sentence.


## Installation
To install the project, you need to run the following commands (req. anaconda - from requirements.txt):
```bash
git clone https://github.com/michakeppler/studienarbeit.git
cd studienarbeit
conda create -n dhbw python=3.8
conda activate dhbw
pip install -r requirements.txt
```

To install the project, you need to run the following commands (req. anaconda - from scratch):
```bash
git clone https://github.com/michakeppler/studienarbeit.git
cd studienarbeit
conda create -n dhbw python=3.8
conda activate dhbw
pip install mediapipe
pip install opencv-python
pip install numpy
pip install pykinect-azure
```

### Table of Contents
1. [Motivation](#motivation)
2. [Examples](#examples)
3. [Architecture](#architecture)
4. [Experiment](#experiment)
5. [Coming Soon](#coming-soon)

## Motivation

## Examples
Examples of the project can be found in the `examples` folder. 
To run the full project (camera + experiment):
```python
import pykinect_azure as pykinect

from src.camera import AzureKinect
from src.solver import Solver

if __name__ == '__main__':
    pykinect.initialize_libraries(track_body=False)

    device_configuration = pykinect.default_configuration
    device_configuration.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
    device_configuration.color_resolution = pykinect.K4A_COLOR_RESOLUTION_720P
    device_configuration.depth_mode = pykinect.K4A_DEPTH_MODE_NFOV_UNBINNED

    camera = AzureKinect(
        color_format=device_configuration.color_format,
        color_resolution=device_configuration.color_resolution,
        depth_mode=device_configuration.depth_mode,
        track_body=False
    )
    solver = Solver(
        camera=camera
    )
    solver.start()
    solver.solve()
    solver.stop()
```


If you just wanna work with the camera:
```python
from src.camera import AzureKinect

# Creating an instance of `AzureKinect`
camera = AzureKinect()

# Start
camera.start()

# Do something :-)
for i in range(5):
    gyro = camera.get_imu_gyroscope()
    print(f"Gyroscope: {gyro}")

# Stop
camera.stop()
```

## Architecture

## Experiment
Optional

## Coming soon
Optional