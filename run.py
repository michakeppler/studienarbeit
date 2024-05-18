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
