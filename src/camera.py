import cv2

import numpy as np
import pykinect_azure as pykinect
import matplotlib.pyplot as plt

from src.point_cloud import PointCloudVisualizer


class AzureKinect:
    def __init__(
            self,
            color_format: int = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32,
            color_resolution: int = pykinect.K4A_COLOR_RESOLUTION_720P,
            depth_mode: int = pykinect.K4A_DEPTH_MODE_NFOV_2X2BINNED,
            track_body: bool = False
    ):
        pykinect.initialize_libraries(track_body=track_body)

        self.device_configuration = pykinect.default_configuration
        self.device_configuration.color_format = color_format
        self.device_configuration.color_resolution = color_resolution
        self.device_configuration.depth_mode = depth_mode

        self.device = None

    def start(self) -> pykinect.Device:
        self.device: pykinect.Device = pykinect.start_device(config=self.device_configuration)
        return self.device

    def stop(self):
        if self.device is None:
            raise ValueError("Device is not initialized. Please call start() method first.")

        self.device.close()

    def get_capture(self) -> pykinect.Device.capture:
        if self.device is None:
            raise ValueError("Device is not initialized. Please call start() method first.")

        return self.device.update()

    def get_color_image(self) -> (bool, np.ndarray):
        capture: pykinect.Capture = self.get_capture()
        is_valid, color_image = capture.get_color_image()
        if is_valid:
            color_image = color_image[..., :3]
            color_image = self.convert_to_cv2(self=self, image=color_image)
        return is_valid, color_image

    def get_depth_image(self) -> (bool, np.ndarray):
        capture: pykinect.Capture = self.get_capture()
        is_valid, depth_image = capture.get_depth_image()
        return is_valid, depth_image

    def get_smoothed_depth_image(self) -> (bool, np.ndarray):
        capture: pykinect.Capture = self.get_capture()
        is_valid, smoothed_depth_image = capture.get_smooth_depth_image()
        return is_valid, smoothed_depth_image

    def get_colored_depth_image(self) -> (bool, np.ndarray):
        capture: pykinect.Capture = self.get_capture()
        is_valid, depth_image = capture.get_colored_depth_image()
        if is_valid:
            depth_image = depth_image[..., :3]
            depth_image = self.convert_to_cv2(self=self, image=depth_image)
        return is_valid, depth_image

    def get_smoothed_colored_depth_image(self) -> (bool, np.ndarray):
        capture: pykinect.Capture = self.get_capture()
        is_valid, smoothed_depth_image = capture.get_smooth_colored_depth_image()
        if is_valid:
            smoothed_depth_image = smoothed_depth_image[..., :3]
            smoothed_depth_image = self.convert_to_cv2(self=self, image=smoothed_depth_image)
        return is_valid, smoothed_depth_image

    def get_ir_image(self) -> (bool, np.ndarray):
        capture: pykinect.Capture = self.get_capture()
        is_valid, ir_image = capture.get_ir_image()
        return is_valid, ir_image

    def get_point_cloud(self) -> (bool, np.ndarray):
        capture: pykinect.Capture = self.get_capture()
        is_valid, point_cloud = capture.get_pointcloud()
        return is_valid, point_cloud

    @staticmethod
    def convert_to_cv2(self, image: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)


if __name__ == '__main__':
    azure_kinect = AzureKinect()
    azure_kinect.start()

    visualizer = PointCloudVisualizer(
        window_name="Azure Kinect Point Cloud Visualizer",
        width=640,
        height=480
    )

    while True:
        is_valid_di, depth_image = azure_kinect.get_depth_image()

        if depth_image is not None:
            plt.pcolormesh(depth_image, cmap='viridis')
            plt.colorbar()
            plt.show()
            break

        if cv2.waitKey(1) == ord('q'):
            break

    azure_kinect.stop()