import cv2

import numpy as np
import pykinect_azure as pykinect


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

        self.device: pykinect.Device = None

    def start(self) -> pykinect.Device:
        self.device: pykinect.Device = pykinect.start_device(config=self.device_configuration)
        return self.device

    def stop(self):
        if self.device is None:
            raise ValueError("Device is not initialized. Please call start() method first.")

        self.device.close()

    def get_device_calibration(self, calibration_type: str = "color") -> np.ndarray:
        if self.device is None:
            raise ValueError("Device is not initialized. Please call start() method first.")

        calibration: pykinect.Calibration = self.device.calibration
        if calibration_type == "color":
            matrix = calibration.get_matrix(camera=pykinect.K4A_CALIBRATION_TYPE_COLOR)
        elif calibration_type == "depth":
            matrix = calibration.get_matrix(camera=pykinect.K4A_CALIBRATION_TYPE_DEPTH)
        else:
            raise ValueError("Invalid calibration type. Please choose between 'color' and 'depth'.")

        return np.array(matrix, dtype=np.float64)

    def get_capture(self) -> pykinect.Capture:
        if self.device is None:
            raise ValueError("Device is not initialized. Please call start() method first.")

        return self.device.update()

    def get_imu_capture(self) -> pykinect.ImuSample:
        if self.device is None:
            raise ValueError("Device is not initialized. Please call start() method first.")

        return self.device.update_imu()

    def get_imu_accelerometer(self) -> np.ndarray:
        imu_sample: pykinect.ImuSample = self.get_imu_capture()
        accelerometer = imu_sample.get_acc()
        return accelerometer

    def get_imu_gyroscope(self) -> np.ndarray:
        imu_sample: pykinect.ImuSample = self.get_imu_capture()
        gyroscope = imu_sample.get_gyro()
        return gyroscope

    def get_color_image(self) -> (bool, np.ndarray):
        capture: pykinect.Capture = self.get_capture()
        is_valid, color_image = capture.get_color_image()
        if is_valid:
            color_image = color_image[..., :3]
            # color_image = self.convert_to_cv2(self=self, image=color_image)
        return is_valid, color_image

    def get_depth_image(self) -> (bool, np.ndarray):
        capture: pykinect.Capture = self.get_capture()
        is_valid, depth_image = capture.get_transformed_depth_image()
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
        is_valid, point_cloud = capture.get_transformed_pointcloud()
        return is_valid, point_cloud

    def get_transformed_depth_image(self) -> (bool, np.ndarray):
        capture: pykinect.Capture = self.get_capture()
        is_valid, transformed_depth_image = capture.get_transformed_depth_image()
        return is_valid, transformed_depth_image

    @staticmethod
    def convert_2d_to_3d(
        self,
        source_point_2d,
        source_depth,
        source_camera,
        target_camera
    ):
        calibration: pykinect.Calibration = self.device.calibration

        point_3d = calibration.convert_2d_to_3d(
            source_point2d=source_point_2d,
            source_depth=source_depth,
            source_camera=source_camera,
            target_camera=target_camera
        )
        return point_3d

    @staticmethod
    def convert_3d_to_2d(
        self,
        source_point_3d,
        source_camera,
        target_camera
    ):
        calibration: pykinect.Calibration = self.device.calibration

        point_2d = calibration.convert_3d_to_2d(
            source_point3d=source_point_3d,
            source_camera=source_camera,
            target_camera=target_camera
        )
        return point_2d

    @staticmethod
    def get_calibration(self, device: pykinect.Device) -> pykinect.Calibration:
        try:
            calibration = device.calibration
        except Exception as e:
            raise e
        return calibration


    @staticmethod
    def convert_to_cv2(self, image: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)


if __name__ == '__main__':
    camera = AzureKinect()

    camera.start()
    calibration = camera.get_device_calibration()
    print(calibration)

    camera.stop()