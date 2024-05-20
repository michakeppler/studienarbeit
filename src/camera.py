import cv2

import numpy as np
import pykinect_azure as pykinect


class AzureKinect:
    """
    The AzureKinect class provides an interface to interact with Azure Kinect devices.

    Args:
        color_format (int): The color format of the device. Defaults to pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32.
        color_resolution (int): The color resolution of the device. Defaults to pykinect.K4A_COLOR_RESOLUTION_720P.
        depth_mode (int): The depth mode of the device. Defaults to pykinect.K4A_DEPTH_MODE_NFOV_2X2BINNED.
        track_body (bool): Whether to track body using the device. Defaults to False.

    Attributes:
        device_configuration (pykinect.DeviceConfiguration): The device configuration.
        device (pykinect.Device): The Azure Kinect device.

    Methods:
        start: Starts the Azure Kinect device and returns the device object.
        stop: Stops the Azure Kinect device.
        get_device_calibration: Gets the calibration matrix of the device.
        get_capture: Gets the current capture from the device.
        get_imu_capture: Gets the current IMU sample from the device.
        get_imu_accelerometer: Get the accelerometer data from the IMU sample.
        get_imu_gyroscope: Get the gyroscope data from the IMU sample.
        get_color_image: Gets the current color image capture from the device.
        get_depth_image: Gets the current depth image capture from the device.
        get_smoothed_depth_image: Gets the current smoothed depth image capture from the device.
        get_colored_depth_image: Gets the current colored depth image capture from the device.
        get_smoothed_colored_depth_image: Gets the current smoothed colored depth image capture from the device.
        get_ir_image: Gets the current IR image capture from the device.
        get_point_cloud: Gets the current point cloud capture from the device.
        get_transformed_depth_image: Gets the current transformed depth image capture from the device.
        convert_2d_to_3d: Converts a 2D point on the color image to a 3D point in the camera space.
        convert_3d_to_2d: Converts a 3D point in the camera space to a 2D point on the color image.
        get_calibration: Gets the calibration matrix from the device object.
        convert_to_cv2: Converts a color image from BGRA format to RGB format using OpenCV.

    """
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
        """
        Starts the Kinect device.

        :return: The initialized Kinect device.
        """
        self.device: pykinect.Device = pykinect.start_device(config=self.device_configuration)
        return self.device

    def stop(self):
        """
        Stop the device.

        :raises ValueError: If the device is not initialized.
        """
        if self.device is None:
            raise ValueError("Device is not initialized. Please call start() method first.")

        self.device.close()

    def get_device_calibration(self, calibration_type: str = "color") -> np.ndarray:
        """
        :param calibration_type: The type of calibration to retrieve. Should be either 'color' or 'depth'. Defaults to 'color'.
        :return: The calibration matrix as a numpy array of dtype np.float64.
        """
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
        """
        Returns the next capture frame from the Kinect device.

        :return: The next capture frame.
        :rtype: pykinect.Capture
        :raises ValueError: If the device is not initialized. Please call start() method first.
        """
        if self.device is None:
            raise ValueError("Device is not initialized. Please call start() method first.")

        return self.device.update()

    def get_imu_capture(self) -> pykinect.ImuSample:
        """
        Method to get the latest IMU (Inertial Measurement Unit) capture.

        :return: The latest IMU capture as an instance of pykinect.ImuSample.
        :rtype: pykinect.ImuSample
        :raises ValueError: If the device is not initialized. Please call the start() method first.
        """
        if self.device is None:
            raise ValueError("Device is not initialized. Please call start() method first.")

        return self.device.update_imu()

    def get_imu_accelerometer(self) -> np.ndarray:
        """
        Returns the accelerometer data from the IMU.

        :return: A numpy array containing the accelerometer data.
        """
        imu_sample: pykinect.ImuSample = self.get_imu_capture()
        accelerometer = imu_sample.get_acc()
        return accelerometer

    def get_imu_gyroscope(self) -> np.ndarray:
        """
        Get the gyroscope data from the IMU capture.

        :return: A numpy array representing the gyroscope data.
        """
        imu_sample: pykinect.ImuSample = self.get_imu_capture()
        gyroscope = imu_sample.get_gyro()
        return gyroscope

    def get_color_image(self) -> (bool, np.ndarray):
        """
        Retrieves the color image from the Kinect sensor.

        :return: A tuple containing a boolean value indicating if the color image is valid and the color image as a numpy array.
        """
        capture: pykinect.Capture = self.get_capture()
        is_valid, color_image = capture.get_color_image()
        if is_valid:
            color_image = color_image[..., :3]
            # color_image = self.convert_to_cv2(self=self, image=color_image)
        return is_valid, color_image

    def get_depth_image(self) -> (bool, np.ndarray):
        """
        Retrieves the depth image from the Kinect sensor.

        :return: A tuple containing a boolean indicating the validity of the depth image and the depth image itself as a numpy.ndarray.
        """
        capture: pykinect.Capture = self.get_capture()
        is_valid, depth_image = capture.get_transformed_depth_image()
        return is_valid, depth_image

    def get_smoothed_depth_image(self) -> (bool, np.ndarray):
        """
        :return: A tuple containing a boolean value indicating the validity of the smoothed depth image and the smoothed depth image itself as a numpy ndarray.
        """
        capture: pykinect.Capture = self.get_capture()
        is_valid, smoothed_depth_image = capture.get_smooth_depth_image()
        return is_valid, smoothed_depth_image

    def get_colored_depth_image(self) -> (bool, np.ndarray):
        """
        Returns the colored depth image captured by the Kinect sensor.

        :return: A tuple containing a boolean value indicating if the captured image is valid (True) or not (False),
                 and a numpy array representing the colored depth image. The depth image is represented using the RGB
                 color space, with shape (height, width, 3) where height and width are the dimensions of the image.

        Example usage:
            is_valid, depth_image = object.get_colored_depth_image()
            if is_valid:
                cv2.imshow("Colored Depth Image", depth_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        """
        capture: pykinect.Capture = self.get_capture()
        is_valid, depth_image = capture.get_colored_depth_image()
        if is_valid:
            depth_image = depth_image[..., :3]
            depth_image = self.convert_to_cv2(self=self, image=depth_image)
        return is_valid, depth_image

    def get_smoothed_colored_depth_image(self) -> (bool, np.ndarray):
        """
        Retrieves a smoothed colored depth image from the capture.

        :return: A tuple containing a boolean indicating if the capture is valid and a numpy array representing the
                 smoothed colored depth image.
        """
        capture: pykinect.Capture = self.get_capture()
        is_valid, smoothed_depth_image = capture.get_smooth_colored_depth_image()
        if is_valid:
            smoothed_depth_image = smoothed_depth_image[..., :3]
            smoothed_depth_image = self.convert_to_cv2(self=self, image=smoothed_depth_image)
        return is_valid, smoothed_depth_image

    def get_ir_image(self) -> (bool, np.ndarray):
        """
        Gets the infrared (IR) image from the Kinect sensor.

        :return: A tuple containing a boolean flag indicating the validity of the IR image, and the IR image itself as a NumPy array.
        :rtype: tuple(bool, np.ndarray)
        """
        capture: pykinect.Capture = self.get_capture()
        is_valid, ir_image = capture.get_ir_image()
        return is_valid, ir_image

    def get_point_cloud(self) -> (bool, np.ndarray):
        """
        Retrieve the transformed point cloud from the current capture.

        :return: A tuple containing a boolean value indicating the availability and validity of the point cloud,
                 and a numpy array containing the transformed point cloud data.
        :rtype: tuple
        """
        capture: pykinect.Capture = self.get_capture()
        is_valid, point_cloud = capture.get_transformed_pointcloud()
        return is_valid, point_cloud

    def get_transformed_depth_image(self) -> (bool, np.ndarray):
        """
        Returns the transformed depth image.

        :return: A tuple containing a boolean value indicating whether the depth image is valid and
                 a NumPy array representing the transformed depth image.
        """
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
        """
        Converts a 2D point to 3D using calibration information.

        :param self: The instance of the class.
        :param source_point_2d: The 2D point to convert to 3D.
        :param source_depth: The depth of the source point.
        :param source_camera: The source camera used to capture the 2D point.
        :param target_camera: The target camera to convert the 2D point to.
        :return: The converted 3D point.

        """
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
        """
        :param self: The object instance.
        :param source_point_3d: The 3D point to be converted to 2D.
        :param source_camera: The source camera used for the conversion.
        :param target_camera: The target camera used for the conversion.
        :return: The 2D point obtained from the conversion.

        Converts a 3-dimensional point to a 2-dimensional point using the provided source and target cameras.

        Example usage:
            source_point_3d = [1, 2, 3]
            source_camera = 'CameraA'
            target_camera = 'CameraB'
            point_2d = convert_3d_to_2d(self, source_point_3d, source_camera, target_camera)
            print(point_2d)  # Output: [x, y]
        """
        calibration: pykinect.Calibration = self.device.calibration

        point_2d = calibration.convert_3d_to_2d(
            source_point3d=source_point_3d,
            source_camera=source_camera,
            target_camera=target_camera
        )
        return point_2d

    @staticmethod
    def get_calibration(self, device: pykinect.Device) -> pykinect.Calibration:
        """
        :param self: The current instance of the class containing this method.
        :param device: The pykinect Device object from which to get the calibration.
        :return: The pykinect Calibration"""
        try:
            calibration = device.calibration
        except Exception as e:
            raise e
        return calibration

    @staticmethod
    def convert_to_cv2(self, image: np.ndarray) -> np.ndarray:
        """
        Converts an image from BGRA color space to RGB color space using OpenCV.

        :param self: The object instance.
        :param image: The input image array in BGRA format.
        :return: The converted image array in RGB format.
        """
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
