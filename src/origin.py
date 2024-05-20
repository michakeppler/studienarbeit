import math

import numpy as np
from pykinect_azure import k4a_float2_t, k4a_float3_t, K4A_CALIBRATION_TYPE_COLOR

from src.camera import AzureKinect
from src.coordinate import ImageCoordinate, WorldCoordinate


class Origin:
    """

    The `Origin` class represents the origin point in the coordinate system of a camera.

    **Attributes:**
    - `camera_angle`: The angle of the camera in degrees.
    - `camera`: An instance of the `AzureKinect` class representing the camera.

    **Methods:**
    - `__init__(self, camera_angle: int, camera: AzureKinect)`: Initializes a new instance of the `Origin` class.
    - `compute_origin(self, depth_image: np.ndarray, coordinates: list) -> ImageCoordinate`: Computes the origin point based on the provided depth image and image coordinates.
    - `compute_coordinate_origin(self, depth_image: np.ndarray, coordinate: ImageCoordinate) -> (ImageCoordinate, WorldCoordinate)`: Computes the origin point for a specific image coordinate.
    - `estimate_height(self, x: int, y: int, depth_image: np.ndarray) -> float`: Estimates the height at a specific pixel location.
    - `get_origin(self) -> ImageCoordinate`: Returns the origin point.
    - `get_image_corner(self, index: int) -> ImageCoordinate`: Returns the image corner at the specified index.
    - `get_world_corner(self, index: int) -> WorldCoordinate`: Returns the world corner at the specified index.
    - `get_image_corners(self) -> dict`: Returns a dictionary containing the image corners.
    - `get_world_corners(self) -> dict`: Returns a dictionary containing the world corners.

    """
    def __init__(self, camera_angle: int, camera: AzureKinect):
        self.camera_angle = math.radians(camera_angle)
        self.camera_height = 900
        self.camera = camera

        self.origin = None
        self.image_corners = {
            0: None,
            1: None,
            2: None,
            3: None
        }
        self.world_corners = {
            0: None,
            1: None,
            2: None,
            3: None
        }

    def compute_origin(self, depth_image: np.ndarray, coordinates: list) -> ImageCoordinate:
        """
        Compute the origin of the image.

        :param depth_image: The depth image as a numpy array.
        :param coordinates: The list of coordinates for which to compute the origin.
        :return: The computed origin as an ImageCoordinate object.
        """
        if self.origin is None:
            xy_origins = []
            index = 0
            for coordinate in coordinates:
                image_origin, world_corner_coordinate = self.compute_coordinate_origin(
                    depth_image=depth_image,
                    coordinate=coordinate
                )
                xy_origins.append(image_origin)
                self.image_corners[index] = coordinate
                self.world_corners[index] = world_corner_coordinate
                index += 1

            return self.average_origin(self=self, origins=xy_origins)
        else:
            return self.origin

    def compute_coordinate_origin(
        self,
        depth_image: np.ndarray,
        coordinate: ImageCoordinate
    ) -> (ImageCoordinate, WorldCoordinate):
        """
        Compute the coordinate origin using the depth image and the given coordinate.

        :param depth_image: A numpy array representing the depth image.
        :param coordinate: An instance of ImageCoordinate.
        :return: A tuple containing the origin coordinate in the image and the corresponding world coordinate.
        """
        rgb_depth = depth_image[coordinate.get_y(), coordinate.get_x()]
        source_point_2d = k4a_float2_t(
            (coordinate.get_x(), coordinate.get_y())
        )
        position_3d_color: k4a_float3_t = self.camera.convert_2d_to_3d(
            self=self.camera,
            source_point_2d=source_point_2d,
            source_depth=rgb_depth,
            source_camera=K4A_CALIBRATION_TYPE_COLOR,
            target_camera=K4A_CALIBRATION_TYPE_COLOR
        )

        # Update x, y, z based on the camera angle
        x = position_3d_color.xyz.x
        z = position_3d_color.xyz.z
        y = position_3d_color.xyz.y

        y = self.camera_angle * y

        # Update the position_3d_color with the rotated coordinates
        world_coordinate = WorldCoordinate(
            x=int(x),
            y=int(y),
            z=int(z)
        )

        xy_origin = k4a_float3_t((0, 0, z))
        xy_origin_2d = self.camera.convert_3d_to_2d(
            self=self.camera,
            source_point_3d=xy_origin,
            source_camera=K4A_CALIBRATION_TYPE_COLOR,
            target_camera=K4A_CALIBRATION_TYPE_COLOR
        )
        xy_origin_image = ImageCoordinate(
            x=int(xy_origin_2d.xy.x),
            y=int(xy_origin_2d.xy.y)
        )
        return xy_origin_image, world_coordinate

    def get_origin(self) -> ImageCoordinate:
        """
        Return the origin of the image.

        :return: The origin of the image as an ImageCoordinate object.
        """
        return self.origin

    def get_image_corner(self, index: int) -> ImageCoordinate:
        """
        :param index: An integer representing the index of the image corner.
        :return: An ImageCoordinate object that corresponds to the image corner at the specified index.
        """
        return self.image_corners[index]

    def get_world_corner(self, index: int) -> WorldCoordinate:
        """
        :param index: The index of the desired world corner.
        :return: The WorldCoordinate object representing the world corner at the specified index.

        """
        return self.world_corners[index]

    def get_image_corners(self) -> dict:
        """
        Get the corners of the image.

        :return: A dictionary containing the coordinates of the four corners of the image.
        :rtype: dict
        """
        return self.image_corners

    def get_world_corners(self) -> dict:
        """
        Get the world corners.

        :return: A dictionary containing the world corners.
        """
        return self.world_corners

    @staticmethod
    def average_origin(self, origins: list) -> ImageCoordinate:
        """
        Calculate the average origin from a list of ImageCoordinate instances.

        :param self: The class instance.
        :param origins: A list of ImageCoordinate instances.
        :type origins: list
        :return: The average origin as an ImageCoordinate instance.
        :rtype: ImageCoordinate
        """
        x = np.mean([origin.get_x() for origin in origins])
        y = np.mean([origin.get_y() for origin in origins])
        return ImageCoordinate(x=int(x), y=int(y))