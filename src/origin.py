import math

import numpy as np

from pykinect_azure import k4a_float2_t, k4a_float3_t, K4A_CALIBRATION_TYPE_COLOR, K4A_CALIBRATION_TYPE_DEPTH

from src.camera import AzureKinect
from src.coordinate import ImageCoordinate, WorldCoordinate


class Origin:
    def __init__(self, camera_angle: int, camera: AzureKinect):
        self.camera_angle = camera_angle
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
        rgb_depth = depth_image[coordinate.get_y(), coordinate.get_x()]
        source_point_2d = k4a_float2_t(
            (coordinate.get_x(), coordinate.get_y())
        )
        position_3d_color: k4a_float3_t = self.camera.convert_2d_to_3d(
            self=self.camera,
            source_point_2d=source_point_2d,
            source_depth=rgb_depth,
            source_camera=K4A_CALIBRATION_TYPE_COLOR,
            target_camera=K4A_CALIBRATION_TYPE_DEPTH
        )

        # Update x, y, z based on the camera angle
        x = position_3d_color.xyz.x
        y = position_3d_color.xyz.y
        z = position_3d_color.xyz.z

        # Convert camera angle from degrees to radians
        angle_rad = np.radians(self.camera_angle)

        # Rotate the coordinates around the y-axis by the camera angle
        x_rot = x * np.cos(angle_rad) + z * np.sin(angle_rad)
        z_rot = -x * np.sin(angle_rad) + z * np.cos(angle_rad)

        # Update the position_3d_color with the rotated coordinates
        world_coordinate = WorldCoordinate(
            x=x_rot,
            y=y,
            z=z_rot
        )

        xy_origin = k4a_float3_t((0, 0, world_coordinate.get_z()))
        xy_origin_2d = self.camera.convert_3d_to_2d(
            self=self.camera,
            source_point_3d=xy_origin,
            source_camera=K4A_CALIBRATION_TYPE_DEPTH,
            target_camera=K4A_CALIBRATION_TYPE_COLOR
        )
        xy_origin_image = ImageCoordinate(
            x=int(xy_origin_2d.xy.x),
            y=int(xy_origin_2d.xy.y)
        )
        return xy_origin_image, world_coordinate

    def get_origin(self) -> ImageCoordinate:
        return self.origin

    def get_image_corner(self, index: int) -> ImageCoordinate:
        return self.image_corners[index]

    def get_world_corner(self, index: int) -> WorldCoordinate:
        return self.world_corners[index]

    def get_image_corners(self) -> dict:
        return self.image_corners

    def get_world_corners(self) -> dict:
        return self.world_corners

    @staticmethod
    def average_origin(self, origins: list) -> ImageCoordinate:
        x = np.mean([origin.get_x() for origin in origins])
        y = np.mean([origin.get_y() for origin in origins])
        return ImageCoordinate(x=int(x), y=int(y))