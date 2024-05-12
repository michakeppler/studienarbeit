import math

import cv2
import numpy as np
from pykinect_azure import k4a_float2_t, k4a_float3_t, K4A_CALIBRATION_TYPE_COLOR, K4A_CALIBRATION_TYPE_DEPTH

from src.camera import AzureKinect


def get_color_interval(color: str) -> tuple:
    if color not in ["red", "green", "blue", "yellow", "black"]:
        raise ValueError("Please provide a valid color. Colors can be red, green, blue, yellow or black.")

    if color == "red":
        lower_bound = (0, 0, 100)
        upper_bound = (100, 100, 255)
    elif color == "green":
        lower_bound = (0, 100, 0)
        upper_bound = (100, 255, 100)
    elif color == "blue":
        lower_bound = (100, 0, 0)
        upper_bound = (255, 100, 100)
    elif color == "yellow":
        lower_bound = (0, 0, 0)
        upper_bound = (179, 45, 96)
    elif color == "black":
        lower_bound = (0, 0, 0)
        upper_bound = (100, 100, 100)

    return lower_bound, upper_bound


class Solver:
    def __init__(
        self,
        camera: AzureKinect,
        max_depth: float,
    ):
        self.camera = camera
        self.max_depth = max_depth

    def start(self):
        self.camera.start()

    def process_ground_line_detection(
        self,
        color_image: np.ndarray,
        area_threshold: int = 1000
    ):
        color_image_copy = color_image.copy()

        lab_image = cv2.cvtColor(color_image_copy, cv2.COLOR_BGR2LAB)

        ret, threshold_image = cv2.threshold(lab_image[:, :, 2], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Draw contours
        contours, _ = cv2.findContours(threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > area_threshold:
                cv2.drawContours(color_image_copy, [contour], -1, (0, 255, 0), 2)

        # Check if contour is a rectangle
        corner_coordinates = []
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.018 * perimeter, True)
            if len(approx) == 4:
                cv2.drawContours(color_image_copy, [approx], -1, (0, 0, 255), 2)

                for point in approx:
                    corner_coordinates.append(point[0])

        return color_image_copy, corner_coordinates

    def solve(self):
        n_calibration_time = 2
        calibration_counter = 0
        calibration_coordinates = []
        while True:
            is_valid_image, color_image = self.camera.get_color_image()
            # is_valid_depth, depth_image = self.camera.get_depth_image()
            is_valid_depth, transformed_depth_image = self.camera.get_transformed_depth_image()

            if is_valid_image and is_valid_depth:
                # Process ground line detection
                result_image, corner_coordinates = self.process_ground_line_detection(
                    color_image=color_image,
                    area_threshold=10000
                )
                color_image = np.ascontiguousarray(color_image)

                if len(corner_coordinates) == 4 and calibration_counter < n_calibration_time:
                    print(corner_coordinates)
                    coord_array = np.array(corner_coordinates)
                    calibration_coordinates.append(coord_array)
                    calibration_counter += 1
                    print(f"Calibration Counter: {calibration_counter}/{n_calibration_time}")
                    color_image = result_image

                if calibration_counter >= n_calibration_time:
                    # Max of 1000 calibration points
                    coordinates = np.stack(calibration_coordinates, axis=0)

                    # Compute percentile
                    coordinates = np.percentile(coordinates, 90, axis=0)

                    coordinates = np.round(coordinates).astype(np.int32)
                    if len(coordinates.shape) == 3:
                        coordinates = np.squeeze(coordinates, axis=0)

                    display_coordinates = (
                        np.array(coordinates[0]),
                        np.array(coordinates[1]),
                        np.array(coordinates[2]),
                        np.array(coordinates[3])
                    )

                    kinect_origin = []
                    for coordinate in display_coordinates:
                        cv2.circle(color_image, tuple(coordinate), 5, (0, 0, 255), -1)
                        angle = math.radians(-56)
                        rgb_depth = transformed_depth_image[coordinate[1], coordinate[0]]
                        source_point_2d = k4a_float2_t(
                            (coordinate[0], coordinate[1])
                        )
                        position_3d_color = self.camera.convert_2d_to_3d(
                            self=self.camera,
                            source_point_2d=source_point_2d,
                            source_depth=rgb_depth,
                            source_camera=K4A_CALIBRATION_TYPE_COLOR,
                            target_camera=K4A_CALIBRATION_TYPE_DEPTH
                        )
                        position_3d_color.xyz.x = position_3d_color.xyz.x
                        position_3d_color.xyz.y = position_3d_color.xyz.y
                        position_3d_color.xyz.z = position_3d_color.xyz.z

                        text_str_x = float(position_3d_color.xyz.x) / 10
                        text_str_y = float(position_3d_color.xyz.y) / 10
                        text_str_z = float(position_3d_color.xyz.z) / 10
                        text_string = "Position: x={:.2f}cm, y={:.2f}cm, z={:.2f}cm".format(text_str_x, text_str_y, text_str_z)
                        cv2.putText(color_image, f"{text_string}", tuple(coordinate), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

                        xy_origin = k4a_float3_t(
                            (0, 0, position_3d_color.xyz.z)
                        )
                        xy_origin_2d = self.camera.convert_3d_to_2d(
                            self=self.camera,
                            source_point_3d=xy_origin,
                            source_camera=K4A_CALIBRATION_TYPE_DEPTH,
                            target_camera=K4A_CALIBRATION_TYPE_COLOR
                        )
                        kinect_origin.append([
                            xy_origin_2d.xy.x,
                            xy_origin_2d.xy.y
                        ])

                    kinect_origin = np.array(kinect_origin)
                    kinect_origin = np.average(kinect_origin, axis=0)
                    cv2.circle(color_image, tuple(kinect_origin), 5, (0, 0, 255), -1)

                cv2.imshow("Image", color_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def stop(self):
        self.camera.stop()