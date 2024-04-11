import cv2
import numpy as np

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
        ground_truth_line_color: str = "red",
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
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            if len(approx) == 4:
                cv2.drawContours(color_image_copy, [approx], -1, (0, 0, 255), 2)

                for point in approx:
                    corner_coordinates.append(point[0])

        return color_image_copy, corner_coordinates

    def solve(self):
        n_calibration_time = 50
        calibration_counter = 0
        calibration_coordinates = []
        while True:
            is_valid_image, color_image = self.camera.get_color_image()
            is_valid_depth, depth_image = self.camera.get_depth_image()

            if is_valid_image and is_valid_depth:
                # Process ground line detection
                result_image, corner_coordinates = self.process_ground_line_detection(
                    color_image=color_image,
                    ground_truth_line_color="red",
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

                    print(coordinates)

                    display_coordinates = (
                        np.array(coordinates[0]),
                        np.array(coordinates[1]),
                        np.array(coordinates[2]),
                        np.array(coordinates[3])
                    )
                    for coordinate in display_coordinates:
                        cv2.circle(color_image, tuple(coordinate), 5, (0, 0, 255), -1)

                        distance = depth_image[coordinate[1], coordinate[0]]
                        cv2.putText(color_image, f"{distance:.2f}m", tuple(coordinate), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                    # Draw dotted lines between the points
                    cv2.line(color_image, tuple(display_coordinates[0]), tuple(display_coordinates[1]), (0, 0, 255), 2)
                    cv2.line(color_image, tuple(display_coordinates[1]), tuple(display_coordinates[2]), (0, 0, 255), 2)
                    cv2.line(color_image, tuple(display_coordinates[2]), tuple(display_coordinates[3]), (0, 0, 255), 2)
                    cv2.line(color_image, tuple(display_coordinates[3]), tuple(display_coordinates[0]), (0, 0, 255), 2)

                cv2.imshow("Image", color_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def stop(self):
        self.camera.stop()