import cv2
import numpy as np
import mediapipe as mp

from src.calibration import GroundLineCalibrator
from src.camera import AzureKinect
from src.coordinate import ImageCoordinate, WorldCoordinate
from src.detection import GroundLineDetector
from src.origin import Origin


class Solver:
    """
    The Solver class is responsible for solving a specific task using the AzureKinect camera and various other components.

    """
    def __init__(
        self,
        camera: AzureKinect,
        gld_area_threshold: int = 10000,
        glc_calibration_samples: int = 100
    ):
        """
        :param camera: An instance of the AzureKinect class representing the camera used for tracking.
        :param gld_area_threshold: An integer representing the area threshold for the GroundLineDetector.
        :param glc_calibration_samples: An integer representing the number of calibration samples for the GroundLineCalibrator.

        """
        self.camera = camera

        # Solvers internal class instances
        self.gl_detector = GroundLineDetector(
            area_threshold=gld_area_threshold
        )
        self.gl_calibrator = GroundLineCalibrator(
            calibration_samples=glc_calibration_samples
        )
        self.origin = Origin(
            camera_angle=42,
            camera=self.camera
        )

        # Hand tracking
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def start(self):
        """
        Starts the camera.

        :return: None
        """
        self.camera.start()

    def solve(self):
        """
        This method `solve` is responsible for processing and displaying the color image and depth image obtained from the camera. It performs the following steps:

        1. Retrieves the color image and depth image from the camera.
        2. If both images are valid:
            a. Detects the ground line and obtains the corner coordinates.
            b. Calibrates the ground line if it is not already calibrated.
            c. If the ground line is calibrated:
                - Converts the color image to a numpy array.
                - Computes the origin coordinates based on the transformed depth image and calibrated corner coordinates.
                - Draws 3D distance lines between the corners.
                - Displays the distance between each pair of corners.
                - Draws the corners on the color image.
                - Displays the world coordinates for each corner.
                - Draws the origin point.
                - Performs hand tracking on the color image using a hand tracking model.
                - Draws circles on the detected hand landmarks.
                - Estimates the height of the hand based on the detected landmarks and the transformed depth image.
                - Displays the estimated hand height.
            d. Displays the color image.
        3. Checks for a key press event to exit the loop.

        :return: None
        """
        while True:
            is_valid_image, color_image = self.camera.get_color_image()
            is_valid_depth, transformed_depth_image = self.camera.get_transformed_depth_image()

            if is_valid_image and is_valid_depth:
                # Ground line detection
                result_image, corner_coordinates = self.gl_detector.step(
                    color_image=color_image
                )
                self.gl_calibrator.step(coordinates=corner_coordinates)

                if not self.gl_calibrator.is_calibrated():
                    color_image = result_image
                else:
                    color_image: np.ndarray = np.ascontiguousarray(color_image)
                    origin: ImageCoordinate = self.origin.compute_origin(
                        depth_image=transformed_depth_image,
                        coordinates=self.gl_calibrator.get_calibrated_coordinates()
                    )

                    # Draw 3d distance between corners
                    for i in range(4):
                        for j in range(i + 1, 4):
                            cv2.line(
                                color_image,
                                self.origin.get_image_corner(index=i).get_coordinates(),
                                self.origin.get_image_corner(index=j).get_coordinates(),
                                (0, 255, 0),
                                2
                            )
                            cv2.putText(
                                img=color_image,
                                text=f"{round(self.origin.get_world_corner(index=i).distance_to(self.origin.get_world_corner(index=j)))}mm",
                                org=(
                                    int((self.origin.get_image_corner(index=i).get_x() + self.origin.get_image_corner(
                                        index=j).get_x()) / 2),
                                    int((self.origin.get_image_corner(index=i).get_y() + self.origin.get_image_corner(
                                        index=j).get_y()) / 2)
                                ),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.5,
                                color=(0, 0, 0),
                                thickness=2
                            )

                    # Draw corners
                    for corner_index in range(4):
                        image_corner = self.origin.get_image_corner(index=corner_index)
                        cv2.circle(
                            img=color_image,
                            center=image_corner.get_coordinates(),
                            radius=5,
                            color=(0, 0, 0),
                            thickness=-1
                        )
                        # Draw world coordinates
                        cv2.putText(
                            img=color_image,
                            text=f"{self.origin.get_world_corner(index=corner_index).get_coordinates()}",
                            org=(image_corner.get_x(), image_corner.get_y()),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5,
                            color=(0, 0, 0),
                            thickness=2
                        )

                    # Draw origin
                    cv2.circle(
                        img=color_image,
                        center=origin.get_coordinates(),
                        radius=5,
                        color=(255, 0, 0),
                        thickness=-1
                    )

                    # Hand tracking
                    image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                    results = self.hands.process(image)

                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            for point in hand_landmarks.landmark:
                                height, width, _ = color_image.shape
                                cx, cy = int(point.x * width), int(point.y * height)
                                cv2.circle(color_image, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

                    # Estimate height of the hand
                    hand_height = 0
                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            for point in hand_landmarks.landmark:
                                height, width, _ = color_image.shape
                                cx, cy = int(point.x * width), int(point.y * height)
                                hand_height = self.origin.estimate_height(
                                    x=cx,
                                    y=cy,
                                    depth_image=transformed_depth_image
                                )

                    # Draw hand height
                    cv2.putText(
                        img=color_image,
                        text=f"{hand_height}mm",
                        org=(50, 50),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1,
                        color=(0, 0, 0),
                        thickness=2
                    )

                cv2.imshow("Image", color_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def stop(self):
        """
        Stop method.

        :return: None
        """
        self.camera.stop()
