import cv2
import numpy as np

from src.calibration import GroundLineCalibrator
from src.camera import AzureKinect
from src.coordinate import ImageCoordinate
from src.detection import GroundLineDetector
from src.origin import Origin


class Solver:
    def __init__(
        self,
        camera: AzureKinect,
        gld_area_threshold: int = 350000,
        glc_calibration_samples: int = 5
    ):
        self.camera = camera

        # Solvers internal class instances
        self.gl_detector = GroundLineDetector(
            area_threshold=gld_area_threshold
        )
        self.gl_calibrator = GroundLineCalibrator(
            calibration_samples=glc_calibration_samples
        )
        self.origin = Origin(
            camera_angle=85,
            camera=self.camera
        )

    def start(self):
        self.camera.start()

    def solve(self):
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

                    # Image drawing
                    # Draw corners
                    for corner_index in range(4):
                        image_corner = self.origin.get_image_corner(index=corner_index)
                        world_corner = self.origin.get_world_corner(index=corner_index)
                        cv2.circle(color_image, image_corner.get_coordinates(), 5, (255, 0, 0), -1)
                        cv2.putText(
                            img=color_image,
                            text=f"Corner (Position: x={round(world_corner.get_x())}mm, y={round(world_corner.get_y())}mm, z={round(world_corner.get_z())}mm)",
                            org=(image_corner.get_x() + 10, image_corner.get_y() + 10),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5,
                            color=(255, 0, 0),
                            thickness=2
                        )

                    # Draw origin
                    cv2.circle(color_image, origin.get_coordinates(), 5, (0, 0, 255), -1)
                    cv2.putText(
                        img=color_image,
                        text="Origin",
                        org=(origin.get_x() + 10, origin.get_y() + 10),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        color=(0, 0, 255),
                        thickness=2
                    )

                cv2.imshow("Image", color_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def stop(self):
        self.camera.stop()
