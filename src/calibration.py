import numpy as np

from src.coordinate import ImageCoordinate


class GroundLineCalibrator:
    def __init__(self, calibration_samples: int = 100):
        self.calibration_samples = calibration_samples

        self.calibration_iteration = 0
        self.calibration_coordinates = []

        self.calibrated = False
        self.calibrated_coordinates = None

    def step(self, coordinates: list):
        if len(coordinates) == 4 and not self.calibrated:
            coordinates = np.array(coordinates)

            self.calibration_coordinates.append(coordinates)
            self.calibration_iteration += 1
            print(f"Calibration iteration: {self.calibration_iteration}/{self.calibration_samples}")

        if self.calibration_iteration >= self.calibration_samples:
            if self.calibrated_coordinates is None:
                stacked_coordinates = np.stack(self.calibration_coordinates, axis=0)
                filtered_coordinates = np.percentile(stacked_coordinates, 90, axis=0)
                rounded_coordinates = np.round(filtered_coordinates).astype(np.int32)
                if len(rounded_coordinates) == 3:
                    rounded_coordinates = np.squeeze(rounded_coordinates, axis=0)

                self.calibrated = True
                self.calibrated_coordinates = (
                    ImageCoordinate(x=int(rounded_coordinates[0][0]), y=int(rounded_coordinates[0][1])),
                    ImageCoordinate(x=int(rounded_coordinates[1][0]), y=int(rounded_coordinates[1][1])),
                    ImageCoordinate(x=int(rounded_coordinates[2][0]), y=int(rounded_coordinates[2][1])),
                    ImageCoordinate(x=int(rounded_coordinates[3][0]), y=int(rounded_coordinates[3][1]))
                )

    def is_calibrated(self) -> bool:
        return self.calibrated

    def get_calibrated_coordinates(self) -> (ImageCoordinate, ImageCoordinate, ImageCoordinate, ImageCoordinate):
        return self.calibrated_coordinates