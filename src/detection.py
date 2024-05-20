import cv2
import numpy as np


class GroundLineDetector:
    """
    A class that detects yellow rectangles in an image.

    Parameters:
        area_threshold (int): The minimum area of a valid contour. Default is 1000.

    Attributes:
        area_threshold (int): The minimum area of a valid contour.

    Methods:
        step(color_image: np.ndarray) -> (np.ndarray, list):
            Performs the entire process of detecting yellow rectangles in the given color image.

        find_contours(
            threshold_image: np.ndarray,
            color_image: np.ndarray,
            area_threshold: int = 1000
        ) -> (np.ndarray, list):
            Finds valid contours in the threshold image based on the provided area threshold.

        detect_rectangle(color_image: np.ndarray, contours: list) -> (np.ndarray, list):
            Detects and draws rectangles in the color image based on the provided contours.

    """
    def __init__(self, area_threshold: int = 1000):
        self.area_threshold = area_threshold

    def step(self, color_image: np.ndarray) -> (np.ndarray, list):
        """
        Perform a step in an image processing pipeline.

        :param color_image: A numpy array representing the color image.
        :return: A tuple containing the yellow image with applied filters and the corner coordinates.
        """
        color_image_copy = color_image.copy()

        # Convert color image to HLS
        hls = cv2.cvtColor(src=color_image, code=cv2.COLOR_BGR2HLS)

        # Apply gaussian blur
        hls = cv2.GaussianBlur(hls, (5, 5), 0)

        # Yellow-ish areas in image
        # H value must be appropriate (see HSL color space), e.g. within [40 ... 60]
        # L value can be arbitrary (we want everything between bright and dark yellow), e.g. within [0.0 ... 1.0]
        # S value must be above some threshold (we want at least some saturation), e.g. within [0.00 ... 1.0]
        yellow_lower = np.array([np.round(20 / 2), np.round(0.00 * 255), np.round(0.00 * 255)])
        yellow_upper = np.array([np.round(60 / 2), np.round(1.00 * 255), np.round(1.00 * 255)])
        yellow_mask = cv2.inRange(hls, yellow_lower, yellow_upper)
        yellow_image = cv2.bitwise_and(color_image, color_image, mask=yellow_mask)
        yellow_image = cv2.cvtColor(yellow_image, cv2.COLOR_BGR2GRAY)

        # Apply gaussian blur
        structuring_element = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        yellow_image = cv2.morphologyEx(yellow_image, cv2.MORPH_CLOSE, structuring_element)
        yellow_image = cv2.morphologyEx(yellow_image, cv2.MORPH_OPEN, structuring_element)

        # Apply threshold
        _, yellow_image = cv2.threshold(yellow_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        yellow_image = cv2.medianBlur(yellow_image, 5)

        # Find necessary contours
        color_image_copy, contours = self.find_contours(
            self=self,
            threshold_image=yellow_image,
            color_image=color_image_copy,
            area_threshold=self.area_threshold
        )

        # Detect rectangle
        color_image_copy, corner_coordinates = self.detect_rectangle(
            self=self,
            color_image=color_image_copy,
            contours=contours
        )

        return yellow_image, corner_coordinates

    @staticmethod
    def find_contours(
        self,
        threshold_image: np.ndarray,
        color_image: np.ndarray,
        area_threshold: int = 1000
    ) -> (np.ndarray, list):
        """
        Find contours in an image based on a threshold value and area threshold.

        :param self: The instance of the class.
        :param threshold_image: The thresholded image (numpy array) where contours are to be found.
        :param color_image: The original image (numpy array) where valid contours are to be drawn.
        :param area_threshold: The minimum area threshold for a contour to be considered valid.
        :return: A tuple containing the color image with valid contours drawn and a list of valid contours.

        """
        contours, _ = cv2.findContours(
            image=threshold_image,
            mode=cv2.RETR_EXTERNAL,
            method=cv2.CHAIN_APPROX_SIMPLE
        )
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour=contour)
            if area > area_threshold:
                cv2.drawContours(
                    image=color_image,
                    contours=[contour],
                    contourIdx=-1,
                    color=(0, 255, 0),
                    thickness=2
                )
                valid_contours.append(contour)

        return color_image, valid_contours

    @staticmethod
    def detect_rectangle(self, color_image: np.ndarray, contours: list) -> (np.ndarray, list):
        """
        Detects rectangles from a given image using contours.

        :param self: The current instance of the class.
        :param color_image: A numpy array representing the color image.
        :param contours: A list of contours found in the image.
        :return: A tuple containing the modified color image and a list of corner coordinates.
        """
        corner_coordinates = []
        for contour in contours:
            perimeter = cv2.arcLength(curve=contour, closed=True)
            approx = cv2.approxPolyDP(curve=contour, epsilon=0.009 * perimeter, closed=True)
            if len(approx) == 4:
                cv2.drawContours(
                    image=color_image,
                    contours=[approx],
                    contourIdx=-1,
                    color=(0, 0, 255),
                    thickness=2
                )

                for point in approx:
                    corner_coordinates.append(point[0])

        return color_image, corner_coordinates
