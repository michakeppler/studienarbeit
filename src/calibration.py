import cv2
import numpy as np
import mediapipe as mp
import pykinect_azure as pykinect

from camera import AzureKinect


class AzureKinectCalibration:
    def __init__(self):
        # Create camera object for the Azure Kinect
        self.camera = AzureKinect(
            color_format=pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32,
            color_resolution=pykinect.K4A_COLOR_RESOLUTION_720P,
            depth_mode=pykinect.K4A_DEPTH_MODE_NFOV_UNBINNED,
            track_body=False
        )
        self.camera.start()

        # Create window
        self.window_name = "Calibration"
        self.width = 640
        self.height = 480
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        # Create mediapipe hands object
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def update(self) -> bool:
        # Get color image
        is_valid_ci, color_image = self.camera.get_color_image()
        is_valid_depth, depth_image = self.camera.get_depth_image()

        if is_valid_ci and is_valid_depth:

            # Detect hands
            hands_result = self.hands.process(color_image)
            hands_landmarks = hands_result.multi_hand_landmarks
            if hands_landmarks:
                depth_values = []
                for hand_landmarks in hands_landmarks:
                    x_max = 0
                    y_max = 0
                    x_min = self.width
                    y_min = self.height
                    for landmark in hand_landmarks.landmark:
                        x, y = int(landmark.x * self.width), int(landmark.y * self.height)
                        if x > x_max:
                            x_max = x
                        if x < x_min:
                            x_min = x
                        if y > y_max:
                            y_max = y
                        if y < y_min:
                            y_min = y

                        # Get the depth value of the center of the hand
                        depth = depth_image[int((y_max + y_min) / 2), int((x_max + x_min) / 2)]
                        depth_values.append(depth)

                    # Calculate the average depth value
                    avg_depth = np.mean(depth_values)
                    print("Distance of hand to camera: ", avg_depth / 10, "cm")

            # Show the color image
            cv2.imshow(self.window_name, color_image)

            # Wait for key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                return False

        return True


if __name__ == '__main__':
    calibration = AzureKinectCalibration()
    while calibration.update():
        pass
    cv2.destroyAllWindows()
    calibration.camera.stop()