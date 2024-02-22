import mediapipe as mp
import pykinect_azure as pykinect
import cv2


if __name__ == '__main__':
    pykinect.initialize_libraries(track_body=False)

    device_configuration = pykinect.default_configuration
    device_configuration.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
    device_configuration.color_resolution = pykinect.K4A_COLOR_RESOLUTION_720P
    device_configuration.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED
    print(device_configuration)

    # Start device
    device = pykinect.start_device(config=device_configuration)

    cv2.namedWindow('Depth Image', cv2.WINDOW_NORMAL)

    # Init mediapipe
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    while True:

        # Get capture
        capture = device.update()

        # Depth Image
        # ret, depth_image = capture.get_colored_depth_image()

        # Colored Image
        ret, color_image = capture.get_color_image()
        if color_image is not None:
            # Infrared Image
            # ret, ir_image = capture.get_ir_image()

            color_image = color_image[..., :3]
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

            # Detect hands, face and pose
            hands_result = hands.process(color_image)
            face_result = face_detection.process(color_image)
            pose_result = pose.process(color_image)

            if hands_result.multi_hand_landmarks:
                for hand_landmarks in hands_result.multi_hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(color_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if face_result.detections:
                for detection in face_result.detections:
                    mp.solutions.drawing_utils.draw_detection(color_image, detection)

            if pose_result.pose_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(color_image, pose_result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Draw the hand landmarks
            if hands_result.multi_hand_landmarks:
                for hand_landmarks in hands_result.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image=color_image,
                        landmark_list=hand_landmarks,
                        connections=mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                        connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style())

            # Draw the face landmarks
            if face_result.detections:
                for detection in face_result.detections:
                    mp_drawing.draw_detection(
                        image=color_image,
                        detection=detection,
                        keypoint_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

            # Draw the pose landmarks
            if pose_result.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image=color_image,
                    landmark_list=pose_result.pose_landmarks,
                    connections=mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

            # Zeigen Sie das Bild an
            cv2.imshow('MediaPipe with Azure Kinect', color_image)

            # Press q key to stop
            if cv2.waitKey(1) == ord('q'):
                break

