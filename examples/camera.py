from src.camera import AzureKinect


def test_azure_kinect():
    # Creating an instance of `AzureKinect`
    camera = AzureKinect()

    # Start
    camera.start()

    for i in range(5):
        gyro = camera.get_imu_gyroscope()
        print(f"Gyroscope: {gyro}")

    # Stop
    camera.stop()