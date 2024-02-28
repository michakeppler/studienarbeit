import cv2
import numpy as np
import open3d as o3d


class PointCloudVisualizer:
    def __init__(self, window_name: str = "Point Cloud Visualizer", width: int = 640, height: int = 480):
        self.window_name = window_name
        self.width = width
        self.height = height

        self.started = False

        self.visualizer = o3d.visualization.Visualizer()
        self.visualizer.create_window(
            window_name=window_name,
            width=width,
            height=height,
            visible=True
        )
        self.point_cloud = o3d.geometry.PointCloud()

    def update(self, point_cloud: np.ndarray, color_image: np.ndarray):
        point_cloud = point_cloud.astype(np.float64)
        self.point_cloud.points = o3d.utility.Vector3dVector(point_cloud)

        if color_image is not None:
            color_image = color_image.reshape((-1, 3)) / 255.0
            self.point_cloud.colors = o3d.utility.Vector3dVector(color_image)

        if not self.started:
            self.visualizer.add_geometry(self.point_cloud)
            self.started = True
        else:
            self.visualizer.update_geometry(self.point_cloud)

        self.visualizer.poll_events()
        self.visualizer.update_renderer()