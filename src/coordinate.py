
class Coordinate:
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z

    def get_x(self) -> float:
        return self.x

    def get_y(self) -> float:
        return self.y

    def get_z(self) -> float:
        return self.z

    def get_coordinates(self) -> (float, float, float):
        return self.x, self.y, self.z


class ImageCoordinate(Coordinate):
    def __init__(self, x: int, y: int):
        super().__init__(x=x, y=y, z=0)

    def get_x(self) -> int:
        return int(self.x)

    def get_y(self) -> int:
        return int(self.y)

    def get_z(self) -> int:
        return 0

    def get_coordinates(self) -> (int, int):
        return int(self.x), int(self.y)


class WorldCoordinate(Coordinate):
    def __init__(self, x: float, y: float, z: float):
        super().__init__(x=x, y=y, z=z)