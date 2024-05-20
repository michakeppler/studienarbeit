import math


class Coordinate:
    """
    Coordinate Class
    ================

    This class represents a coordinate in three-dimensional space.

    Methods
    -------

    __init__(x: float, y: float, z: float)
        Initialize a Coordinate object with the given x, y, and z values.

    get_x() -> float
        Get the x value of the coordinate.

    get_y() -> float
        Get the y value of the coordinate.

    get_z() -> float
        Get the z value of the coordinate.

    get_coordinates() -> Tuple[float, float, float]
        Get the x, y, and z values of the coordinate as a tuple.

    distance_to(coordinate) -> float
        Calculate the Euclidean distance between this coordinate and the given coordinate.

    """
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

    def distance_to(self, coordinate) -> float:
        return math.sqrt(
            (self.x - coordinate.x) ** 2 +
            (self.y - coordinate.y) ** 2 +
            (self.z - coordinate.z) ** 2
        )


class ImageCoordinate(Coordinate):
    """
    The `ImageCoordinate` class is a subclass of `Coordinate` and represents a coordinate point in an image.

    Attributes:
        x (int): The x-coordinate of the image coordinate.
        y (int): The y-coordinate of the image coordinate.

    Methods:
        __init__(x: int, y: int): Initializes a new instance of the `ImageCoordinate` class with the specified x and y coordinates.
        get_x() -> int: Returns the x-coordinate of the image coordinate.
        get_y() -> int: Returns the y-coordinate of the image coordinate.
        get_z() -> int: Returns the z-coordinate of the image coordinate, which is always 0.
        get_coordinates() -> (int, int): Returns a tuple containing the x and y coordinates of the image.

    Usage:

    # Creating an instance of `ImageCoordinate`
    coordinate = ImageCoordinate(x=10, y=20)

    # Getting the x-coordinate
    x = coordinate.get_x()

    # Getting the y-coordinate
    y = coordinate.get_y()

    # Getting the z-coordinate
    z = coordinate.get_z()

    # Getting the coordinates as a tuple
    coordinates = coordinate.get_coordinates()
    """
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
    """
    WorldCoordinate Class

    A class representing a coordinate in the world.

    Attributes:
        x (float): The x-coordinate of the world coordinate.
        y (float): The y-coordinate of the world coordinate.
        z (float): The z-coordinate of the world coordinate.

    Methods:
        __init__(x: float, y: float, z: float): Initializes a new instance of the WorldCoordinate class with the specified coordinates.

    """
    def __init__(self, x: float, y: float, z: float):
        super().__init__(x=x, y=y, z=z)