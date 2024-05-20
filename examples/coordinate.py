from src.coordinate import Coordinate, ImageCoordinate, WorldCoordinate


def test_coordinate():
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

    assert x == 10
    assert y == 20
    assert z == 0
    assert coordinates == (10, 20)


def test_world_coordinate():
    # Creating an instance of `WorldCoordinate`
    coordinate = WorldCoordinate(x=10, y=20, z=30)

    # Getting the x-coordinate
    x = coordinate.get_x()

    # Getting the y-coordinate
    y = coordinate.get_y()

    # Getting the z-coordinate
    z = coordinate.get_z()

    # Getting the coordinates as a tuple
    coordinates = coordinate.get_coordinates()

    assert x == 10
    assert y == 20
    assert z == 30
    assert coordinates == (10, 20, 30)