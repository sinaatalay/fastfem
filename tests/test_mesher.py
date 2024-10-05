import fastfem
from fastfem.mesh.generator.fundamentals import _Database
import pytest

from fastfem.mesh.generator.fundamentals import Point, Line


def setup_function(function):
    """Setup for test functions"""
    _Database().clear()


def test_points_database():
    p1 = Point(0.0, 0.0, 0.0)
    p2 = Point(0.0, 0.0, 0.0)

    assert p1.tag == p2.tag


def test_lines_database():
    p1 = Point(0.0, 0.0, 0.0)
    p2 = Point(1.0, 0.0, 0.0)

    l1 = Line(p1, p2)
    l2 = Line(p1, p2)

    assert l1.tag == l2.tag


@pytest.mark.parametrize(
    "nx, ny, element_type",
    [
        (10, 10, "triangle"),
        (10, 10, "quadrangle"),
        (None, None, "triangle"),
        (None, None, "quadrangle"),
    ],
)
def test_rectangle_mesh(nx, ny, element_type):
    rectangle = fastfem.mesh.generator.Rectangle(
        width=1.0, height=1.0, nx=nx, ny=ny, element_type=element_type
    )
    rectangle.mesh()
