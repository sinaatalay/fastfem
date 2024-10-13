import pytest

import fastfem.mesh as m
import fastfem.mesh.generator as mg


def setup_function(function):
    """Setup for test functions"""
    mg.Geometry().clear()


def test_points_database():
    p1 = mg.Point(0.0, 0.0, 0.0)
    p2 = mg.Point(0.0, 0.0, 2.0)
    p3 = mg.Point(0.0, 0.0, 0.0)

    assert p1.tag == p3.tag
    assert mg.Geometry().points == [p1, p2]


def test_lines_database():
    p1 = mg.Point(0.0, 0.0, 0.0)
    p2 = mg.Point(1.0, 0.0, 0.0)
    p3 = mg.Point(1.0, 1.0, 0.0)

    l1 = mg.Line(p1, p2)
    l2 = mg.Line(p1, p3)
    l3 = mg.Line(p1, p2)

    assert l1.tag == l3.tag
    assert mg.Geometry().lines == [l1, l2]


def test_surfaces_database():
    p1 = mg.Point(0.0, 0.0, 0.0)
    p2 = mg.Point(1.0, 0.0, 0.0)
    p3 = mg.Point(1.0, 1.0, 0.0)
    p4 = mg.Point(0.0, 1.0, 0.0)

    l1 = mg.Line(p1, p2)
    l2 = mg.Line(p2, p3)
    l3 = mg.Line(p3, p1)
    l4 = mg.Line(p3, p4)
    l5 = mg.Line(p4, p1)

    s1 = mg.Surface([l1, l2, l3])
    s2 = mg.Surface([l1, l2, l4, l5])
    s3 = mg.Surface([l1, l2, l3])

    assert s1.tag == s3.tag
    assert mg.Geometry().surfaces == [s1, s2]


@pytest.mark.parametrize(
    "nx",
    [
        10,
        None,
    ],
)
@pytest.mark.parametrize(
    "ny",
    [
        10,
        None,
    ],
)
@pytest.mark.parametrize(
    "element_type",
    [
        "triangle",
        "quadrangle",
    ],
)
def test_rectangle_mesh(nx, ny, element_type):
    m.Rectangle(
        horizontal_length=1.0,
        vertical_length=1.0,
        nodes_in_horizontal_direction=nx,
        nodes_in_vertical_direction=ny,
        element_type=element_type,
    )


@pytest.mark.parametrize(
    "nx",
    [
        10,
        None,
    ],
)
@pytest.mark.parametrize(
    "ny",
    [
        10,
        None,
    ],
)
@pytest.mark.parametrize(
    "element_type",
    [
        "triangle",
        "quadrangle",
    ],
)
def test_square_mesh(nx, ny, element_type):
    m.Square(
        side_length=1.0,
        nodes_in_horizontal_direction=nx,
        nodes_in_vertical_direction=ny,
        element_type=element_type,
    )
