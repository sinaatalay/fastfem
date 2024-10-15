import pytest

import fastfem.mesh as m
import fastfem.mesh.generator as mg
import gmsh


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
    l4 = mg.Line(p2, p1)

    assert l1.tag == l3.tag
    assert l4.tag == -l1.tag
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


def test_negative_duplicate_lines():
    l1 = mg.Line(mg.Point(0.0, 0.0, 0.0), mg.Point(1.0, 0.0, 0.0))
    l2 = mg.Line(mg.Point(1.0, 0.0, 0.0), mg.Point(0.0, 0.0, 0.0))

    assert l1.tag == -l2.tag


def test_invalid_surface_mesh():
    l1 = mg.Line(mg.Point(0.0, 0.0, 0.0), mg.Point(1.0, 0.0, 0.0))
    l2 = mg.Line(mg.Point(1.0, 0.0, 0.0), mg.Point(1.0, 1.0, 0.0))
    l3 = mg.Line(mg.Point(1.0, 1.0, 0.0), mg.Point(0.0, 1.0, 0.0))
    l4 = mg.Line(mg.Point(0.0, 1.0, 0.0), mg.Point(0.0, 0.0, 0.0))

    with pytest.raises(ValueError):
        # Surface cannot be transfinite if lines are not transfinite
        # (number_of_nodes is None for at least one line)
        mg.Surface([l1, l2, l3, l4], transfinite=True)


def test_invalid_surface_mesh_2():
    l1 = mg.Line(mg.Point(0.0, 0.0, 0.0), mg.Point(0.5, 0.0, 0.0), number_of_nodes=10)
    l2 = mg.Line(mg.Point(0.5, 0.0, 0.0), mg.Point(1.0, 0.0, 0.0), number_of_nodes=10)
    l3 = mg.Line(mg.Point(1.0, 0.0, 0.0), mg.Point(1.0, 1.0, 0.0), number_of_nodes=10)
    l4 = mg.Line(mg.Point(1.0, 1.0, 0.0), mg.Point(0.0, 1.0, 0.0), number_of_nodes=10)
    l5 = mg.Line(mg.Point(0.0, 1.0, 0.0), mg.Point(0.0, 0.0, 0.0), number_of_nodes=10)

    with pytest.raises(ValueError):
        # Transfinite surfaces must have 3 or 4 lines
        mg.Surface([l1, l2, l3, l4, l5], transfinite=True)


def test_invalid_surface_mesh_3():
    l1 = mg.Line(mg.Point(0.0, 0.0, 0.0), mg.Point(0.5, 0.0, 0.0))
    l2 = mg.Line(mg.Point(0.5, 0.0, 0.0), mg.Point(1.0, 0.0, 0.0))
    with pytest.raises(ValueError):
        mg.Surface([l1, l2])


def test_surface_with_lines_in_wrong_order():
    l1 = mg.Line(mg.Point(0.0, 0.0, 0.0), mg.Point(1.0, 0.0, 0.0))
    l2 = mg.Line(mg.Point(1.0, 0.0, 0.0), mg.Point(1.0, 1.0, 0.0))
    l3 = mg.Line(mg.Point(1.0, 1.0, 0.0), mg.Point(0.0, 1.0, 0.0))
    l4 = mg.Line(mg.Point(0.0, 1.0, 0.0), mg.Point(0.0, 0.0, 0.0))

    with pytest.raises(ValueError):
        mg.Surface([l3, l2, l1, l4])


def test_surface_with_lines_correct_order_but_wrong_orientation():
    l1 = mg.Line(mg.Point(0.0, 0.0, 0.0), mg.Point(1.0, 0.0, 0.0))
    l2 = mg.Line(mg.Point(1.0, 0.0, 0.0), mg.Point(1.0, 1.0, 0.0))
    l3 = mg.Line(mg.Point(1.0, 1.0, 0.0), mg.Point(0.0, 1.0, 0.0))
    l4 = mg.Line(mg.Point(0.0, 1.0, 0.0), mg.Point(0.0, 0.0, 0.0))

    mg.Surface([l1, l2, -l3, l4])


def test_invalid_domains():
    with pytest.raises(ValueError):
        l1 = mg.Line(
            mg.Point(0.0, 0.0, 0.0),
            mg.Point(1.0, 0.0, 0.0, domain_name="bottom_boundary"),
            domain_name="bottom_boundary",
        )
        mg.Geometry().create_domains()


def test_valid_domains():
    l1 = mg.Line(
        mg.Point(1.0, 0.0, 0.0),
        mg.Point(1.0, 1.0, 0.0),
        domain_name="right_boundary",
    )
    l2 = mg.Line(
        mg.Point(1.0, 1.0, 0.0),
        mg.Point(1.0, 2.0, 0.0),
        domain_name="right_boundary",
    )

    mg.Geometry().create_domains()

    entities = list(gmsh.model.get_entities_for_physical_group(1, 1))
    entities = [int(tag) for tag in entities]

    assert entities == [l1.tag, l2.tag]


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
