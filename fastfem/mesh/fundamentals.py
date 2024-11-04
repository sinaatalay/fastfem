import pathlib
from typing import Final, Literal, Optional

from .generator import Domain, Line, Mesh, Point, Surface, TwoDElementType, mesh

__all__ = ["create_a_rectangle_mesh", "create_a_square_mesh"]


def __dir__() -> list[str]:
    return __all__


RectangleDomainName = Literal[
    "bottom_boundary",
    "right_boundary",
    "top_boundary",
    "left_boundary",
    "rectangle",
]

SquareDomainName = Literal[
    "bottom_boundary",
    "right_boundary",
    "top_boundary",
    "left_boundary",
    "square",
]


class RectangleMesh(Mesh):
    """A class that is identical to the `Mesh` class, but with typing hints for a
    rectangle mesh."""

    def __getitem__(  # type: ignore
        self,
        key: RectangleDomainName,
    ) -> Domain:
        return self[key]


class SquareMesh(Mesh):
    """A class that is identical to the `Mesh` class, but with typing hints for a
    square mesh."""

    def __getitem__(  # type: ignore
        self,
        key: SquareDomainName,
    ) -> Domain:
        return self[key]


def create_a_rectangle_mesh(
    horizontal_length: float,
    vertical_length: float,
    nodes_in_horizontal_direction: Optional[int] = None,
    nodes_in_vertical_direction: Optional[int] = None,
    element_type: TwoDElementType = "quadrangle",
    file_name: Optional[pathlib.Path] = None,
) -> RectangleMesh:
    """Create a 2D mesh of a rectangle.

    Args:
        horizontal_length: The horizontal length of the rectangle.
        vertical_length: The vertical length of the rectangle.
        nodes_in_horizontal_direction: The number of nodes in the horizontal direction.
            Defaults to None.
        nodes_in_vertical_direction: The number of nodes in the vertical direction.
            Defaults to None.
        element_type: The type of element to use. Defaults to "quadrangle".
        file_name: The file name to save the mesh to. Defaults to None.

    Returns:
        The mesh of the rectangle.
    """

    both_nx_and_ny_are_provided = all(
        [nodes_in_horizontal_direction, nodes_in_vertical_direction]
    )
    transfinite = False
    if both_nx_and_ny_are_provided:
        transfinite = True

    Surface(
        outer_lines=[
            Line(
                Point(0, 0, 0),
                Point(horizontal_length, 0, 0),
                nodes_in_horizontal_direction,
                domain_name="bottom_boundary",
            ),
            Line(
                Point(horizontal_length, 0, 0),
                Point(horizontal_length, vertical_length, 0),
                nodes_in_vertical_direction,
                domain_name="right_boundary",
            ),
            Line(
                Point(horizontal_length, vertical_length, 0),
                Point(0, vertical_length, 0),
                nodes_in_horizontal_direction,
                domain_name="top_boundary",
            ),
            Line(
                Point(0, vertical_length, 0),
                Point(0, 0, 0),
                nodes_in_vertical_direction,
                domain_name="left_boundary",
            ),
        ],
        transfinite=transfinite,
        element_type=element_type,
        domain_name="rectangle",
    )

    return mesh(file_name=file_name)


def create_a_square_mesh(
    side_length: float,
    nodes_in_horizontal_direction: Optional[int] = None,
    nodes_in_vertical_direction: Optional[int] = None,
    element_type: TwoDElementType = "quadrangle",
    file_name: Optional[pathlib.Path] = None,
):
    """Create a 2D mesh of a square.

    Args:
        side_length: The side length of the square.
        nodes_in_horizontal_direction: The number of nodes in the horizontal direction.
        Defaults to None.
        nodes_in_vertical_direction: The number of nodes in the vertical direction.
        Defaults to None.
        element_type: The type of element to use. Defaults to "quadrangle".
    """
    horizontal_length = side_length
    vertical_length = side_length
    return create_a_rectangle_mesh(
        horizontal_length,
        vertical_length,
        nodes_in_horizontal_direction,
        nodes_in_vertical_direction,
        element_type,
        file_name,
    )
