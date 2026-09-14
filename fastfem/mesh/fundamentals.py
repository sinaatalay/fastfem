import dataclasses
import pathlib
from typing import Literal

from .generator import (
    Domain,
    Geometry,
    Line,
    Mesh,
    Point,
    Surface,
    TwoDElementType,
    mesh,
)

__all__ = ["Rectangle", "Square", "create_a_rectangle_mesh", "create_a_square_mesh"]


def __dir__() -> list[str]:
    return __all__


@dataclasses.dataclass(kw_only=True)
class Rectangle:
    """Create a 2D mesh of a rectangle.

    Args:
        horizontal_length (float): The horizontal length of the rectangle.
        vertical_length (float): The vertical length of the rectangle.
        nodes_in_horizontal_direction (int, optional): The number of nodes in the
            horizontal direction. Defaults to None.
        nodes_in_vertical_direction (int, optional): The number of nodes in the vertical
            direction. Defaults to None.
        element_type (TwoDElementType, optional): The type of element to use. Defaults
            to "quadrangle".
    """

    horizontal_length: float
    vertical_length: float
    nodes_in_horizontal_direction: int | None = None
    nodes_in_vertical_direction: int | None = None
    element_type: TwoDElementType = "quadrangle"

    def __post_init__(
        self,
    ):
        both_nx_and_ny_are_provided = all(
            [self.nodes_in_horizontal_direction, self.nodes_in_vertical_direction]
        )
        self.transfinite = False
        if both_nx_and_ny_are_provided:
            self.transfinite = True

        self.surface = Surface(
            outer_lines=[
                Line(
                    Point(0, 0, 0),
                    Point(self.horizontal_length, 0, 0),
                    self.nodes_in_horizontal_direction,
                ),
                Line(
                    Point(self.horizontal_length, 0, 0),
                    Point(self.horizontal_length, self.vertical_length, 0),
                    self.nodes_in_vertical_direction,
                ),
                Line(
                    Point(self.horizontal_length, self.vertical_length, 0),
                    Point(0, self.vertical_length, 0),
                    self.nodes_in_horizontal_direction,
                ),
                Line(
                    Point(0, self.vertical_length, 0),
                    Point(0, 0, 0),
                    self.nodes_in_vertical_direction,
                ),
            ],
            transfinite=self.transfinite,
            element_type=self.element_type,
        )
        Geometry().mesh()


@dataclasses.dataclass(kw_only=True)
class Square(Rectangle):
    """Create a 2D mesh of a square.

    Args:
        side_length (float): The side length of the square.
        nodes_in_horizontal_direction (int, optional): The number of nodes in the
            horizontal direction. Defaults to None.
        nodes_in_vertical_direction (int, optional): The number of nodes in the vertical
            direction. Defaults to None.
        element_type (TwoDElementType, optional): The type of element to use. Defaults
            to "quadrangle".
    """

    side_length: float
    horizontal_length: float = dataclasses.field(init=False)
    vertical_length: float = dataclasses.field(init=False)

    def __post_init__(
        self,
    ):
        self.horizontal_length = self.side_length
        self.vertical_length = self.side_length
        super().__post_init__()


RectangleDomainName = Literal[
    "bottom_boundary",
    "right_boundary",
    "top_boundary",
    "left_boundary",
    "surface",
]


class RectangleMesh(Mesh):
    """A class that is identical to the `Mesh` class, but with typing hints for a
    rectangle mesh."""

    def __getitem__(  # type: ignore
        self,
        key: RectangleDomainName,
    ) -> Domain:
        return self[key]


class SquareMesh(RectangleMesh): ...


def create_a_rectangle_mesh(
    horizontal_length: float,
    vertical_length: float,
    nodes_in_horizontal_direction: int | None = None,
    nodes_in_vertical_direction: int | None = None,
    element_type: TwoDElementType = "quadrangle",
    file_name: pathlib.Path | None = None,
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
        domain_name="surface",
    )
    return mesh(file_name=file_name)  # type: ignore


def create_a_square_mesh(
    side_length: float,
    nodes_in_horizontal_direction: int | None = None,
    nodes_in_vertical_direction: int | None = None,
    element_type: TwoDElementType = "quadrangle",
    file_name: pathlib.Path | None = None,
) -> SquareMesh:
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
    )  # type: ignore
