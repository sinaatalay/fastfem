import dataclasses
from typing import Optional

from .generator import Geometry, Line, Point, Surface, TwoDElementType

__all__ = ["Rectangle", "Square"]


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
    nodes_in_horizontal_direction: Optional[int] = None
    nodes_in_vertical_direction: Optional[int] = None
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
            lines=[
                Line(
                    Point(0, 0, 0),
                    Point(self.horizontal_length, 0, 0),
                    self.nodes_in_horizontal_direction,
                    domain_name="bottom_boundary",
                ),
                Line(
                    Point(self.horizontal_length, 0, 0),
                    Point(self.horizontal_length, self.vertical_length, 0),
                    self.nodes_in_vertical_direction,
                    domain_name="right_boundary",
                ),
                Line(
                    Point(self.horizontal_length, self.vertical_length, 0),
                    Point(0, self.vertical_length, 0),
                    self.nodes_in_horizontal_direction,
                    domain_name="top_boundary",
                ),
                Line(
                    Point(0, self.vertical_length, 0),
                    Point(0, 0, 0),
                    self.nodes_in_vertical_direction,
                    domain_name="left_boundary",
                ),
            ],
            transfinite=self.transfinite,
            element_type=self.element_type,
            domain_name="domain",
        )
        self.mesh = Geometry().mesh()


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
