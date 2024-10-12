from typing import Optional, Literal
import gmsh
import numpy as np

ElementType = Literal["triangle", "quadrangle"]


class Rectangle:
    def __init__(
        self,
        width: float,
        height: float,
        nx: Optional[int] = None,
        ny: Optional[int] = None,
        element_type: ElementType = "quadrangle",
    ):
        self.transfinite = False
        if all([nx, ny]):
            self.transfinite = True
            self.nx = nx
            self.ny = ny
        elif any([nx, ny]):
            raise ValueError(
                "Either both nx and ny provided to create a transfinite"
                " mesh or none of them."
            )

        self.width = width
        self.height = height

        self.surface = Surface(
            lines=[
                Line(Point(0, 0, 0), Point(self.width, 0, 0), nx),
                Line(Point(self.width, 0, 0), Point(self.width, self.height, 0), ny),
                Line(Point(self.width, self.height, 0), Point(0, self.height, 0), nx),
                Line(Point(0, self.height, 0), Point(0, 0, 0), ny),
            ],
            transfinite=self.transfinite,
            element_type=element_type,
        )

    def mesh(self):
        self.surface.mesh()
