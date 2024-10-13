"""
The `fastfem.mesh.generator` module is a wrapper around Gmsh to create meshes by
defining points with coordinates, lines with points, and surfaces with lines. The
wrapper is designed to be used in a more object-oriented way to create meshes easily. It
also avoids creating duplicate entities in the Gmsh model by checking if the entity
already exists.
"""

from typing import Literal, Optional

import gmsh
import numpy as np

__all__ = ["TwoDElementType", "Point", "Line", "Surface", "Geometry"]


def __dir__() -> list[str]:
    return __all__


# Gmsh needs to be initialized before creating any entities:
gmsh.initialize()

# The available 2D element types:
TwoDElementType = Literal["triangle", "quadrangle"]


class Geometry:
    """This is a singleton class to store all the created geometric entities. It keeps
    track of all the points, lines, and surfaces created. If the user tries to create a
    new entity that already exists, this class will avoid the duplication of the entity,
    which is not done by Gmsh.

    Attributes:
        points (list[Point]): All the created points
        lines (list[Line]): All the created lines
        surfaces (list[Surface]): All the created surfaces

    Methods:
        clear: Clear all the geometric entities.
        mesh: Generate the mesh using Gmsh.
    """

    _instance = None

    _points: list["Point"] = []
    _lines: list["Line"] = []
    _surfaces: list["Surface"] = []

    _points_coordinates: np.ndarray = np.empty((0, 3), dtype=np.float64)
    _line_point_tags: np.ndarray = np.empty((0, 2), dtype=np.int64)
    _surface_line_tags: list[np.ndarray] = []

    def __new__(cls):
        """If an instance of the class already exists, return it. Otherwise, create a
        new instance and return it. This is the Singleton design pattern."""
        if cls._instance is None:
            cls._instance = super(Geometry, cls).__new__(cls)
        return cls._instance

    @classmethod
    def _does_it_already_exist(
        cls, entity: "Point | Line | Surface"
    ) -> Optional["Point | Line | Surface"]:
        """Check if the entity already exists. If it does, return the existing entity.
        Otherwise, return None.

        Args:
            entity (Point | Line | Surface): The entity to check if it already exists.

        Returns:
            Optional[Point | Line | Surface]: The existing entity if it already exists,
            otherwise None.
        """
        if isinstance(entity, Point):
            comparison = np.isclose(cls._points_coordinates, entity.coordinates).all(
                axis=1
            )
            entities = cls._points
        elif isinstance(entity, Line):
            comparison = (
                cls._line_point_tags == np.array([point.tag for point in entity.points])
            ).all(axis=1)
            entities = cls._lines
        elif isinstance(entity, Surface):
            entity_lines = np.array([line.tag for line in entity.lines])
            comparison = np.array(
                [
                    np.array_equal(line_tag, entity_lines)
                    for line_tag in cls._surface_line_tags
                ]
            )
            entities = cls._surfaces

        if comparison.any():
            where_true = np.where(comparison)[0][0]
            return entities[where_true]

    @classmethod
    def _add(
        cls, entity: "Point | Line | Surface"
    ) -> Optional["Point | Line | Surface"]:
        """Add a geometric entity. If the entity already exists, return the existing
        entity. Otherwise, add the entity and return None.

        Args:
            entity (Point | Line | Surface): The entity to be added

        Returns:
            Optional[Point | Line | Surface]: The existing entity if it already exists,
            otherwise the new entity
        """
        existing_entity = cls._does_it_already_exist(entity)
        if existing_entity:
            return existing_entity

        if isinstance(entity, Point):
            cls._points_coordinates = np.vstack(
                [cls._points_coordinates, entity.coordinates]
            )
            cls._points.append(entity)
        elif isinstance(entity, Line):
            cls._line_point_tags = np.vstack(
                [
                    cls._line_point_tags,
                    [point.tag for point in entity.points],
                ]
            )
            cls._lines.append(entity)
        elif isinstance(entity, Surface):
            cls._surface_line_tags.append(
                np.array(
                    np.array([line.tag for line in entity.lines]),
                )
            )
            cls._surfaces.append(entity)

    @classmethod
    def _add_point(cls, point: "Point") -> "Point":
        """Add a point.

        Args:
            point (Point): The point to be added.

        Returns:
            Point: The existing point if it already exists, otherwise None.
        """
        return cls._add(point)  # type: ignore

    @classmethod
    def _add_line(cls, line: "Line") -> "Line":
        """Add a line.

        Args:
            line (Line): The line to be added.

        Returns:
            Line: The existing line if it already exists, otherwise None.
        """
        return cls._add(line)  # type: ignore

    @classmethod
    def _add_surface(cls, surface: "Surface") -> "Surface":
        """Add a surface.

        Args:
            surface (Surface): The surface to be added.

        Returns:
            Surface: The existing surface if it already exists, otherwise None.
        """
        return cls._add(surface)  # type: ignore

    @classmethod
    def clear(cls) -> None:
        """Clear all the geometric entities."""
        cls._points = []
        cls._lines = []
        cls._surfaces = []

        cls._points_coordinates = np.empty((0, 3), dtype=np.float64)
        cls._line_point_tags = np.empty((0, 2), dtype=np.int64)
        cls._surface_line_tags = []

        gmsh.clear()

    @classmethod
    def mesh(cls) -> None:
        """Generate the mesh using Gmsh."""
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.generate()

    @property
    def points(self) -> list["Point"]:
        """Return all the created points."""
        return self._points

    @property
    def lines(self) -> list["Line"]:
        """Return all the created lines."""
        return self._lines

    @property
    def surfaces(self) -> list["Surface"]:
        """Return all the created surfaces."""
        return self._surfaces


class Point:
    """Create a point with the given coordinates.

    Args:
        x (float): The x-coordinate of the point.
        y (float): The y-coordinate of the point.
        z (float): The z-coordinate of the point.
    """

    def __init__(self, x: float, y: float, z: float):
        self.coordinates = (x, y, z)

        point = Geometry()._add_point(self)
        if point:
            # This point already exists, send the old point instead of creating a new
            # one
            self.__dict__.update(point.__dict__)
        else:
            self.tag: int = gmsh.model.occ.addPoint(
                self.coordinates[0], self.coordinates[1], self.coordinates[2]
            )


class Line:
    """Create a line with the given points.

    Args:
        p1 (Point): The first point of the line.
        p2 (Point): The second point of the line.
        number_of_nodes (Optional[int], optional): The number of nodes on the line. If
        provided, the line will be transfinite. Defaults to None.
    """

    def __init__(self, p1: Point, p2: Point, number_of_nodes: Optional[int] = None):
        self.points = (p1, p2)

        line = Geometry()._add_line(self)
        if line:
            # This line already exists, send the old line instead of creating a new one
            self.__dict__.update(line.__dict__)
        else:
            self.tag: int = gmsh.model.occ.addLine(
                self.points[0].tag, self.points[1].tag
            )

            self.transfinite = False
            self.number_of_nodes = number_of_nodes
            if self.number_of_nodes:
                self.transfinite = True
                gmsh.model.occ.synchronize()
                gmsh.model.mesh.setTransfiniteCurve(self.tag, self.number_of_nodes)

    def __neg__(self) -> "Line":
        """Create a new line with the opposite orientation."""
        self.points = (self.points[1], self.points[0])
        self.tag = -self.tag

        return self


class Surface:
    def __init__(
        self,
        lines: list[Line],
        transfinite: bool = False,
        element_type: TwoDElementType = "quadrangle",
    ):
        """Create a surface with the given lines.

        Args:
            lines (list[Line]): The lines that form the surface. The lines must be
            connected (each line's end point is the start point of the next line).
            transfinite (bool, optional): If True, the surface will be transfinite. All
            the lines' number_of_nodes argument must be provided if the surface is
            transfinite. Defaults to False.
        """
        # Make sure all the lines are transfinite if the surface is transfinite
        self.transfinite = transfinite
        if self.transfinite:
            they_are_all_transfinite = all([line.transfinite for line in lines])
            if not they_are_all_transfinite:
                raise ValueError(
                    "If you would like to create a transfinite surface, all the lines'"
                    " number_of_nodes argument must be provided."
                )

            if len(lines) not in [3, 4]:
                raise ValueError("All the transfinite surfaces must have 3 or 4 lines.")

        # Make sure the points of lines are connected (each line's end point is the
        # start point of the next line) and lines are in clockwise order:
        ordered_lines: list[Line] = []
        current_point_tag = lines[0].points[0].tag
        for line in lines:
            if line.points[0].tag == current_point_tag:
                ordered_lines.append(line)
                current_point_tag = line.points[1].tag
            elif line.points[1].tag == current_point_tag:
                # If the line is in the opposite direction, use negative tag
                ordered_lines.append(-line)
                current_point_tag = line.points[0].tag
            else:
                raise ValueError(
                    "Lines are not properly connected. Make sure the lines are ordered."
                )

        # Check if the loop is closed
        if current_point_tag != lines[0].points[0].tag:
            raise ValueError("Lines do not form a closed loop.")

        self.lines = ordered_lines

        surface = Geometry()._add_surface(self)
        if surface:
            # This surface already exists, send the old surface instead of creating a
            # new one
            self.__dict__.update(surface.__dict__)
        else:
            # Create a line loop and a plane surface
            lines_tags = [line.tag for line in ordered_lines]
            line_loop_tag = gmsh.model.occ.addCurveLoop(lines_tags)
            self.tag = gmsh.model.occ.addPlaneSurface([line_loop_tag])
            self.element_type = element_type

            if self.transfinite or self.element_type == "quadrangle":
                gmsh.model.occ.synchronize()

            if self.transfinite:
                gmsh.model.mesh.setTransfiniteSurface(self.tag)

            if self.element_type == "quadrangle":
                gmsh.model.mesh.setRecombine(2, self.tag)
