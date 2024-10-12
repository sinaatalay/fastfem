from typing import Optional, Literal
import gmsh
import numpy as np

ElementType = Literal["triangle", "quadrangle"]


class _Database:
    """Singleton class to store all the entities created in the gmsh model."""

    _instance = None
    points_coordinates: np.ndarray = np.empty((0, 3))
    points: list["Point"] = []

    line_point_tags: np.ndarray = np.empty((0, 2), dtype=np.int64)
    lines: list["Line"] = []

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(_Database, cls).__new__(cls)
        return cls._instance

    @classmethod
    def add_point(cls, point: "Point") -> Optional["Point"]:
        comparison_with_database = np.isclose(
            cls.points_coordinates, point.coordinates
        ).all(axis=1)
        where_true = np.where(comparison_with_database)[0]
        if where_true.size:
            where_true = int(where_true[0])
            return cls.points[where_true]

        cls.points_coordinates = np.vstack([cls.points_coordinates, point.coordinates])
        cls.points.append(point)
        return None

    @classmethod
    def add_line(cls, line: "Line") -> Optional["Line"]:
        comparison_with_database = np.isclose(
            cls.line_point_tags, [line.points[0].tag, line.points[1].tag]
        ).all(axis=1)
        where_true = np.where(comparison_with_database)[0]
        if where_true.size:
            where_true = int(where_true[0])
            return cls.lines[where_true]

        cls.line_point_tags = np.vstack(
            [
                cls.line_point_tags,
                [line.points[0].tag, line.points[1].tag],
            ]
        )
        cls.lines.append(line)

    @classmethod
    def clear(cls):
        cls.points_coordinates = np.empty((0, 3))
        cls.points = []

        cls.line_point_tags = np.empty((0, 2), dtype=np.int64)
        cls.lines = []


class Point:
    def __init__(self, x: float, y: float, z: float):
        gmsh.initialize()
        self.coordinates = (x, y, z)

        point = _Database().add_point(self)
        if point:
            # This point already exists in the database, send the old point instead of
            # creating a new one
            self.tag = point.tag
        else:
            self.tag: int = gmsh.model.occ.addPoint(
                self.coordinates[0], self.coordinates[1], self.coordinates[2]
            )


class Line:
    def __init__(self, p1: Point, p2: Point, number_of_nodes: Optional[int] = None):
        self.points = (p1, p2)

        line = _Database().add_line(self)
        if line:
            # This line already exists in the database, send the old line instead of
            # creating a new one
            self.tag = line.tag
            self.number_of_nodes = line.number_of_nodes
            self.transfinite = line.transfinite
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

    def __neg__(self):
        self.points = (self.points[1], self.points[0])
        self.tag = -self.tag

        return self


class Surface:
    def __init__(
        self,
        lines: list[Line],
        transfinite: bool = False,
        element_type: ElementType = "quadrangle",
    ):
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

        # Create a line loop and a plane surface
        lines_tags = [line.tag for line in ordered_lines]
        line_loop_tag = gmsh.model.occ.addCurveLoop(lines_tags)
        self.tag = gmsh.model.occ.addPlaneSurface([line_loop_tag])

        gmsh.model.occ.synchronize()

        if self.transfinite:
            gmsh.model.mesh.setTransfiniteSurface(self.tag)

        self.element_type = element_type

        if self.element_type == "quadrangle":
            gmsh.model.mesh.setRecombine(2, self.tag)

    def mesh(self):
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.generate(2)

