"""
The `fastfem.mesh.generator` module is a wrapper around Gmsh to create meshes by
defining points with coordinates, lines with points, and surfaces with lines. The
wrapper is designed to be used in a more object-oriented way to create meshes easily. It
also avoids creating duplicate entities in the Gmsh model by checking if the entity
already exists.
"""

import copy
from typing import Literal, Optional, Any
import dataclasses

import gmsh
import numpy as np

__all__ = ["TwoDElementType", "Point", "Line", "Surface", "Geometry"]


def __dir__() -> list[str]:
    return __all__


# Gmsh needs to be initialized before creating any entities:
gmsh.initialize()

# The available 2D element types:
ZeroDElementType = Literal["point"]
OneDElementType = Literal["line"]
TwoDElementType = Literal["triangle", "quadrangle"]

gmsh_element_type_dictionary: dict[
    int, ZeroDElementType | OneDElementType | TwoDElementType
] = {
    1: "line",
    2: "triangle",
    3: "quadrangle",
    15: "point",
}


@dataclasses.dataclass(kw_only=True, frozen=True)
class Mesh:
    type: ZeroDElementType | OneDElementType | TwoDElementType
    node_tags: np.ndarray
    coordinates_of_nodes: np.ndarray
    element_tags: np.ndarray
    nodes_of_elements: np.ndarray


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

    _domains: dict[tuple[str, int], int] = {}  # {(domain_name, dim): tag}

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

            # Check lines in the opposite direction:
            if not comparison.any():
                comparison = (
                    cls._line_point_tags
                    == np.array([point.tag for point in entity.points[::-1]])
                ).all(axis=1)
                if comparison.any():
                    where_true = np.where(comparison)[0][0]
                    new_line = copy.deepcopy(entities[where_true])
                    return -new_line

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
    def create_domains(cls) -> None:
        """Create physical groups in Gmsh."""
        gmsh.model.occ.synchronize()

        # domains = {domain_name: {dim: [tags]}}
        domains: dict[str, dict[int, list[int]]] = {}

        def find_their_domains(
            entities: list[Point] | list[Line] | list[Surface], dim: int
        ) -> None:
            for entity in entities:
                if entity.domain_name:
                    if entity.domain_name not in domains:
                        domains[entity.domain_name] = {dim: []}
                    try:
                        domains[entity.domain_name][dim].append(entity.tag)
                    except KeyError:
                        raise ValueError(
                            "The same domain name cannot be used for entities of"
                            " different dimensions (a point and a line cannot have the"
                            " same domain name, for example)."
                        )

        find_their_domains(cls._points, 0)
        find_their_domains(cls._lines, 1)
        find_their_domains(cls._surfaces, 2)

        for domain_name, domain_entities in domains.items():
            dim = list(domain_entities.keys())[0]
            tags = domain_entities[dim]
            domain_tag = gmsh.model.add_physical_group(dim, tags, name=domain_name)
            cls._domains[domain_name, dim] = domain_tag

    @classmethod
    def mesh(cls) -> dict[tuple[str, int], list[Mesh]]:
        """Generate the mesh using Gmsh."""
        cls.create_domains()
        gmsh.model.mesh.generate()

        # mesh = {(domain_name, tag): [Mesh1, Mesh2]}
        mesh: dict[tuple[str, int], list[Mesh]] = {}

        for domain_name_and_dim, domain_tag in cls._domains.items():
            name = domain_name_and_dim[0]
            dim = domain_name_and_dim[1]
            nodes_from_gmsh = gmsh.model.mesh.get_nodes_for_physical_group(
                dim, domain_tag
            )
            tags_of_nodes = nodes_from_gmsh[0]
            coordinates_of_nodes = np.array(nodes_from_gmsh[1])

            geometric_entities = gmsh.model.get_entities_for_physical_group(
                dim, domain_tag
            )
            types_and_elements = {}
            for entity_tag in geometric_entities:
                elements_from_gmsh = gmsh.model.mesh.get_elements(dim, entity_tag)
                for i, type in enumerate(elements_from_gmsh[0]):
                    element_type = gmsh_element_type_dictionary[type]
                    if element_type not in types_and_elements:
                        types_and_elements[element_type] = {
                            "element_tags": elements_from_gmsh[1][i],
                            "nodes_of_elements": elements_from_gmsh[2][i],
                        }
                    else:
                        types_and_elements[element_type]["element_tags"] = np.vstack(
                            [
                                types_and_elements[element_type]["element_tags"],
                                elements_from_gmsh[1][i],
                            ]
                        )
                        types_and_elements[element_type]["nodes_of_elements"] = (
                            np.vstack(
                                [
                                    types_and_elements[element_type][
                                        "nodes_of_elements"
                                    ],
                                    elements_from_gmsh[2][i],
                                ]
                            )
                        )
            # For each element type, create a Mesh object and append it to the meshes:
            meshes: list[Mesh] = []
            for element_type, elements_and_nodes in types_and_elements.items():
                meshes.append(
                    Mesh(
                        type=element_type,
                        node_tags=tags_of_nodes,  # type: ignore
                        coordinates_of_nodes=coordinates_of_nodes,  # type: ignore
                        element_tags=elements_and_nodes["element_tags"],
                        nodes_of_elements=elements_and_nodes["nodes_of_elements"],
                    )
                )

            mesh[name, domain_tag] = meshes

        return mesh

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


@dataclasses.dataclass
class Point:
    """Create a point with the given coordinates.

    Args:
        x (float): The x-coordinate of the point.
        y (float): The y-coordinate of the point.
        z (float): The z-coordinate of the point.
        domain_name (Optional[str], optional): The name of the domain the point belongs
        to. Defaults to None.
    """

    x: float
    y: float
    z: float
    domain_name: Optional[str] = None

    def __post_init__(self):
        self.coordinates = (self.x, self.y, self.z)

        point = Geometry()._add_point(self)
        if point:
            # This point already exists, send the old point instead of creating a new
            # one
            self.__dict__.update(point.__dict__)
        else:
            self.tag: int = gmsh.model.occ.add_point(self.x, self.y, self.z)


@dataclasses.dataclass
class Line:
    """Create a line with the given points.

    Args:
        start_point (Point): The start point of the line.
        end_point (Point): The end point of the line.
        number_of_nodes (Optional[int], optional): The number of nodes on the line. If
        provided, the line will be transfinite. Defaults to None.
        domain_name (Optional[str], optional): The name of the domain the line belongs
        to. Defaults to None.
    """

    start_point: Point
    end_point: Point
    number_of_nodes: Optional[int] = None
    domain_name: Optional[str] = None

    def __post_init__(
        self,
    ):
        self.points = (self.start_point, self.end_point)

        line = Geometry()._add_line(self)
        if line:
            # This line already exists, send the old line instead of creating a new one
            self.__dict__.update(line.__dict__)
        else:
            self.tag: int = gmsh.model.occ.add_line(
                self.points[0].tag, self.points[1].tag
            )

            self.transfinite = False
            if self.number_of_nodes:
                self.transfinite = True
                gmsh.model.occ.synchronize()
                gmsh.model.mesh.set_transfinite_curve(self.tag, self.number_of_nodes)

    def __neg__(self) -> "Line":
        """Create a new line with the opposite orientation."""
        self.points = (self.points[1], self.points[0])
        self.tag = -self.tag

        return self


@dataclasses.dataclass
class Surface:
    """Create a surface with the given lines.

    Args:
        lines (list[Line]): The lines that form the surface. The lines must be
        connected (each line's end point is the start point of the next line).
        transfinite (bool, optional): If True, the surface will be transfinite. All the
        lines' number_of_nodes argument must be provided if the surface is transfinite.
        Defaults to False.
        element_type (TwoDElementType, optional): The type of element to use. Defaults
        to "triangle".
        domain_name (Optional[str], optional): The name of the domain the surface
        belongs to. Defaults to None.

    """

    lines: list[Line]
    transfinite: bool = False
    element_type: TwoDElementType = "triangle"
    domain_name: Optional[str] = None

    def __post_init__(self):
        """Create a surface with the given lines.

        Args:
            lines (list[Line]): The lines that form the surface. The lines must be
            connected (each line's end point is the start point of the next line).
            transfinite (bool, optional): If True, the surface will be transfinite. All
            the lines' number_of_nodes argument must be provided if the surface is
            transfinite. Defaults to False.
        """
        # Make sure all the lines are transfinite if the surface is transfinite
        if self.transfinite:
            lines_are_all_transfinite = all([line.transfinite for line in self.lines])
            if not lines_are_all_transfinite:
                raise ValueError(
                    "If you would like to create a transfinite surface, all the lines'"
                    " number_of_nodes argument must be provided."
                )

            if len(self.lines) not in [3, 4]:
                raise ValueError("All the transfinite surfaces must have 3 or 4 lines.")

        # Make sure the points of lines are connected (each line's end point is the
        # start point of the next line) and lines are in clockwise order:
        ordered_lines: list[Line] = []
        current_point_tag = self.lines[0].points[0].tag
        for line in self.lines:
            if line.points[0].tag == current_point_tag:
                ordered_lines.append(line)
                current_point_tag = line.points[1].tag
            elif line.points[1].tag == current_point_tag:
                # If the line is in the opposite direction, use negative tag
                ordered_lines.append(-line)
                current_point_tag = line.points[1].tag
            else:
                raise ValueError(
                    "Lines are not properly connected. Make sure the lines are ordered."
                )

        # Check if the loop is closed
        if current_point_tag != self.lines[0].points[0].tag:
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
            line_loop_tag = gmsh.model.occ.add_curve_loop(lines_tags)
            self.tag = gmsh.model.occ.add_plane_surface([line_loop_tag])

            if self.transfinite or self.element_type == "quadrangle":
                gmsh.model.occ.synchronize()

            if self.transfinite:
                gmsh.model.mesh.set_transfinite_surface(self.tag)

            if self.element_type == "quadrangle":
                gmsh.model.mesh.set_recombine(2, self.tag)
