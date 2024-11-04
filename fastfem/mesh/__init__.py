"""
The `fastfem.mesh` package contains utilities for generating and reading meshes.
"""

from .fundamentals import create_a_rectangle_mesh, create_a_square_mesh
from .generator import (
    Domain,
    Line,
    Mesh,
    OneDElementType,
    Point,
    Submesh,
    Surface,
    TwoDElementType,
    ZeroDElementType,
    mesh,
)

__all__ = [
    "create_a_rectangle_mesh",
    "create_a_square_mesh",
    "ZeroDElementType",
    "OneDElementType",
    "TwoDElementType",
    "Mesh",
    "Domain",
    "Submesh",
    "Point",
    "Line",
    "Surface",
    "mesh",
]
