"""
The `fastfem.mesh` package contains utilities for generating and reading meshes.
"""

from .fundamentals import Rectangle, Square
from .generator import Geometry, Line, Point, Surface

__all__ = ["Rectangle", "Geometry", "Point", "Line", "Surface", "Square"]
