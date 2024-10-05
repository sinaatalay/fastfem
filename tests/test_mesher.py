import fastfem
import gmsh


def test_rectangle_mesh():
    domain = fastfem.mesh.generator.Rectangle(
        width=1.0,
        height=1.0,
        nx=10,
        ny=10,
        element_type="triangle",
    )

    gmsh.fltk.run()
    gmsh.finalize()
