import fastfem.elements.spectral_element as SE
import numpy as np
import argparse


def run_demo(
    elem_order: int,
    nelem_x: int,
    nelem_y: int,
    dt: float,
    bump_x: float = 0.3,
    bump_y: float = 0.6,
):
    """Runs a spectral element method to solve the wave equation
    on Ω=(0,1)² with homogeneous Dirichlet (essential) boundary conditions.
    Wave speed is locked to one. The initial condition is a bump at
    (`bump_x`,`bump_y`) of amplitude 1.

    After each step, the state is yielded as the tuple (`step`,`t`,`coords`,`u`), where
    `coords` is of shape (`nelem_x`,`nelem_y`,`elem_order`,
    `elem_order`,2) representing
    nodal positions, and `u` is of shape (`nelem_x`,`nelem_y`,`elem_order`,
    `elem_order`) representing the nodal values of the solution at the given
    time `t`. `step` is the number of steps taken (namely, `t = step * dt`)

    Stepping is done using Newmark-beta.

    Args:
        elem_order (int): the degree of the polynomial used in the element
        nelem_x (int): how many elements along the x-axis
        nelem_y (int): how many elements along the y-axis
        dt (float): The step size.
    """
    elem = SE.SpectralElement2D(elem_order)

    # each point in each element is assigned a global index (assembly)
    # that matches neighboring elements across boundaries

    global_indices, global_dim = build_assembly(elem_order, nelem_x, nelem_y)
    cell_width = 1 / nelem_x
    cell_height = 1 / nelem_y
    coords = np.empty((nelem_x, nelem_y, elem_order + 1, elem_order + 1, 2))

    # u in global basis with additional throw-away value for boundaries
    U = np.empty(global_dim + 1)
    UDOT = np.empty(global_dim + 1)
    UDDOT = np.empty(global_dim + 1)
    MASS_MAT = np.zeros(global_dim + 1)

    char_wavenum = 10

    def init_cond_U(x, y):
        return np.exp(
            (-0.5 * char_wavenum**2) * ((x - bump_x) ** 2 + (y - bump_y) ** 2)
        )

    def init_cond_UDOT(x, y):
        return np.zeros(np.broadcast_shapes(x.shape, y.shape))

    for i in range(nelem_x):
        for j in range(nelem_y):
            coords[i, j, :, :, 0] = cell_width * (
                i + (elem.knots[:, np.newaxis] + 1) / 2
            )
            coords[i, j, :, :, 1] = cell_height * (
                j + (elem.knots[np.newaxis, :] + 1) / 2
            )
            MASS_MAT[global_indices[i, j]] += elem.basis_mass_matrix(coords[i, j])
            U[global_indices[i, j]] = init_cond_U(
                coords[i, j, :, :, 0], coords[i, j, :, :, 1]
            )
            UDOT[global_indices[i, j]] = init_cond_UDOT(
                coords[i, j, :, :, 0], coords[i, j, :, :, 1]
            )
    U[-1] = 0
    UDOT[-1] = 0

    def update_UDDOT():
        UDDOT[:] = 0
        for i in range(nelem_x):
            for j in range(nelem_y):
                UDDOT[global_indices[i, j]] -= elem.basis_stiffness_matrix_times_field(
                    coords[i, j], U[global_indices[i, j]]
                )
        UDDOT[:-1] /= MASS_MAT[:-1]

    u = np.empty((nelem_x, nelem_y, elem_order + 1, elem_order + 1))
    nsteps = 0
    t = 0
    update_UDDOT()

    while True:
        u[...] = U[global_indices]
        yield nsteps, t, coords, u

        # newmark-beta:
        U[:-1] += dt * UDOT[:-1] + (0.5 * dt**2) * UDDOT[:-1]  # update u to next step
        UDOT[:-1] += (0.5 * dt) * UDDOT[:-1]  # update udot to half step
        update_UDDOT()  # no longer need UDDOT of current step; get next
        UDOT[:-1] += (0.5 * dt) * UDDOT[:-1]  # update udot to the next full step
        nsteps += 1
        t += dt


def build_assembly(elem_order, nelem_x, nelem_y):
    global_indices = np.zeros(
        (nelem_x, nelem_y, elem_order + 1, elem_order + 1), dtype=int
    )
    global_size = 0
    for i in range(nelem_x):
        for j in range(nelem_y):
            # every element has the interior
            global_indices[i, j, 1:-1, 1:-1] = global_size + np.arange(
                (elem_order - 1) ** 2
            ).reshape((elem_order - 1, elem_order - 1))
            global_size += (elem_order - 1) ** 2

            # -x
            if i == 0:
                # exclude from global basis
                global_indices[i, j, 0, :] = -1
            else:
                # continuous with left elem
                global_indices[i, j, 0, :] = global_indices[i - 1, j, -1, :]

            # +x
            if i == nelem_x - 1:
                # exclude from global basis
                global_indices[i, j, -1, :] = -1
            else:
                # new indices
                global_indices[i, j, -1, 1:-1] = global_size + np.arange(elem_order - 1)
                global_size += elem_order - 1
                if j < nelem_y - 1:
                    global_indices[i, j, -1, -1] = global_size
                    global_size += 1

            # -y
            if j == 0:
                # exclude from global basis
                global_indices[i, j, :, 0] = -1
            else:
                # continuous with bottom elem
                global_indices[i, j, :, 0] = global_indices[i, j - 1, :, -1]

            # +y
            if j == nelem_y - 1:
                # exclude from global basis
                global_indices[i, j, :, -1] = -1
            else:
                # new indices
                global_indices[i, j, 1:-1, -1] = global_size + np.arange(elem_order - 1)
                global_size += elem_order - 1
    return global_indices, global_size


def build_argparse():
    parser = argparse.ArgumentParser(
        prog="demo_spectral_wave",
        description="""
Example code for the spectral elements to solve the Wave Equation.
This code uses a custom mesh assembly, separate from the package.
                    """,
        epilog="""
The equation is 
        { ∂ₜu² = Δu on Ω=(0,1)²   and   u = 0 on ∂Ω } with wave speed 1.
        """,
    )
    parser.add_argument(
        "-o",
        "--order",
        help="""
Element order (degree of the polynomial to be used). The number of
nodes per axis
    """,
        action="store",
        default=5,
        type=int,
        metavar="ELEM_ORDER",
    )
    parser.add_argument(
        "-x",
        "--nelemx",
        "--nx",
        help="""
Number of elements along the x-axis
    """,
        action="store",
        default=10,
        type=int,
        metavar="NELEM_X",
    )
    parser.add_argument(
        "-y",
        "--nelemy",
        "--ny",
        help="""
Number of elements along the y-axis
    """,
        action="store",
        default=10,
        type=int,
        metavar="NELEM_Y",
    )
    parser.add_argument(
        "-s",
        "--step",
        "--dt",
        help="""
Time step size
    """,
        action="store",
        default=1e-3,
        type=float,
        metavar="DT",
    )
    parser.add_argument(
        "-t",
        "--tmax",
        help="""
Simulation end time
    """,
        action="store",
        default=5,
        type=float,
        metavar="TMAX",
    )
    parser.add_argument(
        "-p",
        "--plot",
        help="""
Set to plot using matplotlib.
    """,
        action="store_true",
    )
    return parser


if __name__ == "__main__":
    args = build_argparse().parse_args()
    nelemx = args.nelemx
    nelemy = args.nelemy
    elem_order = args.order
    print(
        f"Demo-ing spectral elements: order {elem_order} on a "
        + f"{nelemx}x{nelemy} grid"
    )
    dt = args.step
    tmax = args.tmax
    print(f"  Running to t={tmax} with step size {dt}.")

    UINDS = np.empty((4, nelemx * nelemy * (elem_order + 1) ** 2), dtype=int)
    UINDS[0, :] = np.arange(nelemx * nelemy * (elem_order + 1) ** 2)
    UINDS[3, :] = UINDS[0, :] % (elem_order + 1)
    UINDS[0, :] //= elem_order + 1
    UINDS[2, :] = UINDS[0, :] % (elem_order + 1)
    UINDS[0, :] //= elem_order + 1
    UINDS[1, :] = UINDS[0, :] % nelemy
    UINDS[0, :] //= nelemy

    tri = None
    contour = None
    if args.plot:
        import matplotlib.pyplot as plt
        import matplotlib.tri as mpltri

        def init_plt(coords):
            global tri
            tri = mpltri.Triangulation(coords[*UINDS, 0], coords[*UINDS, 1])
            plt.ion()
            plt.figure(0)
            plt.show(block=False)

        def update_plt(t, u):
            plt.cla()
            plt.title(f"t = {t:.4f}")
            plt.tricontourf(tri, u[*UINDS], 100, vmin=-0.25, vmax=0.25)
            plt.draw()
            plt.pause(0.001)

    else:

        def init_plt(coords):
            pass

        def update_plt(t, u):
            pass

    for i, t, coords, u in run_demo(elem_order, nelemx, nelemy, dt):
        if i == 0 and args.plot:
            init_plt(coords)
        update_plt(t, u)
        if t >= tmax:
            break

        print(
            f"step {i:6d} (t = {t:8.4f}): max |u| = {np.max(np.abs(u)):5.2f}", end="\r"
        )
