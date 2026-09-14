import types
from math import factorial

import numpy as np
import pytest
from numpy.typing import ArrayLike

from fastfem.elements.linear_simplex2d import LinearSimplex2D
from fastfem.fields.field import Field


def meshquad(X, Y, F):
    """
    In case this is ever needed, this is a function to compute the integral on a 2D mesh.
    Integrals are estimated by estimating integrals on triangles, which are defined by
    indices ([i,j] - [i+1,j] - [i,j+1]) ([i+1,j+1] - [i+1,j] - [i,j+1])
    """

    def cross_mag(XA, YA, XB, YB):
        return np.abs(XA * YB - XB * YA)

    F00 = F[:-1, :-1]
    F01 = F[:-1, 1:]
    F10 = F[1:, :-1]
    F11 = F[1:, 1:]
    X00 = X[:-1, :-1]
    X01 = X[:-1, 1:]
    X10 = X[1:, :-1]
    X11 = X[1:, 1:]
    Y00 = Y[:-1, :-1]
    Y01 = Y[:-1, 1:]
    Y10 = Y[1:, :-1]
    Y11 = Y[1:, 1:]
    axes = np.arange(X.ndim)
    return np.sum(
        cross_mag(X10 - X00, Y10 - Y00, X01 - X00, Y01 - Y00) * (F00 + F01 + F10) / 6,
        axes,
    ) + np.sum(
        cross_mag(X10 - X11, Y10 - Y11, X01 - X11, Y01 - Y11) * (F11 + F01 + F10) / 6,
        axes,
    )


def _analytic_integrate_reftri(powx: int, powy: int) -> float:
    """Integrates x**powx * y**powy on the reference triangle
    [0,0] - [1,0] - [0,1]

    Args:
        powx (int): exponent for x
        powy (int): exponent for y

    Returns:
        float: the resultant integral
    """
    #   int_0^1 x**powx int_0^{1-x}  y**powy   dy dx
    return factorial(powx) * factorial(powy) / factorial(powx + powy + 2)


@pytest.fixture(scope="module")
def element():
    return LinearSimplex2D()


@pytest.fixture(scope="module")
def transformed_element(element, transformation):
    return (
        element,
        Field(
            element.basis_shape(),
            (2,),
            transformation(element.reference_element_position_field().coefficients),
        ),
        transformation,
    )


@pytest.fixture(scope="session")
def triangle_quadrature():
    reference_rules = dict()

    class TriangleQuadratureRuleBuilder:
        """A triangular quadrature rule tuned to a specific
        degree of exactness.
        """

        def __call__(
            self, exactness: int, tribounds: ArrayLike = [[0, 0], [1, 0], [0, 1]]
        ):
            if not isinstance(tribounds, np.ndarray):
                tribounds = np.array(tribounds)

            if exactness not in reference_rules:
                # num polys: count (a,b) for which a+b <= exactness
                # for simplicity of solving for weights, we use a
                # 2d newton-coates variant, where our "equally-spaced"
                # nodes are given as
                def line(y, n):
                    pts = np.empty((n, 2))
                    pts[:, 0] = np.linspace(0, 1 - y, n)
                    pts[:, 1] = y
                    return pts

                meshpts = np.concatenate(
                    [
                        line(1 - n / (exactness + 1), n + 1)
                        for n in range(exactness + 1)
                    ],
                    axis=0,
                )
                integrals = np.empty(meshpts.shape[0])
                coef_evals = np.empty((meshpts.shape[0], meshpts.shape[0]))

                iterm: int = 0
                for powx in range(exactness + 1):
                    for powy in range(exactness - powx + 1):
                        coef_evals[iterm, :] = (
                            meshpts[:, 0] ** powx * meshpts[:, 1] ** powy
                        )
                        integrals[iterm] = _analytic_integrate_reftri(powx, powy)
                        iterm += 1

                # CE_{ij} w_j = I_i
                # where CE_{ij} = f_i(x_j),  I_i = int(f_i)
                weights = np.linalg.solve(coef_evals, integrals)
                reference_rules[exactness] = types.SimpleNamespace(
                    weights=weights, knots=meshpts
                )

            # we have the reference element values. we want quadratures on tribounds
            affine_shift = tribounds[0, :]
            affine_linT = tribounds[1:, :] - tribounds[0, :]

            weights = reference_rules[exactness].weights * abs(
                np.linalg.det(affine_linT)
            )
            knots = affine_shift + np.einsum(
                "ij,...i->...j", affine_linT, reference_rules[exactness].knots
            )

            return types.SimpleNamespace(
                weights=weights, knots=knots, exactness=exactness, tribounds=tribounds
            )

    return TriangleQuadratureRuleBuilder()


@pytest.mark.skip
def test_integrate_reftri(triangle_quadrature):
    # this is used to do midpoint rule on triangles
    def cross_mag(A, B):
        return np.abs(A[..., 0] * B[..., 1] - B[..., 0] * A[..., 1])

    # 10 subdivs -> ~3 mil points?
    n_subdivisions = 10
    reftri_mesh = np.array([[[0, 0], [1, 0], [0, 1]]])
    for _ in range(n_subdivisions):
        mid01 = 0.5 * (reftri_mesh[:, 0, :] + reftri_mesh[:, 1, :])
        mid12 = 0.5 * (reftri_mesh[:, 2, :] + reftri_mesh[:, 1, :])
        mid20 = 0.5 * (reftri_mesh[:, 0, :] + reftri_mesh[:, 2, :])
        numtris = reftri_mesh.shape[0]
        mesh_next = np.empty((numtris * 4, 3, 2))

        mesh_next[:numtris, 0, :] = reftri_mesh[:, 0, :]
        mesh_next[:numtris, 1, :] = mid01
        mesh_next[:numtris, 2, :] = mid20

        mesh_next[numtris : (2 * numtris), 0, :] = mid01
        mesh_next[numtris : (2 * numtris), 1, :] = reftri_mesh[:, 1, :]
        mesh_next[numtris : (2 * numtris), 2, :] = mid12

        mesh_next[(2 * numtris) : (3 * numtris), 0, :] = mid20
        mesh_next[(2 * numtris) : (3 * numtris), 1, :] = mid12
        mesh_next[(2 * numtris) : (3 * numtris), 2, :] = reftri_mesh[:, 2, :]

        mesh_next[(3 * numtris) :, 0, :] = mid12
        mesh_next[(3 * numtris) :, 1, :] = mid20
        mesh_next[(3 * numtris) :, 2, :] = mid01
        reftri_mesh = mesh_next

    midpoints = np.mean(reftri_mesh, axis=1)

    for powx in range(4):
        for powy in range(4):
            res = _analytic_integrate_reftri(powx, powy)
            analytic = np.sum(
                # f(x,y) at midpoints
                (midpoints[:, 0] ** powx * midpoints[:, 1] ** powy)
                # areas
                * 0.5
                * cross_mag(
                    reftri_mesh[:, 1, :] - reftri_mesh[:, 0, :],
                    reftri_mesh[:, 2, :] - reftri_mesh[:, 0, :],
                )
            )
            assert res == pytest.approx(analytic, rel=1e-5), (
                f"Failed _analytic_integrate_reftri({powx},{powy})"
            )
            for exactdeg in range(powx + powy, 10):
                quad_rule = triangle_quadrature(exactdeg)
                quad_result = np.sum(
                    quad_rule.weights
                    * (quad_rule.knots[:, 0] ** powx * quad_rule.knots[:, 1] ** powy)
                )
                assert quad_result == pytest.approx(res, rel=1e-9), (
                    f"Failed exactness for x**{powx} * y**{powy} for DOE {exactdeg}."
                )


def test_interpolate_field(element):
    field = element.basis_fields()
    constfield = Field(tuple(), tuple(), 3)
    for x, y in [(0, 0), (1, 0), (0, 1), (0.5, 0.5), (0.25, 0.25)]:
        field_outs = np.array([1 - x - y, x, y])
        np.testing.assert_allclose(
            element.interpolate_field(field, x, y).coefficients, field_outs
        )

        assert element.interpolate_field(
            constfield, x, y
        ).coefficients == pytest.approx(constfield.coefficients)


def test_integrate_field(transformed_element, triangle_quadrature):
    element, pts, transform = transformed_element
    quad = triangle_quadrature(3, tribounds=pts.coefficients)
    ref_quad = triangle_quadrature(3)

    field = element.basis_fields()
    jac_scale = Field(
        element.basis_shape(), tuple(), field.coefficients[:, np.newaxis, :]
    )
    result = element.integrate_field(pts, field, jac_scale).coefficients

    field_quad = element.interpolate_field(
        field.stack[np.newaxis, :],
        ref_quad.knots[:, 0],
        ref_quad.knots[:, 1],
    ).coefficients
    jac_quad = element.interpolate_field(
        jac_scale,
        ref_quad.knots[:, 0],
        ref_quad.knots[:, 1],
    ).coefficients

    quad_result = np.sum(quad.weights[:] * field_quad * jac_quad, axis=-1)

    np.testing.assert_allclose(result, quad_result, atol=1e-10)


def test_integrate_basis_times_field(transformed_element, triangle_quadrature):
    element, pts, transform = transformed_element
    quad = triangle_quadrature(4, tribounds=pts.coefficients)
    ref_quad = triangle_quadrature(4)

    field = element.basis_fields()
    jac_scale = Field(
        element.basis_shape(), tuple(), field.coefficients[:, np.newaxis, :]
    )
    result = element.integrate_basis_times_field(pts, field, None, jac_scale)

    field_quad = element.interpolate_field(
        field,
        ref_quad.knots[:, 0],
        ref_quad.knots[:, 1],
    ).coefficients
    jac_quad = element.interpolate_field(
        jac_scale,
        ref_quad.knots[:, 0],
        ref_quad.knots[:, 1],
    ).coefficients
    basis_field = Field(
        element.basis_shape(), tuple(), field.coefficients[:, np.newaxis, np.newaxis, :]
    )
    basis_quad = element.interpolate_field(
        basis_field,
        ref_quad.knots[:, 0],
        ref_quad.knots[:, 1],
    ).coefficients

    quad_result = np.sum(
        quad.weights * field_quad * jac_quad * basis_quad,
        axis=-1,
    )

    np.testing.assert_allclose(result.coefficients, quad_result, atol=1e-10)


def test_integrate_grad_basis_dot_field(transformed_element, triangle_quadrature):
    element, pts, transform = transformed_element
    quad = triangle_quadrature(4, tribounds=pts.coefficients)
    ref_quad = triangle_quadrature(4)

    field = element.basis_fields()
    jac_scale = Field(
        element.basis_shape(), tuple(), field.coefficients[:, np.newaxis, :]
    )
    basis_field = Field(
        element.basis_shape(), tuple(), field.coefficients[:, np.newaxis, np.newaxis, :]
    )
    grad_basis = element.compute_field_gradient(basis_field, pts)
    basis_quad = element.interpolate_field(
        grad_basis,
        ref_quad.knots[:, 0],
        ref_quad.knots[:, 1],
    ).coefficients
    for eind in range(2):
        efield = Field(
            field.basis_shape,
            (2,),
            field.coefficients[..., np.newaxis] * np.eye(2)[:, eind],
        )

        result = element.integrate_grad_basis_dot_field(pts, efield, None, jac_scale)

        field_quad = element.interpolate_field(
            efield,
            ref_quad.knots[:, 0],
            ref_quad.knots[:, 1],
        ).coefficients
        jac_quad = element.interpolate_field(
            jac_scale,
            ref_quad.knots[:, 0],
            ref_quad.knots[:, 1],
        ).coefficients

        quad_result = np.moveaxis(
            np.sum(
                quad.weights[:]
                * jac_quad
                * np.einsum("...i,...i->...", field_quad, basis_quad),
                axis=-1,
            ),
            0,
            -1,
        )  # shift basis to right of stack

        np.testing.assert_allclose(result.coefficients, quad_result, atol=1e-10)


def test_field_gradient_const(element):
    constfield = Field(tuple(), tuple(), 3)
    np.testing.assert_allclose(
        element._compute_field_gradient(constfield).coefficients, 0, atol=1e-10
    )
