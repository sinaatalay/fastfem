import numpy as np
import pytest

from fastfem.elements.element2d import Element2D
from fastfem.elements.linear_simplex2d import LinearSimplex2D
from fastfem.elements.spectral_element2d import SpectralElement2D
from fastfem.fields import Field
from fastfem.fields import numpy_similes as fnp
from fastfem.fields.field import ShapeComponent

# ======================================================================================
#                                  Adding elements
#     When adding new elements, insert them into the dictionary `elements_to_test`. The
# key used to insert the element will be referenced in the name of the test
# parameterization. Additionally, to test the interpolate_... methods, a collection of
# points can be inserted into `element_test_localcoords`, which sould be an array of
# shape (...,2).
#
# This test suite will verify the basic functionality of the element,
# while things like integration will only be tested insofar as validating things such
# as field stacking. For an element to have a complete collection of tests, one should
# additionally have the following tested separately:
#
#   - interpolate_field(field,x,y). Agreement between different configurations of
#       stack_shape, point_shape or x,y shapes will be tested,
#       as well as linearity of solutions. So, the developer only needs to test a
#       spanning set of the scalar field space.
#
#   - compute_field_gradient(field, pos_matrix). As above, the developer only needs to
#       test a spanning set of the scalar field space. This test can be skipped by
#       appending the name of the element to
#       `elements_for_field_gradient_finitediff_test`, which tests
#       compute_field_gradient according to interpolate_field(),
#       reference_element_position_field() and finite differencing. It is assumed
#       that compute_field_gradient yields an exact value, and that nonlinear
#       transformations need not be checked.
#
#   - integrate_field(pos_matrix, field, jacobian_scale),
#   - integrate_basis_times_field(pos_matrix, field, indices, jacobian_scale)
#   - integrate_grad_basis_dot_field(pos_matrix,field,is_field_upper_index,indices,
#       jacobian_scale)
#   - integrate_grad_basis_dot_grad_field(pos_matrix,field,indices,jacobian_scale)
#       The same agreements will be verified (except with bilinearity of field and
#       jacobian_scale instead of linearity), so the developer only needs to test a
#       spanning set of the triple Cartesian product of the scalar field space and a
#       satisfactory set of transformed elements. Linear transformations on pos_matrix
#       will be tested by this suite, so only nonlinear transformations need to be
#       verified. Additionally, agreement with different index configurations will be
#       verified, so only indices=None needs to be taken as a case. Since
#       grad_basis_dot_field requires a vector field, so testing needs only be tested
#       with multiples of the elementary basis of R2.
#
#       Under most cases, integrate_grad_basis_dot_grad_field should be equivalent to
#       integrate_grad_basis_dot_field(..., compute_field_grad(field), ...). The test
#       for this is written in this test suite, and can be run by adding the element
#       name to `elements_for_grad_basis_dot_grad_field_test`
#
# ======================================================================================
elements_to_test = dict()
element_test_localcoords = dict()
elements_for_field_gradient_finitediff_test = set()
elements_for_grad_basis_dot_grad_field_test = set()
elements_for_ref_element_def_grad_is_identity_test = set()

# register linearsimplex2D
elements_to_test["linearsimplex"] = LinearSimplex2D()
element_test_localcoords["linearsimplex"] = np.array(
    [[0, 0], [1, 0], [0, 1], [0.5, 0.5], [0.25, 0.25]]
)
elements_for_field_gradient_finitediff_test.add("linearsimplex")
elements_for_grad_basis_dot_grad_field_test.add("linearsimplex")
elements_for_ref_element_def_grad_is_identity_test.add("linearsimplex")

# register spectral elements of different orders
for i in [3, 4, 5]:
    testname = f"spectral{i}"
    elements_to_test[testname] = SpectralElement2D(i)
    elements_for_field_gradient_finitediff_test.add(testname)
    elements_for_grad_basis_dot_grad_field_test.add(testname)
    elements_for_ref_element_def_grad_is_identity_test.add(testname)
    test_pts = np.empty((2 * i, 2 * i, 2))
    test_pts[:, :, 0] = np.linspace(-1, 1, 2 * i)[:, np.newaxis]
    test_pts[:, :, 1] = np.linspace(-1, 1, 2 * i)[np.newaxis, :]
    element_test_localcoords[testname] = test_pts

# ======================================================================================


@pytest.fixture(scope="module", params=elements_to_test.keys())
def element(request):
    return elements_to_test[request.param]


@pytest.fixture(scope="module", params=element_test_localcoords.keys())
def element_and_local_coords(request):
    key = request.param
    coords = element_test_localcoords[key]
    assert key in elements_to_test, (
        f"Element {key} found in element_test_localcoords, but not elements_to_test."
    )
    assert coords.shape[-1] == 2, (
        f"Last axis of element_test_localcoords['{key}'] is not size 2! shape is"
        f" {coords.shape}."
    )
    return elements_to_test[request.param], coords


def test_basis_and_reference_shapes(element: Element2D):
    """Validates the shapes of basis_fields() and reference_element_position_field()
    against basis_shape().
    """
    shape = element.basis_shape()
    basis = element.basis_fields()

    assert basis.basis_shape == shape, (
        "basis_fields() should have basis_shape == basis_shape(). "
        + f"basis_fields().basis_shape: {basis.basis_shape} basis_shape(): {shape}."
    )
    assert basis.point_shape == tuple(), (
        "basis_fields() should be a stack of scalar fields. Instead, point_shape =="
        f" {basis.point_shape}."
    )

    ref_elem_pts = element.reference_element_position_field()

    assert ref_elem_pts.basis_shape == shape, (
        "reference_element_position_field() should have basis_shape == basis_shape(). "
        + "reference_element_position_field().basis_shape:"
        f" {ref_elem_pts.basis_shape} basis_shape(): {shape}."
    )

    assert ref_elem_pts.stack_shape == tuple(), (
        "reference_element_position_field() should have stack_shape == tuple()."
        f" reference_element_position_field().stack_shape: {ref_elem_pts.stack_shape}"
    )

    assert ref_elem_pts.point_shape == (2,), (
        "reference_element_position_field() should have point_shape == (2,)."
        f" reference_element_position_field().point_shape: {ref_elem_pts.point_shape}"
    )


@pytest.mark.parametrize("elem", elements_for_ref_element_def_grad_is_identity_test)
def test_reference_deformation_gradient(elem: str):
    """Validates the reference position matrix's deformation gradient, which should
    be the identity.
    """
    element = elements_to_test[elem]
    def_grad = element.compute_field_gradient(
        element.reference_element_position_field()
    )
    np.testing.assert_allclose(def_grad.coefficients - np.eye(2), 0, atol=1e-6)


def random_basis_coefs(element: Element2D, num_trials: int | None = None):
    # maybe Hypothesis would have been a good idea?

    # default num_trials to twice the field space dimension
    basis_shape = element.basis_shape()
    trial_count = (
        num_trials if num_trials is not None else np.prod(basis_shape, dtype=int) * 2
    )

    for _ in range(trial_count):
        yield (np.random.rand(*basis_shape) * 2 - 1)


def test_interpolate_field_linearity(
    element_and_local_coords: tuple[Element2D, np.ndarray],
):
    element, coords = element_and_local_coords

    basis_shape = element.basis_shape()

    x = coords[..., 0]
    y = coords[..., 1]
    basis_field = element.basis_fields()
    interped_fields = element.interpolate_field(basis_field, x, y)
    collapse_str = "ABCDEFGHIJKLMNOPQURSTUVWXYZ"[: len(element.basis_shape())]
    for field_coefs in random_basis_coefs(element):
        expected = fnp.einsum(
            ShapeComponent.STACK,
            f"{collapse_str},{collapse_str}",
            interped_fields,
            field_coefs,
        ).coefficients
        np.testing.assert_allclose(
            element.interpolate_field(
                Field(basis_shape, tuple(), field_coefs), x, y
            ).coefficients,
            expected,
            atol=1e-8,
        )


@pytest.mark.parametrize("elem", elements_for_field_gradient_finitediff_test)
def test_compute_field_gradient(elem: str):
    assert elem in element_test_localcoords, (
        "The automatic test_compute_field_gradient test requires a sample of local"
        f" coordinates, but {elem} is not a key in element_test_localcoords."
    )
    element = elements_to_test[elem]
    coords = element_test_localcoords[elem]
    basis_field = element.basis_fields()

    # =======verify local derivs=======
    grads = element.compute_field_gradient(basis_field)
    x_padded = coords[..., 0]
    y_padded = coords[..., 1]

    grad_at_pts = element.interpolate_field(grads, x_padded, y_padded)

    # use central finite differencing
    eps = 1e-5
    grad_finite_diff_x = (
        element.interpolate_field(basis_field, x_padded + eps, y_padded)
        - element.interpolate_field(basis_field, x_padded - eps, y_padded)
    ).coefficients / (2 * eps)
    grad_finite_diff_y = (
        element.interpolate_field(basis_field, x_padded, y_padded + eps)
        - element.interpolate_field(basis_field, x_padded, y_padded - eps)
    ).coefficients / (2 * eps)
    np.testing.assert_allclose(
        grad_at_pts.coefficients
        - np.stack((grad_finite_diff_x, grad_finite_diff_y), axis=-1),
        0,
        atol=(eps**2) * 1e2,  # central diff has error O(h^2). give a constant factor
    )


@pytest.mark.parametrize("elem", elements_for_grad_basis_dot_grad_field_test)
def test_integrate_grad_basis_dot_grad_field(elem: str, affine_transforms):
    element = elements_to_test[elem]
    field = element.basis_fields()
    jac_scale = field.stack[..., *(np.newaxis for _ in field.basis_shape)]
    ref_points = element.reference_element_position_field().coefficients

    for name, T in affine_transforms:
        points = Field(element.basis_shape(), (2,), T(ref_points))
        grads = element.compute_field_gradient(field, points)
        stiffness = element.integrate_grad_basis_dot_field(
            points, grads, None, jac_scale
        )
        stiffness_computed = element.integrate_grad_basis_dot_grad_field(
            points, field, None, jac_scale
        )
        np.testing.assert_allclose(
            stiffness_computed.coefficients,
            stiffness.coefficients,
            err_msg=(
                "Failed agreement between integrate_grad_basis_dot_field and"
                f" integrate_grad_basis_dot_field for element transformed by '{name}'"
            ),
            verbose=True,
            atol=1e-8,
        )


def test_mass_matrix(element, affine_transforms):
    basis_field = element.basis_fields()
    basis_shape = element.basis_shape()
    ref_points = element.reference_element_position_field().coefficients
    basis_size = np.prod(basis_shape, dtype=int)

    for name, T in affine_transforms:
        pos_matrix = Field(element.basis_shape(), (2,), T(ref_points))

        mat_by_integ_basis = element.integrate_basis_times_field(
            pos_matrix, basis_field
        )
        mat_by_method = element.mass_matrix(pos_matrix)

        np.testing.assert_allclose(
            mat_by_method.coefficients, mat_by_integ_basis.coefficients, atol=1e-8
        )

        # and also test a few elements individually:
        for _ in range(10):
            indI = np.unravel_index(np.random.randint(basis_size), basis_shape)
            indJ = np.unravel_index(np.random.randint(basis_size), basis_shape)
            mat_inds = element.mass_matrix(pos_matrix, indices=indI + indJ)
            np.testing.assert_allclose(
                mat_inds.coefficients,
                mat_by_method.coefficients[..., *(indI + indJ)],
                atol=1e-8,
            )


def test_interpolate_field_gradient(
    element_and_local_coords: tuple[Element2D, np.ndarray], affine_transforms
):
    element, coords = element_and_local_coords
    basis_shape = element.basis_shape()

    x = coords[..., 0]
    y = coords[..., 1]
    x_padded = x[..., *(np.newaxis for ax in basis_shape)]
    y_padded = y[..., *(np.newaxis for ax in basis_shape)]
    basis_field = element.basis_fields()
    interped_grads = element.interpolate_field(
        element.compute_field_gradient(basis_field), x_padded, y_padded
    )
    np.testing.assert_allclose(
        element.interpolate_field_gradient(
            basis_field, x_padded, y_padded
        ).coefficients,
        interped_grads.coefficients,
        atol=1e-8,
    )
    ref_points = element.reference_element_position_field().coefficients
    for name, T in affine_transforms:
        pos_matrix = Field(element.basis_shape(), (2,), T(ref_points))

        interped_grads = element.interpolate_field(
            element.compute_field_gradient(basis_field, pos_matrix),
            x_padded,
            y_padded,
        )
        np.testing.assert_allclose(
            element.interpolate_field_gradient(
                basis_field, x_padded, y_padded, pos_matrix
            ).coefficients,
            interped_grads.coefficients,
            atol=1e-8,
        )


def test_incompatible_fields_throw(element: Element2D):
    pos_field = element.reference_element_position_field()
    pos_stack = pos_field.stack[np.newaxis].stack[np.zeros(10, dtype=int)]
    field_stack = Field(
        pos_field.basis_shape, tuple(), np.zeros((2,) + pos_field.basis_shape)
    )
    field_stack_incomp = Field(
        pos_field.basis_shape, tuple(), np.zeros((3,) + pos_field.basis_shape)
    )
    bad_basis = Field(
        pos_field.basis_shape + (2,), tuple(), np.zeros(pos_field.basis_shape + (2,))
    )
    bad_basis_posfield = Field(
        pos_field.basis_shape + (2,), (2,), np.zeros(pos_field.basis_shape + (2, 2))
    )
    X_incomp = np.array([0, 0, 0, 0])
    Y_incomp = np.array([0, 0, 0])

    with pytest.raises(ValueError):  # incompatible stacks
        element.interpolate_field(field_stack, X_incomp, Y_incomp)
    with pytest.raises(ValueError):  # wrong basis
        element.interpolate_field(bad_basis, 0, 0)
    with pytest.raises(ValueError):  # wrong basis (no pos_field)
        element.compute_field_gradient(bad_basis)
    with pytest.raises(ValueError):  # incompatible stacks
        element.compute_field_gradient(field_stack, pos_field=pos_stack)
    with pytest.raises(ValueError):  # posfield shape != (2,)
        element.compute_field_gradient(field_stack, pos_field=field_stack)
    with pytest.raises(ValueError):  # wrong basis
        element.compute_field_gradient(bad_basis, pos_field=pos_stack)
    with pytest.raises(ValueError):  # incompatible stacks (no posfield)
        element.interpolate_field_gradient(field_stack, X_incomp, Y_incomp)
    with pytest.raises(ValueError):  # wrong basis (no posfield)
        element.interpolate_field_gradient(bad_basis, 0, 0)
    with pytest.raises(ValueError):  # incompatible stacks
        element.interpolate_field_gradient(field_stack, 0, 0, pos_stack)
    with pytest.raises(ValueError):  # incompatible stacks from X,Y
        element.interpolate_field_gradient(field_stack, X_incomp, Y_incomp, pos_field)
    with pytest.raises(ValueError):  # wrong basis
        element.interpolate_field_gradient(
            Field(tuple(), tuple(), 1), 0, 0, bad_basis_posfield
        )
    with pytest.raises(ValueError):  # posfield shape != (2,)
        element.interpolate_field_gradient(field_stack, 0, 0, field_stack)
    for integfunc in [
        element.integrate_field,
        element.integrate_basis_times_field,
        element.integrate_grad_basis_dot_field,
        element.integrate_grad_basis_dot_grad_field,
    ]:
        with pytest.raises(ValueError):  # posfield shape != (2,)
            integfunc(field_stack, field_stack)
        with pytest.raises(ValueError):  # wrong basis
            integfunc(bad_basis_posfield, Field(tuple(), tuple(), 1))
        with pytest.raises(ValueError):  # jac incompatible
            integfunc(pos_field, field_stack, jacobian_scale=field_stack_incomp)
        with pytest.raises(ValueError):  # field incompatible
            integfunc(pos_field, field_stack_incomp, jacobian_scale=field_stack)
        with pytest.raises(ValueError):  # posfield incompatible
            integfunc(pos_stack, field_stack, jacobian_scale=field_stack)
        with pytest.raises(ValueError):  # posfield shape != (2,)
            integfunc(field_stack, field_stack, jacobian_scale=field_stack)
    with pytest.raises(ValueError):  # wrong basis
        element.mass_matrix(bad_basis_posfield)
    with pytest.raises(ValueError):  # posfield shape != (2,)
        element.mass_matrix(field_stack)
    with pytest.raises(ValueError):  # jac incompatible
        element.mass_matrix(pos_stack, jacobian_scale=field_stack_incomp)
