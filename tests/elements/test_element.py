import pytest


from fastfem.elements import spectral_element
from fastfem.elements.element import Element2D

# ======================================================================================
#            Adding elements -- this *should* be sufficient to test that element
# ======================================================================================
elements_to_test = dict()


# register spectral elements of different orders
for i in [3, 4, 5]:
    # class itself
    elements_to_test[f"spectral{i}"] = spectral_element.SpectralElement2D(i)


# ======================================================================================


@pytest.fixture(scope="module", params=elements_to_test.keys())
def element(request):
    return elements_to_test[request.param]


def test_basis_and_reference_shapes(element: Element2D):
    """Validates the shapes of basis_fields() and reference_element_position_matrix()
    against basis_shape().
    """
    shape = element.basis_shape()
    basis = element.basis_fields()

    assert basis.shape == 2 * shape, (
        "basis_fields() should have shape 2*basis_shape(). "
        + f"basis_fields().shape: {basis.shape} basis_shape(): {shape}."
    )

    ref_elem_pts = element.reference_element_position_matrix()

    assert ref_elem_pts.shape[:-1] == shape, (
        "reference_element_position_matrix() should have shape "
        + "(*basis_shape(), ndims). "
        + f"reference_element_position_matrix().shape: {ref_elem_pts.shape} "
        + f"basis_shape(): {shape}."
    )


def test_reference_deformation_gradient(element: Element2D):
    """Validates the reference position matrix's deformation gradient, which should
    be the identity.
    """
