import abc
import numpy as np

class Element(abc.ABC):
    """
    Handles the management of element operations. Element classes define operations
    to compute integrals on them. The elements themselves are handled in a data-driven
    way, where positions and fields are handled as arrays of coefficients of the shape
    functions defined by the element class. As isoparametric elements, positions
    and fields use the same set of shape functions.
    """

    @abc.abstractmethod
    def basis_shape(self) -> tuple[int,...]:
        """Returns a tuple representing the shape of the array corresponding to the
        basis coefficients. A scalar field `f`, given as an array is expected to have
        shape `f.shape == element.basis_shape()`
        """
        pass

    @abc.abstractmethod
    def reference_to_real(self, pos_matrix: np.ndarray,
            X: np.ndarray | float,
            Y: np.ndarray | float) -> np.ndarray:
        """Maps the points (X,Y) from reference coordinates
        to real positions. The result is an array of shape
        `(*point_shape,2)`, where the last index is the dimension, and
        `point_shape = np.broadcast_shapes(X.shape,Y.shape)`.

        `pos_matrix` is an array of shape `(*basis_shape,...,2)`
        for the element's expected shape `basis_shape`, which is obtained by
        `element.basis_shape()`. The internal ellipses `...` map using numpy's rules
        with point_shape.
        pos_matrix[...,i] is the i^th coordinate of the position relevant to
        the basis function indexed in (...).

        Args:
            pos_matrix (ndarray): an array representing the positions of the
                    element nodes. This is of the shape `(*basis_shape,...,2)`.
            X (ndarray | float): X coordinates of the points to map. The
                    shape of X must be compatible with the shape of Y.
            Y (ndarray | float): Y coordinates of the points to map. The
                    shape of Y must be compatible with the shape of X.

        Returns:
            ndarray: An array of real coordinates that (X,Y) map to.
        """
        pass

    @abc.abstractmethod
    def basis_mass_matrix(self, pos_matrix: np.ndarray) -> np.ndarray:
        """Recovers the mass matrix entries $\\int \\phi_i \\phi_j ~ dV$
        for this element.

        Args:
            pos_matrix (ndarray): an array representing the positions of the
                    element nodes. This is of the shape `(basis_shape,...,2)`

        Returns:
            ndarray: an array representing the mass matrix.
            !!WARNING!! spectral elements have a diagonal mass matrix, so we need to
            find a way to respect sparseness!
        """
        pass

    @abc.abstractmethod
    def basis_stiffness_matrix_times_field(self, pos_matrix: np.ndarray,
            field: np.ndarray, fieldshape: tuple[int,...] = tuple()) -> np.ndarray:
        """Computes the vector $\\int (\\nabla \\phi_i) \\cdot \\nabla f ~ dV$ of the
        $H^1$ products on the element
        over the whole basis on the element. Using the multi-index 'I' for the
        field f, we can write these terms as
        $\\int (\\partial_j \\phi_i) g_{jk} (\\partial_k f_I) ~ dV$ where $g_{jk}$
        represents the inner product on the dual space (since $\\partial$ adds a
        covariant index). The result's index order is (i,...,I), where i may be a
        multi-index, based on basis_shape, and is placed into a numpy
        array of shape (*basis_shape,...,*field_shape).

        Args:
            pos_matrix (ndarray): _description_
            field (ndarray): _description_
            fieldshape (tuple[int,...], optional): The shape of field pointwise.
                    Defaults to tuple() for scalar fields.

        Returns:
            ndarray: an array representing the stiffness matrix
        """
        pass

    # TODO boundary integrals, other field * basis integral configurations
