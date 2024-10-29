import abc
import numpy as np


import warnings
import collections.abc as colltypes
from numpy.typing import ArrayLike, NDArray
import typing


class Element2D(abc.ABC):
    """
    Handles the management of element operations. Element classes define operations
    to compute integrals on them. The elements themselves are handled in a data-driven
    way, where positions and fields are handled as arrays of coefficients of the shape
    functions defined by the element class. As isoparametric elements, positions
    and fields use the same set of shape functions.
    """

    def _dev_warn(self, message: str):
        """To be used by developers, only.

        This method is called by the base `Element` class to warn the use of default
        (most likely unoptimized, or potentially undesired)
        strategies to compute certain values.

        For example, `Element.mass_matrix()` computes the mass matrix by delegating to
        `integrate_basis_times_field(...,field=basis_fields())`, which may not be very
        efficient.

        This method can be overridden by the developer to suppress these warnings,
        and new ones can be called oustide of the `Element` base class without issue.

        Args:
            message (str): _description_

        Returns:
            _type_: _description_
        """
        warnings.warn(f"Element developer message: {message}")

    @abc.abstractmethod
    def basis_shape(self) -> tuple[int, ...]:
        """
        Each element has a basis with a corresponding multi-index. The shape
        corresponding to the multi-index is given by this method. Any function defined
        on elements is passed into `element`'s methods as a tensor `f` of coefficients
        for the basis, the indices of which should be the leading indices of `f.shape`.

        Returns:
            tuple[int, ...]: a tuple representing the shape of the array corresponding
                to the basis coefficients. A scalar field `f`, given as an array is
                expected to have shape `f.shape == element.basis_shape()`.
        """
        pass

    def basis_fields(self) -> NDArray:
        """
        Returns a field representing the basis elements within that basis.
        This is the identity matrix within the `basis_shape` indexing. That is,
        for two multi-indices, `I` and `J`, `basis_fields[*I,*J] = (I == J)`.

        Returns:
            NDArray: the identity matrix of shape `(*basis_shape,*basis_shape)`
                representing a vector field of the basis fields.
        """
        basis_shape = self.basis_shape()
        field = np.zeros(basis_shape + basis_shape)
        basis_size = np.prod(basis_shape, dtype=int)
        enumeration = np.unravel_index(np.arange(basis_size), basis_shape)
        field[*enumeration, *enumeration] = 1
        return field

    @abc.abstractmethod
    def boundary_count(self) -> int:
        """The number of boundaries that correspond to this element.

        Returns:
            int: the number of boundaries to the element
        """
        pass

    @abc.abstractmethod
    def reference_element_position_matrix(self) -> NDArray:
        """
        The position field of the reference (un-transformed) element. This is a vector
        field.

        Returns:
            NDArray: An array of shape `(*element.basis_shape(), 2)`
        """
        pass

    @abc.abstractmethod
    def interpolate_field(
        self,
        field: ArrayLike,
        X: ArrayLike,
        Y: ArrayLike,
        fieldshape: tuple[int, ...] = tuple(),
    ) -> typing.Any:
        """Evaluates field at (X,Y) in reference coordinates.
        The result is an array of values `field(X,Y)`.

        `field` is of shape `(*basis_shape,...,*fieldshape)`, where the internal
        ellipses `...` broadcasts with X and Y
        using numpy's rules for broadcasting.

        X and Y must have compatible shape, broadcasting to pointshape.

        Args:
            field (ArrayLike): an array of shape (degree+1,degree+1,*fieldshape)
                representing the field to be interpolated.
            X (ArrayLike): x values (in reference coordinates).
            Y (ArrayLike): y values (in reference coordinates).
            fieldshape (tuple[int,...], optional): the shape of `field` at each point.
                Defaults to tuple() for a scalar field.

        Returns:
            typing.Any: The interpolated values `field(X,Y)`
        """
        pass

    def reference_to_real(
        self, pos_matrix: ArrayLike, X: ArrayLike, Y: ArrayLike
    ) -> NDArray:
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
            pos_matrix (ArrayLike): an array representing the positions of the
                    element nodes. This is of the shape `(*basis_shape,...,2)`.
            X (ArrayLike): X coordinates of the points to map. The
                    shape of X must be compatible with the shape of Y.
            Y (ArrayLike): Y coordinates of the points to map. The
                    shape of Y must be compatible with the shape of X.

        Returns:
            NDArray: An array of real coordinates that (X,Y) map to.
        """
        return self.interpolate_field(pos_matrix, X, Y, fieldshape=(2,))

    @abc.abstractmethod
    def compute_field_gradient(
        self,
        field: ArrayLike,
        pos_matrix: ArrayLike | None = None,
        fieldshape: tuple[int, ...] = tuple(),
    ) -> NDArray:
        """
        Calculates the gradient of a field f, returning a new field.
        The result is an array of shape `(*basis_shape,...,*fieldshape,2)`,
        where field is of shape `(*basis_shape,...,*fieldshape)`.

        The last index is the coordinate of the derivative.

        This gradient can be computed in either reference space (with respect
        to the coordinates of the reference element), or in global space.
        If a global cartesian gradient should be calculated, then pos_matrix
        must be set to the coordinate matrix of the element. Otherwise,
        pos_matrix can be kept None.

        Args:
            field (ArrayLike): an array of shape (*basis_shape,...,*fieldshape)
                representing the field to be interpolated.
            pos_matrix (ArrayLike | None, optional): If set, `pos_matrix` specifies
                the position fields of the element, and the gradient will be computed in
                Cartesian coordinates. This method supports element-stacking.
                Defaults to None.
            fieldshape (tuple[int,...], optional): the shape of `field` pointwise.
                Defaults to tuple(), representing a scalar field.

        Returns:
            NDArray: an array representing the gradient of the field evaluated
                at each point.
        """
        pass

    def interpolate_field_gradient(
        self,
        field: ArrayLike,
        X: ArrayLike,
        Y: ArrayLike,
        pos_matrix: ArrayLike | None = None,
        fieldshape: tuple[int, ...] = tuple(),
    ) -> NDArray:
        """
        Calculates the gradient of a field f at the reference coordinates (X,Y).
        The result is an array of shape `(...,*fieldshape,2)`,
        where field is of shape `(*basis_shape,...,*fieldshape)`.

        X and Y must have compatible shape.

        The last index is the coordinate of the derivative.

        This gradient can be computed in either reference space (with respect
        to the coordinates of the reference element), or in global space.
        If a global cartesian gradient should be calculated, then pos_matrix
        must be set to the coordinate matrix of the element. Otherwise,
        pos_matrix can be kept None.

        Args:
            field (ArrayLike): an array of shape (*basis_shape,*fieldshape)
                representing the field to be interpolated.
            X (ArrayLike): x values (in reference coordinates).
            Y (ArrayLike): y values (in reference coordinates).
            pos_matrix (ArrayLike | None, optional): If set, `pos_matrix` specifies
                the position fields of the element, and the gradient will be computed in
                Cartesian coordinates. Defaults to None.
            fieldshape (tuple[int,...], optional): the shape of `field` pointwise.
                Defaults to tuple(), representing a scalar field.

        Returns:
            NDArray: an array representing the gradient of the field evaluated
                at each point.
        """
        self._dev_warn(
            "Element.interpolate_field_gradient() called, "
            + "which delegates to compute_field_gradient() and interpolate_field()"
        )
        return self.interpolate_field(
            self.compute_field_gradient(field, pos_matrix, fieldshape),
            X,
            Y,
            fieldshape + (2,),
        )

    def compute_deformation_gradient(
        self,
        pos_matrix: ArrayLike,
    ) -> NDArray:
        """
        Calculates the deformation gradient (often called the Jacobian matrix)
        $\\frac{\\partial \\Phi}{\\partial x}$, where $\\Phi$ maps reference coordinates
        to real coordinates.
        The result is a matrix field, an array of shape `(*basis_shape,...,2,2)` where
        the first new index specifies the coordinate in global space and the
        second index specifies the coordinate in reference space.

        `pos_matrix` is an array of shape (*basis_shape,...,2), where the ellipses
        are indices for element stacking.


        Args:
            pos_matrix (ArrayLike): an array representing the positions of the
                    element nodes. This is of the shape `(*basis_shape,...,2)`.


        Returns:
            NDArray: the computed deformation gradient.
        """
        return self.compute_field_gradient(pos_matrix, fieldshape=(2,))

    def interpolate_deformation_gradient(
        self, pos_matrix: ArrayLike, X: ArrayLike, Y: ArrayLike
    ) -> NDArray:
        """Calculates the deformation gradient (often called the Jacobian matrix)
        $\\frac{\\partial \\Phi}{\\partial x}$, where $\\Phi$ maps reference coordinates
        to real coordinates.
        This matrix is evaluated at the reference coordinates `(X,Y)`.
        X and Y must be broadcastable to the same shape.
        The result is an array with shape (...,2,2) where
        the first new index specifies the coordinate in global space and the
        second index specifies the coordinate in reference space.

        `pos_matrix` is an array of shape (*basis_shape,...,2) where
        `pos_matrix[i,j,...] = [x,y]` when `(x,y)` is the position of node i (along
        the x-axis) and j (along the y-axis). The ellipses are broadcasted along with
        `X` and `Y`.

        Args:
            pos_matrix (ArrayLike): an array representing the positions of the
                    element nodes. This is of the shape `(*basis_shape,...,2)`.
            X (ArrayLike): X coordinates of the points to map. The
                    shape of X must be compatible with the shape of Y.
            Y (ArrayLike): Y coordinates of the points to map. The
                    shape of Y must be compatible with the shape of X.


        Returns:
            NDArray: the evaluated deformation gradient.
        """
        return self.interpolate_field_gradient(pos_matrix, X, Y, fieldshape=(2,))

    @abc.abstractmethod
    def integrate_field(
        self,
        pos_matrix: ArrayLike,
        field: ArrayLike,
        fieldshape: tuple[int, ...] = tuple(),
        jacobian_scale: ArrayLike = 1,
    ) -> NDArray:
        """
        Integrates `field` $f$ over the element. The result is the value of
        $\\int \\alpha f ~dV$ over the element, as an array of shape
        `(...,*field_shape)`.

        Args:
            pos_matrix (ArrayLike): an array representing the positions of the
                    element nodes. This is of the shape `(basis_shape,...,2)`
            field (ArrayLike): an array of shape (*basis_shape,...,*fieldshape)
                representing the field to be interpolated.
            fieldshape (tuple[int,...], optional): the shape of `field` pointwise.
                Defaults to tuple(), representing a scalar field.
            jacobian_scale (ArrayLike, optional): an array of shape
                `(*basis_shape,...)` representing the scalar factor $\\alpha$.
                Defaults to 1.

        Returns:
            NDArray: The resultant integral, an array of shape `(...,*fieldshape)`.
        """
        pass

    @abc.abstractmethod
    def integrate_basis_times_field(
        self,
        pos_matrix: ArrayLike,
        field: ArrayLike,
        fieldshape: tuple[int, ...] = tuple(),
        indices: colltypes.Sequence[ArrayLike] | None = None,
        jacobian_scale: ArrayLike = 1,
    ) -> NDArray:
        """
        Computes the integral $\\int \\alpha \\phi_i f ~ dV$
        for fields $f$ and $\\alpha$, on this element, given by `field` and
        `jacobian_scale`, respectively. When None, `jacobian_scale` is interpreted
        as the identity.

        Args:
            pos_matrix (ArrayLike): an array representing the positions of the
                    element nodes. This is of the shape `(basis_shape,...,2)`
            field (ArrayLike): an array of shape `(*basis_shape,...,*fieldshape)`
                representing the field to be interpolated.
            fieldshape (tuple[int,...], optional): the shape of `field` pointwise.
                Defaults to tuple(), representing a scalar field.
            indices (colltypes.Sequence[ArrayLike] | None, optional): Indices of the basis
                    functions to integrate against, or None if the integral should be
                    computed against every basis field. Defaults to None.
            jacobian_scale (ArrayLike, optional): an array of shape
                `(*basis_shape,...)` representing the scalar factor $\\alpha$.
                Defaults to 1.

        Returns:
            NDArray: The resultant integral, an array of shape
                `(indices.shape,...,*fieldshape)`, or `(*basis_shape,...,*fieldshape)`
                if `indices` is None.
        """
        pass

    def mass_matrix(
        self,
        pos_matrix: ArrayLike,
        indices: colltypes.Sequence[np.ndarray] | None = None,
        jacobian_scale: ArrayLike = 1,
    ) -> NDArray:
        """
        Recovers the mass matrix entries $\\int \\alpha \\phi_i \\phi_j ~ dV$
        for this element. `pos_matrix` has shape `(basis_shape,...,2)`, where the
        ellipses represent element stacking.

        The full mass matrix is of shape
        `(*basis_shape, *basis_shape,...)`, where the last indices match the element
        stack, but slices of the mass matrix can be
        obtained by passing into the `indices` argument. The implementation of
        `basis_mass_matrix` should assume that `indices` argument is smaller than the
        whole matrix, so if `M[*indices]` is larger than `M`, one should use
        `basis_mass_matrix(pos_matrix)[*indices,...]` instead of
        `basis_mass_matrix(pos_matrix,indices)`.

        Args:
            pos_matrix (ArrayLike): an array representing the positions of the
                    element nodes. This is of the shape `(basis_shape,...,2)`
            indices (colltypes.Sequence[ArrayLike] | None, optional): Indices of the mass
                    matrix to access, or None if the whole matrix should be returned.
                    Defaults to None.
            jacobian_scale (ArrayLike, optional): an array of shape
                `(*basis_shape,...)` representing the scalar factor $\\alpha$.
                Defaults to 1.

        Returns:
            NDArray: an array representing the mass matrix, or portions thereof.
        """

        self._dev_warn(
            "Element.mass_matrix() called, which delegates "
            + "to integrate_basis_times_field()"
        )
        mat = self.integrate_basis_times_field(
            pos_matrix,
            self.basis_fields(),
            self.basis_shape(),
            None if indices is None else indices[: len(self.basis_shape())],
            jacobian_scale,
        )
        return mat if indices is None else mat[*indices]

    @abc.abstractmethod
    def integrate_grad_basis_dot_field(
        self,
        pos_matrix: ArrayLike,
        field: ArrayLike,
        is_field_upper_index: bool,
        fieldshape: tuple[int, ...],
        indices: colltypes.Sequence[ArrayLike] | None = None,
        jacobian_scale: ArrayLike = 1,
    ) -> NDArray:
        """
        Computes the integral $\\int \\alpha \\nabla \\phi_i \\cdot f ~ dV$
        for a field $f$, on this element. The dot product takes the last axis of
        `field`, which must have size equal to the domain dimension.

        Args:
            pos_matrix (ArrayLike): an array representing the positions of the
                    element nodes. This is of the shape `(basis_shape,...,2)`
            field (ArrayLike): an array of shape (*basis_shape,...,*fieldshape)
                representing the field to be interpolated.
            is_field_upper_index (bool): True if the last axis of field is treated as
                an upper (contravariant / vector) index. False if it is a lower
                (covariant / covector) index.
            fieldshape (tuple[int,...], optional): the shape of `field` pointwise.
                Defaults to tuple(), representing a scalar field.
            indices (colltypes.Sequence[ArrayLike] | None, optional): Indices of the basis
                    functions to integrate against, or None if the integral should be
                    computed against every basis field. Defaults to None.
            jacobian_scale (ArrayLike, optional): an array of shape
                `(*basis_shape,...)` representing the scalar factor $\\alpha$.
                Defaults to 1.

        Returns:
            NDArray: The resultant integral, an array of shape
                `(indices.shape,...,*fieldshape)`, or `(*basis_shape,...,*fieldshape)`
                if `indices` is None.
        """
        pass

    def integrate_grad_basis_dot_grad_field(
        self,
        pos_matrix: ArrayLike,
        field: ArrayLike,
        fieldshape: tuple[int, ...] = tuple(),
        indices: colltypes.Sequence[ArrayLike] | None = None,
        jacobian_scale: ArrayLike = 1,
    ) -> NDArray:
        """
        Computes the integral $\\int \\alpha \\nabla \\phi_i \\cdot \\nabla f ~ dV$
        for a field $f$, on this element. The dot product is over the gradients.

        Args:
            pos_matrix (ArrayLike): an array representing the positions of the
                    element nodes. This is of the shape `(basis_shape,...,2)`
            field (ArrayLike): an array of shape (*basis_shape,...,*fieldshape)
                representing the field to be interpolated.
            fieldshape (tuple[int,...]): the shape of `field` pointwise.
            indices (colltypes.Sequence[ArrayLike] | None, optional): Indices of the basis
                    functions to integrate against, or None if the integral should be
                    computed against every basis field. Defaults to None.
            jacobian_scale (ArrayLike, optional): an array of shape
                `(*basis_shape,...)` representing the scalar factor $\\alpha$.
                Defaults to 1.

        Returns:
            NDArray: The resultant integral, an array of shape
                `(indices.shape,...,*fieldshape)`, or `(*basis_shape,...,*fieldshape)`
                if `indices` is None.
        """
        self._dev_warn(
            "Element.integrate_grad_basis_dot_grad_field() called, "
            + "which delegates to integrate_grad_basis_dot_field()"
        )
        return self.integrate_grad_basis_dot_field(
            pos_matrix,
            self.compute_field_gradient(field, pos_matrix, fieldshape),
            False,
            fieldshape + (2,),
            indices,
            jacobian_scale,
        )

    # TODO boundary integrals,
