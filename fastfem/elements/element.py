import abc
import numpy as np
from fastfem.fields.field import (
    Field as FieldType,
)

import warnings
import collections.abc as colltypes
from numpy.typing import NDArray, ArrayLike


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

    # this emulates a member class definition
    def Field(
        self, field: NDArray, is_const: bool, field_shape: tuple[int, ...] = tuple()
    ) -> FieldType:
        """Converts a `numpy` array to a `Field` object that can be used by this
        element class.

        Args:
            field (NDArray): An array representing the field.
            is_const (bool): If `True`, the field is considered
                constant, so `field.shape` is the same as its pointwise shape. For
                scalar fields, this means that `field.shape` is the field's stack shape.
                Otherwise, the first `len(element.basis_shape())` axes of `field` are
                taken to be the basis axes.
            field_shape (tuple[int, ...], optional): The shape of the field (pointwise,
            per-element). Defaults to tuple() for a scalar field.

        Raises:
            ValueError: when `is_const` is `False`, and the first
                `len(element.basis_shape())` axes of `field` are incompatible with
                `element.basis_shape()`.

        Returns:
            Field: An object corresponding to the given field.
        """
        basis_shape = self.basis_shape()
        return FieldType(
            basis_shape,
            field_shape,
            (
                field[*tuple(np.newaxis for _ in range(len(basis_shape))), ...]
                if is_const
                else field
            ),
        )

    def basis_fields(self) -> FieldType:
        """
        Returns a stack of scalar fields representing the basis elements.
        This is the identity matrix within the `basis_shape` indexing. That is,
        for two multi-indices, `I` and `J`,
        `basis_fields.coefficients[*I,*J] = (I == J)`.

        Returns:
            Field: The Field object with a stack of shape `basis_shape()` representing
                each basis function as a scalar field.
        """
        basis_shape = self.basis_shape()
        field = np.zeros(basis_shape + basis_shape)
        basis_size = np.prod(basis_shape, dtype=int)
        enumeration = np.unravel_index(np.arange(basis_size), basis_shape)
        field[*enumeration, *enumeration] = 1
        return FieldType(basis_shape, tuple(), field)

    @abc.abstractmethod
    def reference_element_position_matrix(self) -> FieldType:
        """
        The position field of the reference (un-transformed) element. This is a vector
        field.

        Returns:
            Field: A field of shape `(basis_shape(),(),(2,))` corresponding to the
            position field.
        """
        pass

    @abc.abstractmethod
    def interpolate_field(
        self,
        field: FieldType,
        X: NDArray,
        Y: NDArray,
    ) -> NDArray:
        """Evaluates field at (X,Y) in reference coordinates.
        The result is an array of values `field(X,Y)`.

        X and Y must have a compatible shape with `field.stack_shape`.

        Args:
            field (Field): the field to interpolate
                representing the field to be interpolated.
            X (NDArray): x values (in reference coordinates).
            Y (NDArray): y values (in reference coordinates).

        Returns:
            NDArray: The interpolated values `field(X,Y)`
        """
        pass

    @abc.abstractmethod
    def compute_field_gradient(
        self,
        field: FieldType,
        pos_matrix: FieldType | None = None,
    ) -> FieldType:
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
            field (Field): an array of shape (*basis_shape,...,*fieldshape)
                representing the field to be interpolated.
            pos_matrix (Field | None, optional): If set, `pos_matrix` specifies
                the position fields of the element, and the gradient will be computed in
                Cartesian coordinates. This method supports element-stacking.
                Defaults to None.

        Returns:
            Field: The field of the gradient.
        """
        pass

    def interpolate_field_gradient(
        self,
        field: FieldType,
        X: NDArray,
        Y: NDArray,
        pos_matrix: FieldType | None = None,
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
            field (Field): an array of shape (*basis_shape,*fieldshape)
                representing the field to be interpolated.
            X (NDArray): x values (in reference coordinates).
            Y (NDArray): y values (in reference coordinates).
            pos_matrix (Field | None, optional): If set, `pos_matrix` specifies
                the position fields of the element, and the gradient will be computed in
                Cartesian coordinates. Defaults to None.

        Returns:
            NDArray: an array representing the gradient of the field evaluated
                at each point.
        """
        self._dev_warn(
            "Element.interpolate_field_gradient() called, "
            + "which delegates to compute_field_gradient() and interpolate_field()"
        )
        return self.interpolate_field(
            self.compute_field_gradient(field, pos_matrix),
            X,
            Y,
        )

    @abc.abstractmethod
    def integrate_field(
        self,
        pos_matrix: FieldType,
        field: FieldType,
        jacobian_scale: FieldType = FieldType(tuple(), tuple(), 1),
    ) -> NDArray:
        """
        Integrates `field` $f$ over the element. The result is the value of
        $\\int \\alpha f ~dV$ over the element, as an array of shape
        `(...,*field_shape)`.

        Args:
            pos_matrix (Field): an array representing the positions of the
                    element nodes. This is of the shape `(basis_shape,...,2)`
            field (Field): an array of shape (*basis_shape,...,*fieldshape)
                representing the field to be interpolated.
            fieldshape (tuple[int,...], optional): the shape of `field` pointwise.
                Defaults to tuple(), representing a scalar field.
            jacobian_scale (Field, optional): an array of shape
                `(*basis_shape,...)` representing the scalar factor $\\alpha$.
                Defaults to 1.

        Returns:
            NDArray: The resultant integral, an array of shape `(...,*fieldshape)`.
        """
        pass

    @abc.abstractmethod
    def integrate_basis_times_field(
        self,
        pos_matrix: FieldType,
        field: FieldType,
        indices: colltypes.Sequence[ArrayLike] | None = None,
        jacobian_scale: FieldType = FieldType(tuple(), tuple(), 1),
    ) -> NDArray:
        """
        Computes the integral $\\int \\alpha \\phi_i f ~ dV$
        for fields $f$ and $\\alpha$, on this element, given by `field` and
        `jacobian_scale`, respectively. When None, `jacobian_scale` is interpreted
        as the identity.

        Args:
            pos_matrix (Field): an array representing the positions of the
                    element nodes. This is of the shape `(basis_shape,...,2)`
            field (Field): an array of shape `(*basis_shape,...,*fieldshape)`
                representing the field to be interpolated.
            fieldshape (tuple[int,...], optional): the shape of `field` pointwise.
                Defaults to tuple(), representing a scalar field.
            indices (colltypes.Sequence[ArrayLike] | None, optional): Indices of the
                    basis
                    functions to integrate against, or None if the integral should be
                    computed against every basis field. Defaults to None.
            jacobian_scale (Field, optional): an array of shape
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
        pos_matrix: FieldType,
        indices: colltypes.Sequence[np.ndarray] | None = None,
        jacobian_scale: FieldType = FieldType(tuple(), tuple(), 1),
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
            pos_matrix (Field): an array representing the positions of the
                    element nodes. This is of the shape `(basis_shape,...,2)`
            indices (colltypes.Sequence[ArrayLike] | None, optional): Indices of the mass
                    matrix to access, or None if the whole matrix should be returned.
                    Defaults to None.
            jacobian_scale (Field, optional): an array of shape
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
            None if indices is None else indices[: len(self.basis_shape())],
            jacobian_scale,
        )
        return mat if indices is None else mat[*indices]

    @abc.abstractmethod
    def integrate_grad_basis_dot_field(
        self,
        pos_matrix: FieldType,
        field: FieldType,
        is_field_upper_index: bool,
        indices: colltypes.Sequence[ArrayLike] | None = None,
        jacobian_scale: FieldType = FieldType(tuple(), tuple(), 1),
    ) -> NDArray:
        """
        Computes the integral $\\int \\alpha \\nabla \\phi_i \\cdot f ~ dV$
        for a field $f$, on this element. The dot product takes the last axis of
        `field`, which must have size equal to the domain dimension.

        Args:
            pos_matrix (Field): an array representing the positions of the
                    element nodes. This is of the shape `(basis_shape,...,2)`
            field (Field): an array of shape (*basis_shape,...,*fieldshape)
                representing the field to be interpolated.
            is_field_upper_index (bool): True if the last axis of field is treated as
                an upper (contravariant / vector) index. False if it is a lower
                (covariant / covector) index.
            indices (colltypes.Sequence[ArrayLike] | None, optional): Indices of the basis
                    functions to integrate against, or None if the integral should be
                    computed against every basis field. Defaults to None.
            jacobian_scale (Field, optional): an array of shape
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
        pos_matrix: FieldType,
        field: FieldType,
        indices: colltypes.Sequence[ArrayLike] | None = None,
        jacobian_scale: FieldType = FieldType(tuple(), tuple(), 1),
    ) -> NDArray:
        """
        Computes the integral $\\int \\alpha \\nabla \\phi_i \\cdot \\nabla f ~ dV$
        for a field $f$, on this element. The dot product is over the gradients.

        Args:
            pos_matrix (Field): an array representing the positions of the
                    element nodes. This is of the shape `(basis_shape,...,2)`
            field (Field): an array of shape (*basis_shape,...,*fieldshape)
                representing the field to be interpolated.
            indices (colltypes.Sequence[ArrayLike] | None, optional): Indices of the basis
                    functions to integrate against, or None if the integral should be
                    computed against every basis field. Defaults to None.
            jacobian_scale (Field, optional): an array of shape
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
            self.compute_field_gradient(field, pos_matrix),
            False,
            indices,
            jacobian_scale,
        )

    # TODO boundary integrals,
