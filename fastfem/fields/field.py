import dataclasses
import itertools
import typing
from dataclasses import dataclass
from enum import IntEnum
from types import ModuleType
from typing import Literal, Union

import jax
import numpy as _np  # noqa: ICN001
from numpy.typing import ArrayLike, NDArray

jnp = _np  # this line is just to make type hinting not complain about np()
HAS_JAX = True
try:
    import jax.numpy as jnp
except ImportError:
    HAS_JAX = False


def np_or_jnp(*use_jax) -> ModuleType:
    if any((entry.use_jax if isinstance(entry, Field) else entry) for entry in use_jax):
        if HAS_JAX:
            return jnp
        message = "JAX was not found, but was requested!"
        raise ValueError(message)
    return _np


np = _np


def _is_broadcastable(base: tuple[int, ...], *shapes: tuple[int, ...]) -> bool:
    """Checks if shapes are broadcastable into base by numpy broadcasting rules.

    Args:
        base (tuple[int,...]): the first (target) shape
        *shapes (tuple[int,...]): the shapes to be broadcasted

    Returns:
        bool: `True` if the shapes are compatible, and false otherwise.
    """
    return all(
        map(
            lambda x: (
                x[0] is not None
                and all(xi == x[0] or xi == 1 or xi is None for xi in x[1:])
            ),
            itertools.zip_longest(
                reversed(base), *(reversed(shape) for shape in shapes), fillvalue=None
            ),
        )
    )


def _is_compatible(*shapes: tuple[int, ...]) -> bool:
    """Checks if shapes are compatible by numpy broadcasting rules.

    Returns:
        bool: `True` if the shapes are compatible, and false otherwise.
    """
    return all(
        map(
            lambda xi: all(  # every shape[i] in x[i] must be compatible
                map(  # recover compatibility boolean
                    lambda fjxi: fjxi[1],
                    # fj(x[i]) = (axsize , axsize compatible with x[i][j]);  j = 0,...
                    itertools.accumulate(
                        xi,
                        func=lambda a, b: (
                            b if b != 1 else a[0],
                            (a[0] == b or a[0] == 1 or b == 1),
                        ),
                        initial=(1, True),
                    ),
                )
            ),
            # x[i] = (shape[i] for shape in shapes) : i = 0,...
            itertools.zip_longest(*(reversed(shape) for shape in shapes), fillvalue=1),
        )
    )


class FieldShapeError(Exception):
    pass


class ShapeComponent(IntEnum):
    STACK = 0
    BASIS = 1
    POINT = 2


@dataclass(frozen=True)
class FieldAxisIndex:
    component: ShapeComponent
    index: int

    @typing.overload
    def __getitem__(self, ind: Literal[0]) -> ShapeComponent: ...
    @typing.overload
    def __getitem__(self, ind: Literal[1]) -> int: ...

    def __getitem__(self, ind):
        if ind == 0:
            return self.component
        return self.index


FieldAxisIndexType = tuple[ShapeComponent, int] | FieldAxisIndex


def _verify_is_permutation(p: tuple[int, ...]) -> None:
    if not isinstance(p, tuple):
        message = "shape_order is not a permutation! (must be a tuple)"
        raise FieldShapeError(message)
    n = len(p)  # size of the permutation
    exists = [False] * n
    for i in p:
        if not isinstance(i, int):
            message = "shape_order is not a permutation! (all entries must be integers)"
            raise FieldShapeError(message)
        exists[i] = True

    if not all(exists):
        message = "shape_order is not a permutation! (must be a bijection)"
        raise FieldShapeError(message)


def _invert_permutation(p: tuple[int, ...]) -> tuple[int, ...]:
    # assume p is a permutation
    pinv = [-1] * len(p)
    for i, k in enumerate(p):
        pinv[k] = i
    return tuple(pinv)


def _reshape(
    field: "Field",
    component_selector: ShapeComponent,
    shape: int | tuple[int],
    order: Literal["C", "F", "A"] = "C",
    copy: bool | None = None,
) -> "Field":
    """This attempts to replicate the numpy "reshape" function.
    https://numpy.org/doc/stable/reference/generated/numpy.reshape.html

    reshape() applies numpy "reshape" to a given component. This function is also called
    when using the field accessor reshape methods. That is, `reshape(field,BASIS,s)` is
    the same as `field.basis.reshape(s)`
    Args:
        field (Field): The field to reshape
        component_selector (ShapeComponent): Which component to reshape.
        shape (int | tuple[int]): The target shape of the component. For any integer i,
            i is equivalent to (i,)
        order ({'C','F','A'}, optional): See the numpy documentation. Defaults to 'C'.
        copy (bool | None, optional): See the numpy documentation. Defaults to None.

    Raises:
        ValueError: when a copy operation is required, but `copy` is False.
    Returns:
        Field: The reshaped field. This is always a new object, but if data is copied,
        the underlying array is a view of the original.
    """
    # if not isinstance(shape, tuple):
    #     shape = (shape,)
    # start_ind = field._axis_field_to_numpy(FieldAxisIndex(component_selector, 0))
    # end_ind = field._axis_field_to_numpy(FieldAxisIndex(component_selector, -1)) + 1
    # shapes = [()] * 3
    # shapes[ShapeComponent.BASIS] = field.basis_shape
    # shapes[ShapeComponent.POINT] = field.point_shape
    # shapes[ShapeComponent.STACK] = field.stack_shape
    # shapes[component_selector] = shape
    # np_target_shape = shapes[0] + shapes[1] + shapes[2]
    # coefs = np_or_jnp(field.use_jax).reshape(
    #         field.coefficients, np_target_shape, order=order, copy=copy
    #     )
    # shape_ = coefs.shape[start_ind:end_ind]
    # return Field(
    #     shape_ if component_selector == ShapeComponent.BASIS else field.basis_shape,
    #     shape_ if component_selector == ShapeComponent.POINT else field.point_shape,
    #     coefs,
    #     shape_order=field.shape_order,
    #     use_jax=field.use_jax,
    # )
    if not isinstance(shape, tuple):
        shape = (shape,)
    # beginning of reshaped section
    start_ind = field._component_offset(component_selector)
    # end of reshaped section (excl)
    end_ind = start_ind + len(shape)
    shapes = [()] * 3
    shapes[field.shape_order[ShapeComponent.BASIS]] = field.basis_shape  # type: ignore
    shapes[field.shape_order[ShapeComponent.POINT]] = field.point_shape  # type: ignore
    shapes[field.shape_order[ShapeComponent.STACK]] = field.stack_shape  # type: ignore
    shapes[field.shape_order[component_selector]] = shape  # type: ignore
    np_target_shape = shapes[0] + shapes[1] + shapes[2]
    coefs = np_or_jnp(field.use_jax).reshape(
        field.coefficients, np_target_shape, order=order, copy=copy
    )
    shape_ = coefs.shape[start_ind:end_ind]
    return Field(
        shape_ if component_selector == ShapeComponent.BASIS else field.basis_shape,
        shape_ if component_selector == ShapeComponent.POINT else field.point_shape,
        coefs,
        shape_order=field.shape_order,
        use_jax=field.use_jax,
    )


class FieldConstructionError(FieldShapeError):
    """Called when constructing a field fails."""

    def __init__(self, basis_shape, point_shape, coeff_shape, shape_order, hint=None):
        errmsg = (
            f"Cannot construct Field object with basis_shape {basis_shape}, point_shape"
            f" {point_shape} given the coefficient shape {coeff_shape} and shape order"
            f" {shape_order}."
        )
        if hint is not None:
            errmsg += f" ({hint})"
        super().__init__(errmsg)


class FieldBasisAccessor:
    """This class is the type returned in the Field __get_attr__ for basis access.
    The sole purpose of this class is to provide the syntax for basis-targeted slicing
    in the same format as numpy.
    """

    def __init__(self, parent):
        self.parent = parent

    def __getitem__(self, slices):
        # do validation
        basis = _np.broadcast_to(0, self.parent.basis_shape)
        new_basis_shape = basis[slices].shape
        slicepad = (slice(None),) * (
            (
                len(self.parent.stack_shape)
                if self.parent.shape_order[ShapeComponent.STACK]
                < self.parent.shape_order[ShapeComponent.BASIS]
                else 0
            )
            + (
                len(self.parent.point_shape)
                if self.parent.shape_order[ShapeComponent.POINT]
                < self.parent.shape_order[ShapeComponent.BASIS]
                else 0
            )
        )
        if not isinstance(slices, tuple):
            slices = (slices,)
        return Field(
            new_basis_shape,
            self.parent.point_shape,
            self.parent.coefficients[*slicepad, *slices],
        )

    def reshape(
        self,
        shape: int | tuple[int],
        order: Literal["C", "F", "A"] = "C",
        copy: bool | None = None,
    ) -> "Field":
        """This attempts to replicate the numpy "reshape" function.
        https://numpy.org/doc/stable/reference/generated/numpy.reshape.html

        Delegates to numpy_similes.reshape()

        Args:
            field (Field): The field to reshape
            component_selector (ShapeComponent): Which component to reshape.
            shape (int | tuple[int]): The target shape of the component. For any integer i,
                i is equivalent to (i,)
            order ({'C','F','A'}, optional): See the numpy documentation. Defaults to 'C'.
            copy (bool | None, optional): See the numpy documentation. Defaults to None.

        Raises:
            ValueError: when a copy operation is required, but `copy` is False.
        Returns:
            Field: The reshaped field. This is always a new object, but if data is copied,
            the underlying array is a view of the original.
        """
        return _reshape(self.parent, ShapeComponent.BASIS, shape, order, copy)


class FieldStackAccessor:
    """This class is the type returned in the Field __get_attr__ for stack access.
    The sole purpose of this class is to provide the syntax for stack-targeted slicing
    in the same format as numpy.
    """

    def __init__(self, parent):
        self.parent = parent

    def __getitem__(self, slices):
        # do validation
        stack = np.broadcast_to(0, self.parent.stack_shape)
        new_stack_shape = stack[slices].shape  # noqa: F841
        slicepad = (slice(None),) * (
            (
                len(self.parent.basis_shape)
                if self.parent.shape_order[ShapeComponent.BASIS]
                < self.parent.shape_order[ShapeComponent.STACK]
                else 0
            )
            + (
                len(self.parent.point_shape)
                if self.parent.shape_order[ShapeComponent.POINT]
                < self.parent.shape_order[ShapeComponent.STACK]
                else 0
            )
        )
        if not isinstance(slices, tuple):
            slices = (slices,)
        return Field(
            self.parent.basis_shape,
            self.parent.point_shape,
            self.parent.coefficients[*slicepad, *slices],
        )

    def reshape(
        self,
        shape: int | tuple[int],
        order: Literal["C", "F", "A"] = "C",
        copy: bool | None = None,
    ) -> "Field":
        """This attempts to replicate the numpy "reshape" function.
        https://numpy.org/doc/stable/reference/generated/numpy.reshape.html

        Delegates to numpy_similes.reshape()

        Args:
            field (Field): The field to reshape
            component_selector (ShapeComponent): Which component to reshape.
            shape (int | tuple[int]): The target shape of the component. For any integer i,
                i is equivalent to (i,)
            order ({'C','F','A'}, optional): See the numpy documentation. Defaults to 'C'.
            copy (bool | None, optional): See the numpy documentation. Defaults to None.

        Raises:
            ValueError: when a copy operation is required, but `copy` is False.
        Returns:
            Field: The reshaped field. This is always a new object, but if data is copied,
            the underlying array is a view of the original.
        """
        return _reshape(self.parent, ShapeComponent.STACK, shape, order, copy)


class FieldPointAccessor:
    """This class is the type returned in the Field __get_attr__ for pointwise field-element
    access.
    The sole purpose of this class is to provide the syntax for field-targeted slicing
    in the same format as numpy.
    """

    def __init__(self, parent):
        self.parent = parent

    def __getitem__(self, slices):
        # do validation
        element = np.broadcast_to(0, self.parent.point_shape)
        new_point_shape = element[slices].shape
        slicepad = (slice(None),) * (
            (
                len(self.parent.basis_shape)
                if self.parent.shape_order[ShapeComponent.BASIS]
                < self.parent.shape_order[ShapeComponent.POINT]
                else 0
            )
            + (
                len(self.parent.stack_shape)
                if self.parent.shape_order[ShapeComponent.STACK]
                < self.parent.shape_order[ShapeComponent.POINT]
                else 0
            )
        )
        if not isinstance(slices, tuple):
            slices = (slices,)
        return Field(
            self.parent.basis_shape,
            new_point_shape,
            self.parent.coefficients[*slicepad, *slices],
        )

    def reshape(
        self,
        shape: int | tuple[int],
        order: Literal["C", "F", "A"] = "C",
        copy: bool | None = None,
    ) -> "Field":
        """This attempts to replicate the numpy "reshape" function.
        https://numpy.org/doc/stable/reference/generated/numpy.reshape.html

        Delegates to numpy_similes.reshape()

        Args:
            field (Field): The field to reshape
            component_selector (ShapeComponent): Which component to reshape.
            shape (int | tuple[int]): The target shape of the component. For any integer i,
                i is equivalent to (i,)
            order ({'C','F','A'}, optional): See the numpy documentation. Defaults to 'C'.
            copy (bool | None, optional): See the numpy documentation. Defaults to None.

        Raises:
            ValueError: when a copy operation is required, but `copy` is False.
        Returns:
            Field: The reshaped field. This is always a new object, but if data is copied,
            the underlying array is a view of the original.
        """
        return _reshape(self.parent, ShapeComponent.POINT, shape, order, copy)


@dataclass(eq=False, frozen=True, unsafe_hash=False, init=False)
class Field:
    """
    A class responsible for storing fields on elements as an `NDArray` of coefficients.
    There are 3 relevant shapes / axis sets to a field:

    - `basis_shape` - The shape of the basis. These axes represent the multi-index for
            the basis function.

    - `stack_shape` - The shape of the element stack. These axes represent the
            multi-index for the element.

    - `point_shape` - The shape of the field. These axes represent the pointwise,
            per-element tensor index.

    The shape of `coefficients` will be some permutation of
    `stack_shape + point_shape + basis_shape`. The order is specified by `shape_order`,
    which is a 3-tuple `(stack_location, field_location, basis_location)`, where each
    entry is an integer specifying the position relative to the other two shapes.
    """

    basis_shape: tuple[int, ...]
    stack_shape: tuple[int, ...]
    point_shape: tuple[int, ...]
    coefficients: NDArray | jax.Array
    shape_order: tuple[int, int, int] = dataclasses.field(repr=False, init=False)
    shape_order_inverse: tuple[ShapeComponent, ShapeComponent, ShapeComponent] = (
        dataclasses.field(repr=False, init=False)
    )
    use_jax: bool = dataclasses.field(repr=False, init=False)

    def __init__(
        self,
        basis_shape: tuple[int, ...],
        point_shape: tuple[int, ...],
        coefficients: ArrayLike,
        shape_order: tuple[int, int, int] = (0, 1, 2),
        use_jax: bool | None = None,
    ):
        _verify_is_permutation(shape_order)
        if not isinstance(coefficients, np.ndarray) and not isinstance(
            coefficients, jax.Array
        ):
            coefficients = np.array(coefficients)
        if use_jax is None:
            use_jax = isinstance(coefficients, jax.Array)
        cshape_orig = np.shape(coefficients)
        if len(cshape_orig) < len(basis_shape) + len(point_shape):
            coefficients = coefficients[
                *(
                    (np.newaxis,)
                    * (len(basis_shape) + len(point_shape) - len(cshape_orig))
                ),
                ...,
            ]
            cshape = np.shape(coefficients)
        else:
            cshape = cshape_orig

        # here, coefficients is at least as large as basis and field shapes combined

        # we need to place two markers to index the separations between basis, field,
        # and stack shapes; start with basis_shape (if not in middle)
        stack_start = 0
        stack_end = len(cshape)

        def cshape_slice_positives(a, b):
            return cshape[a:b]

        def cshape_slice_negatives(a, b):
            return cshape[a : (b if b != 0 else None)] if a != 0 else ()

        if shape_order[ShapeComponent.BASIS] == 0:
            if not _is_broadcastable(
                basis_shape, cshape_slice_positives(0, len(basis_shape))
            ):
                raise FieldConstructionError(
                    basis_shape,
                    point_shape,
                    cshape_orig,
                    shape_order,
                    hint="basis_shape cannot be broadcasted at the beginning",
                )
            stack_start = len(basis_shape)
        elif shape_order[ShapeComponent.BASIS] == 2:
            if not _is_broadcastable(
                basis_shape, cshape_slice_negatives(-len(basis_shape), 0)
            ):
                raise FieldConstructionError(
                    basis_shape,
                    point_shape,
                    cshape_orig,
                    shape_order,
                    hint="basis_shape cannot be broadcasted at the end",
                )
            stack_end -= len(basis_shape)
        # then do point_shape
        if shape_order[ShapeComponent.POINT] == 0:
            if not _is_broadcastable(
                point_shape, cshape_slice_positives(0, len(point_shape))
            ):
                raise FieldConstructionError(
                    basis_shape,
                    point_shape,
                    cshape_orig,
                    shape_order,
                    hint="point_shape cannot be broadcasted at the beginning",
                )
            # if basis_shape was in center, we now have the right offset for it
            if shape_order[ShapeComponent.BASIS] == 1:
                if not _is_broadcastable(
                    basis_shape,
                    cshape_slice_positives(
                        len(point_shape), (len(basis_shape) + len(point_shape))
                    ),
                ):
                    raise FieldConstructionError(
                        basis_shape,
                        point_shape,
                        cshape_orig,
                        shape_order,
                        hint="basis_shape cannot be broadcasted in the center",
                    )
                stack_start = len(basis_shape) + len(point_shape)
            else:
                stack_start = len(point_shape)
        elif shape_order[ShapeComponent.POINT] == 2:
            if not _is_broadcastable(
                point_shape, cshape_slice_negatives(-len(point_shape), 0)
            ):
                raise FieldConstructionError(
                    basis_shape,
                    point_shape,
                    cshape_orig,
                    shape_order,
                    hint="point_shape cannot be broadcasted at the end",
                )
            # if basis_shape was in center, we now have the right offset for it
            if shape_order[ShapeComponent.BASIS] == 1:
                if not _is_broadcastable(
                    basis_shape,
                    cshape_slice_negatives(
                        -(len(basis_shape) + len(point_shape)), -len(point_shape)
                    ),
                ):
                    raise FieldConstructionError(
                        basis_shape,
                        point_shape,
                        cshape_orig,
                        shape_order,
                        hint="basis_shape cannot be broadcasted in the center",
                    )
                stack_end -= len(basis_shape) + len(point_shape)
            else:
                stack_end -= len(point_shape)
        elif shape_order[ShapeComponent.POINT] == 1:
            # cases by basis_location
            if shape_order[ShapeComponent.BASIS] == 0:
                if not _is_broadcastable(
                    point_shape,
                    cshape_slice_positives(
                        len(basis_shape), (len(basis_shape) + len(point_shape))
                    ),
                ):
                    raise FieldConstructionError(
                        basis_shape,
                        point_shape,
                        cshape_orig,
                        shape_order,
                        hint="point_shape cannot be broadcasted in the center",
                    )
                stack_start += len(point_shape)
            else:
                if not _is_broadcastable(
                    point_shape,
                    cshape_slice_negatives(
                        -(len(basis_shape) + len(point_shape)), -len(basis_shape)
                    ),
                ):
                    raise FieldConstructionError(
                        basis_shape,
                        point_shape,
                        cshape_orig,
                        shape_order,
                        hint="point_shape cannot be broadcasted in the center",
                    )
                stack_end -= len(point_shape)

        stack_shape = cshape[stack_start:stack_end]
        shapes: list[tuple[int, ...]] = [()] * 3
        shapes[shape_order[ShapeComponent.BASIS]] = basis_shape
        shapes[shape_order[ShapeComponent.STACK]] = stack_shape
        shapes[shape_order[ShapeComponent.POINT]] = point_shape
        object.__setattr__(
            self,
            "coefficients",
            np_or_jnp(use_jax).broadcast_to(
                coefficients, shapes[0] + shapes[1] + shapes[2]
            ),
        )
        object.__setattr__(self, "basis_shape", basis_shape)
        object.__setattr__(self, "point_shape", point_shape)
        object.__setattr__(self, "stack_shape", stack_shape)
        object.__setattr__(self, "shape_order", shape_order)
        object.__setattr__(
            self, "shape_order_inverse", _invert_permutation(shape_order)
        )
        object.__setattr__(self, "use_jax", use_jax)

    @typing.overload
    def __getattr__(
        self, name: Literal["shape"]
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]: ...
    @typing.overload
    def __getattr__(self, name: Literal["basis"]) -> FieldBasisAccessor: ...
    @typing.overload
    def __getattr__(self, name: Literal["stack"]) -> FieldStackAccessor: ...
    @typing.overload
    def __getattr__(self, name: Literal["point"]) -> FieldPointAccessor: ...

    def __getattr__(self, name):
        if name == "shape":
            return (self.stack_shape, self.basis_shape, self.point_shape)
        if name == "basis":
            return FieldBasisAccessor(self)
        if name == "stack":
            return FieldStackAccessor(self)
        if name == "point":
            return FieldPointAccessor(self)
        raise AttributeError

    def __add__(self, other: Union["Field", float]):
        if isinstance(other, Field):
            a, b = Field.broadcast_fields_full(self, other)
            return Field(
                a.basis_shape,
                a.point_shape,
                a.coefficients + b.coefficients,
                shape_order=a.shape_order,
                use_jax=a.use_jax or b.use_jax,
            )
        if isinstance(other, float):
            return Field(
                self.basis_shape,
                self.point_shape,
                self.coefficients + other,
                shape_order=self.shape_order,
                use_jax=self.use_jax,
            )
        raise NotImplementedError

    def __radd__(self, other: Union["Field", float]):
        return self.__add__(other)

    def __sub__(self, other: Union["Field", float]):
        if isinstance(other, Field):
            a, b = Field.broadcast_fields_full(self, other)
            return Field(
                a.basis_shape,
                a.point_shape,
                a.coefficients - b.coefficients,
                shape_order=a.shape_order,
                use_jax=a.use_jax or b.use_jax,
            )
        if isinstance(other, float):
            return Field(
                self.basis_shape,
                self.point_shape,
                self.coefficients - other,
                shape_order=self.shape_order,
                use_jax=self.use_jax,
            )
        raise NotImplementedError

    def __rsub__(self, other: Union["Field", float]):
        if isinstance(other, Field):
            a, b = Field.broadcast_fields_full(self, other)
            return Field(
                a.basis_shape,
                a.point_shape,
                b.coefficients - a.coefficients,
                shape_order=a.shape_order,
                use_jax=a.use_jax or b.use_jax,
            )
        if isinstance(other, float):
            return Field(
                self.basis_shape,
                self.point_shape,
                other - self.coefficients,
                shape_order=self.shape_order,
                use_jax=self.use_jax,
            )
        raise NotImplementedError

    def __mul__(self, other: Union["Field", float]):
        if isinstance(other, Field):
            a, b = Field.broadcast_fields_full(self, other)
            return Field(
                a.basis_shape,
                a.point_shape,
                a.coefficients * b.coefficients,
                shape_order=a.shape_order,
                use_jax=a.use_jax or b.use_jax,
            )
        if isinstance(other, float):
            return Field(
                self.basis_shape,
                self.point_shape,
                self.coefficients * other,
                shape_order=self.shape_order,
                use_jax=self.use_jax,
            )
        raise NotImplementedError

    def __rmul__(self, other: Union["Field", float]):
        return self.__mul__(other)

    def __neg__(self):
        return Field(
            self.basis_shape,
            self.point_shape,
            -self.coefficients,
            shape_order=self.shape_order,
            use_jax=self.use_jax,
        )

    def get_shape(self, component: ShapeComponent) -> tuple[int, ...]:
        """Recovers the shape of the specified component. This shape is in the same
        format as numpy.shape, that is a tuple.

        Args:
            component (ShapeComponent): The component to sample

        Returns:
            tuple[int,...]: The shape of the specified component.
        """
        return (
            self.basis_shape
            if component == ShapeComponent.BASIS
            else (
                self.stack_shape
                if component == ShapeComponent.STACK
                else self.point_shape
            )
        )

    def _component_offset(self, component: ShapeComponent) -> int:
        """Recovers the axis index of the start of a component.

        Args:
            component (ShapeComponent): The component to sample.

        Returns:
            int: the index of the first axis of the component
        """
        ind = 0
        prec = self.shape_order[component]
        if self.shape_order[ShapeComponent.BASIS] < prec:
            ind += len(self.basis_shape)
        if self.shape_order[ShapeComponent.STACK] < prec:
            ind += len(self.stack_shape)
        if self.shape_order[ShapeComponent.POINT] < prec:
            ind += len(self.point_shape)
        return ind

    def broadcast_to_shape(
        self,
        stack_shape: tuple[int, ...] | None,
        basis_shape: tuple[int, ...] | None,
        point_shape: tuple[int, ...] | None,
        complete_broadcast=True,
    ) -> "Field":
        """This function is related to the numpy broadcast_to function.
        https://numpy.org/doc/stable/reference/generated/numpy.broadcast_to.html#numpy.broadcast_to

        The shape of the desired array is given as separate tuples for each component.
        Instead of the `subok` optional argument, the returned field will always have
        the same coefficient array type. Additionally, the value of `use_jax` is inherited.

        Args:
            stack_shape (tuple[int, ...] | None): The shape for the stack shape to be
                broadcasted to, or None, if the shape should be kept as-is.
            basis_shape (tuple[int, ...] | None): The shape for the basis shape to be
                broadcasted to, or None, if the shape should be kept as-is.
            point_shape (tuple[int, ...] | None): The shape for the field shape to be
                broadcasted to, or None, if the shape should be kept as-is.
            complete_broadcast (bool, optional): If False, only dimensions of size 1
                are added. When true, the shape of the field precisely matches.

        Raises:
            FieldShapeError: if the broadcast cannot be done by standard numpy
                broadcasting rules in each component. Note that broadcasting the basis
                shape is permitted, beyond standard compatibility rules.

        Returns:
            Field: The broadcasted field.
        """
        if (
            (
                basis_shape is not None
                and not _is_broadcastable(basis_shape, self.basis_shape)
            )
            or (
                stack_shape is not None
                and not _is_broadcastable(stack_shape, self.stack_shape)
            )
            or (
                point_shape is not None
                and not _is_broadcastable(point_shape, self.point_shape)
            )
        ):
            message = (
                f"Cannot broadcast field of shape {self.shape} into"
                f" shape {(stack_shape, basis_shape, point_shape)}"
            )
            raise FieldShapeError(message)
        slices: list[typing.Any] = [None, None, None]
        shapes: list[typing.Any] = [None, None, None]
        slices[self.shape_order[ShapeComponent.BASIS]] = (
            itertools.chain(
                (np.newaxis for _ in range(len(basis_shape) - len(self.basis_shape))),
                (slice(None) for _ in range(len(self.basis_shape))),
            )
            if basis_shape is not None
            else (slice(None) for _ in range(len(self.basis_shape)))
        )
        slices[self.shape_order[ShapeComponent.STACK]] = (
            itertools.chain(
                (np.newaxis for _ in range(len(stack_shape) - len(self.stack_shape))),
                (slice(None) for _ in range(len(self.stack_shape))),
            )
            if stack_shape is not None
            else (slice(None) for _ in range(len(self.stack_shape)))
        )
        slices[self.shape_order[ShapeComponent.POINT]] = (
            itertools.chain(
                (np.newaxis for _ in range(len(point_shape) - len(self.point_shape))),
                (slice(None) for _ in range(len(self.point_shape))),
            )
            if point_shape is not None
            else (slice(None) for _ in range(len(self.point_shape)))
        )
        shapes[self.shape_order[ShapeComponent.BASIS]] = (
            self.basis_shape if basis_shape is None else basis_shape
        )
        shapes[self.shape_order[ShapeComponent.STACK]] = (
            self.stack_shape if stack_shape is None else stack_shape
        )
        shapes[self.shape_order[ShapeComponent.POINT]] = (
            self.point_shape if point_shape is None else point_shape
        )
        coefs = self.coefficients[*itertools.chain(*slices)]
        return Field(
            self.basis_shape if basis_shape is None else basis_shape,
            self.point_shape if point_shape is None else point_shape,
            (
                np_or_jnp(self).broadcast_to(coefs, shapes[0] + shapes[1] + shapes[2])
                if complete_broadcast
                else coefs
            ),
            shape_order=self.shape_order,
            use_jax=self.use_jax,
        )

    @staticmethod
    def are_broadcastable(*fields: "Field", strict_basis=True) -> bool:  # NOQA: ARG004
        """Two fields a and b are (fully) broadcastable if they are compatible and have
        broadcastable point shape. Since this relation is associative,
        more than two fields can be passed in.

        Args:
            fields (tuple[Field, ...]): The fields to broadcast.
            strict_basis (bool, optional): If true, the basis rule holds. Otherwise,
                only basis shape numpy-broadcastibility is checked.

        Returns:
            bool: True if the fields are broadcastable. False otherwise.
        """
        return Field.are_compatible(*fields) and _is_compatible(
            *(field.point_shape for field in fields)
        )

    @typing.overload
    @staticmethod
    def broadcast_fields_full(
        *fields: "Field", strict_basis=True, shapes_only: Literal[True]
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]: ...
    @typing.overload
    @staticmethod
    def broadcast_fields_full(
        *fields: "Field", strict_basis=True, shapes_only: Literal[False] = False
    ) -> tuple["Field", ...]: ...
    @staticmethod
    def broadcast_fields_full(
        *fields: "Field",
        strict_basis=True,  # NOQA: ARG004
        shapes_only: bool = False,
    ) -> tuple["Field", ...] | tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        """Two fields a and b are (fully) broadcastable if they are compatible and have
        broadcastable point shape. Since this relation is associative,
        more than two fields can be passed in.

        Args:
            fields (tuple[Field, ...]): The fields to broadcast.
            strict_basis (bool, optional): If true, the basis rule holds. Otherwise,
                only basis shape numpy-broadcastibility is checked.
            shapes_only (bool, optional): If true, the target shape is returned and no
                array brouadcasting occurs.
        Raises:
            FieldShapeError: If the fields are not fully broadcastable together.

        Returns:
            tuple[Field, ...]: The broadcasted fields, in the order they were given.
        """
        if Field.are_broadcastable(*fields):
            basis_shape = np.broadcast_shapes(*[field.basis_shape for field in fields])
            stack_shape = np.broadcast_shapes(*[field.stack_shape for field in fields])
            point_shape = np.broadcast_shapes(*[field.point_shape for field in fields])
            if shapes_only:
                return (stack_shape, basis_shape, point_shape)
            return tuple(
                field.broadcast_to_shape(stack_shape, basis_shape, point_shape)
                for field in fields
            )

        message = "Cannot broadcast fields with incompatible shapes."
        raise FieldShapeError(message)

    @staticmethod
    def are_compatible(*fields: "Field", strict_basis=True) -> bool:
        """Two fields a and b are compatible if they have compatible bases
        (basis_shape equal or at least one of them is size 1 representing a constant)
        and they have broadcastable stack_shapes. This function checks them. Since
        this relation is associative, more than two fields can be passed in.

        Args:
            fields (tuple[Field, ...]): The fields to query compatibility.
            strict_basis (bool, optional): If true, the basis rule holds. Otherwise,
                only basis shape numpy-broadcastibility is checked.

        Returns:
            bool: True if the fields are compatible. False otherwise.
        """
        if strict_basis:
            return all(
                map(
                    lambda x: x[1],  # accumulator -> did nonempty tuple change?
                    itertools.accumulate(
                        (field.basis_shape for field in fields),
                        func=lambda a, b: (
                            a[0] if np.prod(b, dtype=int) == 1 else b,  # nonempty tuple
                            (np.prod(a[0], dtype=int) == 1)
                            or (np.prod(b, dtype=int) == 1)
                            or a[0] == b,  # if nonempty, did shape change?
                        ),
                        initial=((), True),
                    ),
                )
            ) and _is_compatible(*(field.stack_shape for field in fields))
        return _is_compatible(
            *(field.basis_shape for field in fields)
        ) and _is_compatible(*(field.stack_shape for field in fields))

    @staticmethod
    def broadcast_field_compatibility(
        *fields: "Field", strict_basis=True
    ) -> tuple["Field", ...]:
        """Two fields a and b are compatible if they have compatible bases
        (basis_shape equal or at least one of them is size 1 representing a constant)
        and they have broadcastable stack_shapes. Since
        this relation is associative, more than two fields can be passed in.

        This function broadcasts the fields to have the same stack and basis shapes if
        they are compatible, or raises an error if they are not.

        Args:
            fields (tuple[Field, ...]): The fields to broadcast.
            strict_basis (bool, optional): If true, the basis rule holds. Otherwise,
                only basis shape numpy-broadcastibility is checked.

        Raises:
            FieldShapeError: if the given fields are not all compatible.

        Returns:
            tuple[Field, ...]: The broadcasted fields, in the order they were given.
        """
        if Field.are_compatible(*fields, strict_basis=strict_basis):
            basis_shape = np.broadcast_shapes(*[field.basis_shape for field in fields])
            stack_shape = np.broadcast_shapes(*[field.stack_shape for field in fields])
            return tuple(
                field.broadcast_to_shape(stack_shape, basis_shape, field.point_shape)
                for field in fields
            )

        message = "Cannot broadcast fields with incompatible shapes."
        raise FieldShapeError(message)

    def _axis_field_to_numpy(
        self, index: FieldAxisIndexType, out_of_bounds_check: bool = True
    ) -> int:
        """Recovers the axis index (in terms of numpy) from the given field axis specifier.

        Args:
            index (FieldAxisIndexType): The field index to recover the axis of.
            out_of_bounds_check (bool, optional): Whether or not an out-of-bounds check
            occurs. As in python indexing, it is expected for
            -len(component_shape) <= index.index < len(component_shape), where
            component_shape is the shape of index.component. If out-of-bounds check is
            true, then an IndexError is thrown. Defaults to True.

        Raises:
            IndexError: if out_of_bounds_check is True, and the index is out of bounds.

        Returns:
            _type_: the integer index, in the numpy context.
        """
        comp: ShapeComponent = index[0]
        ind: int = index[1]
        shape = self.get_shape(comp)

        if ind < 0:
            if out_of_bounds_check and ind < -len(shape):
                message = (
                    f"Attempting to access axis {ind} ({-1 - ind}) of shape {shape}."
                )
                raise IndexError(message)
            ind = len(shape) + ind
        elif out_of_bounds_check and ind >= len(shape):
            message = f"Attempting to access axis {ind} of shape {shape}."
            raise IndexError(message)

        return ind + self._component_offset(comp)

    def __eq__(self, other) -> bool:
        if not Field.are_broadcastable(self, other):
            return False

        return np_or_jnp(self, other).array_equiv(
            *(f.coefficients for f in Field.broadcast_fields_full(self, other))
        )
