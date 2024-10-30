import numpy as np
import fastfem.elements.element as element


from numpy.typing import ArrayLike, NDArray
import typing

import numpy.typing
import collections.abc as colltypes

from fastfem.fields.field import Field


class DeformationGradient2DBadnessException(Exception):
    """An exception called when the deformation gradient is too poor for invertibility.

    Args:
        val (float): a badness parameter det(F) / (`char_x` * `char_y`) for
            characteristic lengths `char_x` and `char_y`. This may be in reference
            to the size of the element, so that scaling the element yields the same
            badness parameter.
        x (float): the x position in local (reference) coordinates of the determinant.
        y (float): the y position in local (reference) coordinates of the determinant.
    """

    def __init__(self, val: float, x: float, y: float):
        super().__init__(
            "Element has too poor of a shape!\n"
            + f"   def_grad badness = {val:e} at local coordinates"
            + f"({x:.6f},{y:.6f})"
        )

        self.x = x
        self.y = y
        self.val = val


def _build_GLL(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Builds the items necessary for GLL quadrature of degree `n` (`n+1`-node).

    Args:
        n (int): The degree of the quadrature. This results in `n+1` nodes for the
            quadrature.


    Returns:
        tuple[np.ndarray,np.ndarray,np.ndarray]: A tuple `(x,w,L)` for knots
            (node positions) `x`, weights `w`, and Lagrange interpolation polynomials
            `L` associated with the quadrature. A quadrature for a vectorized function f
            is computed as `sum(f(x) * w)`, while the $i^{th}$ Lagrange polynomial is
            $L_i(x) = \\sum_{k=0}^{n} L[i,k] x^k$.
    """
    if n == 1:
        return np.array((-1, 1)), np.array((1, 1)), np.array([[0.5, -0.5], [0.5, 0.5]])

    leg_x = (
        np.polynomial.Polynomial(np.polynomial.legendre.leg2poly([0] * n + [1]))
        .deriv()
        .roots()
    )
    np1 = n + 1
    x = np.array([-1, *leg_x, 1])

    # initialize L to 1; operate in quad precision at derivation
    L = np.zeros((np1, np1), dtype=np.longdouble)
    L[:, 0] = 1

    tmp = np.empty((n, n), dtype=np.longdouble)
    for i, xi in enumerate(x):
        mask = np.arange(np1) != i

        # what we multiply (x-xi) to
        tmp[:, :] = L[mask, :-1]
        tmp[:, :] /= x[mask, np.newaxis] - xi

        # add tmp*(-xi)
        L[mask, :-1] = tmp * -xi
        # add tmp*x
        L[mask, 1:] += tmp

    # return to double precision
    L = L.astype(np.float64)

    # compute weights by integrating L
    endpoint_pows = (
        np.array([-1, 1])[:, np.newaxis] ** np.arange(1, np1 + 1)[np.newaxis, :]
    )
    endpoints_integ = np.einsum(
        "ik,k,jk->ij", L, 1 / np.arange(1, np1 + 1), endpoint_pows
    )

    return x, endpoints_integ[:, 1] - endpoints_integ[:, 0], L


class SpectralElement2D(element.Element2D):
    """A spectral element in 2 dimensions of order N, leading to (N+1)^2 nodes.
    GLL quadrature is used to diagonalize the mass matrix.
    """

    def __init__(self, degree: int):
        super().__init__()
        self.degree = degree  # poly degree (#nodes - 1)
        self.num_nodes = degree + 1

        self.knots, self.weights, self._lagrange_polys = _build_GLL(degree)
        # store derivatives of L_i as needed, so they need only be computed once
        self._lagrange_derivs = dict()
        self._lagrange_derivs[0] = self._lagrange_polys

        self._lagrange_at_knots = dict()

    @typing.override
    def basis_shape(self) -> tuple[int, ...]:
        """Returns a tuple representing the shape of the array corresponding to the
        basis coefficients. A scalar field `f`, given as an array is expected to have
        shape `f.shape == element.basis_shape()`
        """
        return (self.num_nodes, self.num_nodes)

    @typing.override
    def reference_element_position_matrix(self) -> Field:
        """
        The position field of the reference (un-transformed) element. This is a vector
        field.

        Returns:
            Field: An array of shape `(*element.basis_shape(), 2)`
        """
        out = np.empty((self.num_nodes, self.num_nodes, 2))
        out[:, :, 0] = self.knots[:, np.newaxis]
        out[:, :, 1] = self.knots[np.newaxis, :]
        return self.Field(out, False, (2,))

    def lagrange_poly1D(self, deriv_order: int = 0) -> NDArray:
        """
        Returns the polynomial coefficients `P[i,k]`, where
        $\\frac{d^{r}}{dx^{r}} L_{i}(x) = \\sum_k P[i,k] x^k$ with $r$ as `deriv_order`.

        deriv_order (default 0)

        Args:
            deriv_order (int, optional): The order $r$ of the derivative. This is
            expected to be an integer between 0 (inclusive) and degree+1 (exclusive),
            but this check is not done. Defaults to 0.

        Returns:
            np.ndarray: The coefficient array `P[i,k]` which is of shape
            `(degree + 1, degree + 1 - deriv_order)`
        """
        if deriv_order in self._lagrange_derivs:
            return self._lagrange_derivs[deriv_order]

        # inefficient partial partition calc, but only done once, so...
        coefs = self._lagrange_polys
        for i in range(deriv_order):
            coefs = coefs[:, 1:] * np.arange(1, coefs.shape[1])

        self._lagrange_derivs[deriv_order] = coefs
        return coefs

    @typing.override
    def interpolate_field(
        self,
        field: Field,
        X: NDArray,
        Y: NDArray,
    ) -> typing.Any:
        """Evaluates field at (X,Y) in reference coordinates.
        The result is an array of values `field(X,Y)`.

        `field` is of shape `(*basis_shape,...,*fieldshape)`, where the internal
        ellipses `...` broadcasts with X and Y
        using numpy's rules for broadcasting.

        X and Y must have compatible shape, broadcasting to pointshape.

        Args:
            field (Field): an array of shape (degree+1,degree+1,*fieldshape)
                representing the field to be interpolated.
            X (ArrayLike): x values (in reference coordinates).
            Y (ArrayLike): y values (in reference coordinates).
            fieldshape (tuple[int,...], optional): the shape of `field` at each point.
                Defaults to tuple() for a scalar field.

        Returns:
            typing.Any: The interpolated values `field(X,Y)`
        """
        field_pad = (np.newaxis,) * len(field.field_shape)
        X = X[..., *field_pad]
        Y = Y[..., *field_pad]
        # F^{i,j} L_{i,j}(X,Y)
        # lagrange_polys[i,k] : component c in term cx^k of poly i
        return Field(
            self.basis_shape(),
            field.field_shape,
            np.einsum(
                "ij...,ia,...a,jb,...b->...",
                field.coefficients,
                self._lagrange_polys,
                np.expand_dims(X, -1) ** np.arange(self.num_nodes),
                self._lagrange_polys,
                np.expand_dims(Y, -1) ** np.arange(self.num_nodes),
            ),
        )

    # @warnings.deprecated("This can be done (probably cleaner) with JAX.")
    def locate_point(
        self,
        pos_matrix: np.ndarray,
        posx: float,
        posy: float,
        tol: float = 1e-8,
        dmin: float = 1e-7,
        max_iters: int = 1000,
        def_grad_badness_tol: float = 1e-4,
        ignore_out_of_bounds: bool = False,
        char_x: float | None = None,
        char_y: float | None = None,
    ) -> tuple[np.ndarray, bool]:
        """
        Attempts to find the local coordinates corresponding to the
        given global coordinates (posx,posy). Returns (local_pt,success).
        If a point is found, the returned value is ((x,y),True),
        with local coordinates (x,y).
        Otherwise, there are two cases:
          - ((x,y),False) is returned when descent leads out of the
            domain. A step is forced to stay within the local domain
            (max(|x|,|y|) <= 1), but if a constrained minimum is found on an
            edge with a descent direction pointing outwards, that point
            is returned. If ignore_out_of_bounds is true, the interior checking
            does not occur.
          - ((x,y),False) is returned if a local minimum is found inside the
            domain, but the local->global transformation is too far. This
            should technically not occur if the the deformation gradient stays
            nonsingular, but in the case of ignore_out_of_bounds == True, the
            everywhere-invertibility may not hold.

        The initial guess is chosen as the closest node, and a Newton-Raphson
        step is used along-side a descent algorithm. tol is the stopping parameter
        triggering when the error function (ex^2 + ey^2)/2 < tol in global
        coordinates. dmin is the threshold for when a directional derivative
        is considered zero, providing the second stopping criterion.

        def_grad_badness_tol parameterizes how poorly shaped the element is locally.
        If the coordinate vectors line up close enough,
        or one of the vectors gets too small,
        we can catch that with the expression
        (abs(det(def_grad)) < def_grad_badness_tol*char_x*char_y ),
        and raise an exception.
        Here, char_x and char_y are characteristic lengths of the element,
        and are calculated from pos_matrix when not defined.

        Args:
            pos_matrix (np.ndarray): an array representing the positions of the
                    element nodes. This is of the shape `(*basis_shape,2)`.
            posx (float): the x coordinate in global coordinates to find.
            posy (float): the y coordinate in global coordinates to find.
            tol (float, optional): Tolerance of the error function. Defaults to 1e-8.
            dmin (float, optional): The largest size of the gradient for a point to be
                    considered a local minimum. If the descent direction r dotted with
                    the gradient of the error function is less than dmin, then the local
                    minimum condition is met (If the gradient points out of the domain
                    then the dot product may be zero). Defaults to 1e-7.
            max_iters (int, optional): The maximum number of iterations taken.
                    Terminates afterwards, returning ((x,y),False).
            def_grad_badness_tol (float, optional): The minimum allowable badness
            parameter, after which an error is raised. Defaults to 1e-4.
            ignore_out_of_bounds (bool, optional): Whether or not descent directions can
                    point outside of the domain. If False, then locate_point stays
                    within the element. Defaults to False.
            char_x (float | None, optional): characteristic x-length of the element.
                    When not set, a characteristic value is computed from pos_matrix,
                    set to approximately the largest length curve of the position field
                    along constant local y. Defaults to None.
            char_y (float | None, optional): characteristic y-length of the element.
                    When not set, a characteristic value is computed from pos_matrix,
                    set to approximately the largest length curve of the position field
                    along constant local x. Defaults to None.

        Raises:
            DeformationGradient2DBadnessException: if the element is poorly shaped.

        Returns:
            tuple[tuple[float,float],bool]: ((x,y),success), where success is true
            if the error function is less than tol.
        """

        Np1 = self.num_nodes

        if char_x is None:
            # along each x-line, add the distances between nodes
            char_x = np.min(
                np.sum(  # take min across y-values, of x-line sums
                    np.linalg.norm(
                        pos_matrix[1:, :, :] - pos_matrix[:-1, :, :], axis=-1
                    ),
                    axis=0,
                )
            )

        if char_y is None:
            char_y = np.min(
                np.sum(  # take min across x-values, of y-line sums
                    np.linalg.norm(
                        pos_matrix[:, 1:, :] - pos_matrix[:, :-1, :], axis=-1
                    ),
                    axis=1,
                )
            )

        target = np.array((posx, posy))
        node_errs = np.sum((pos_matrix - target) ** 2, axis=-1)
        mindex = np.unravel_index(np.argmin(node_errs), (Np1, Np1))

        # local coords of guess
        local = np.array((self.knots[mindex[0]], self.knots[mindex[1]]))

        # position poly
        # sum(x^{i,j} L_{i,j}) -> [dim,k,l] (coefficient of cx^ky^l for dim)
        x_poly = np.einsum(
            "ijd,ik,jl->dkl", pos_matrix, self._lagrange_polys, self._lagrange_polys
        )

        # local to global
        def l2g(local):
            return np.einsum(
                "dkl,k,l->d",
                x_poly,
                local[0] ** np.arange(Np1),
                local[1] ** np.arange(Np1),
            )

        e = l2g(local) - target
        F = 0.5 * np.sum(e**2)

        # linsearch on gradient descent
        def linsearch(r, step):
            ARMIJO = 0.25
            err = l2g(local + r * step) - target
            F_new = 0.5 * np.sum(err**2)
            while F_new > F + (ARMIJO * step) * drF:
                step *= 0.5
                err[:] = l2g(local + r * step) - target
                F_new = 0.5 * np.sum(err**2)
            return F_new, step

        def clamp_and_maxstep(r):
            # out of bounds check; biggest possible step is to the boundary;
            # note: uses local[]. Additionally r[] is pass-by reference since it
            # can be modified. This is a feature, since that is the "clamp" part

            step = 1
            if ignore_out_of_bounds:
                return step
            for dim in range(2):  # foreach dim
                if (r[dim] < 0 and local[dim] == -1) or (
                    r[dim] > 0 and local[dim] == 1
                ):
                    # ensure descent direction does not point outside domain
                    # this allows constrained minimization
                    r[dim] = 0
                elif r[dim] != 0:
                    # prevent out-of-bounds step by setting maximum step;
                    if r[dim] > 0:
                        step = min(step, (1 - local[dim]) / r[dim])
                    else:
                        step = min(step, (-1 - local[dim]) / r[dim])
            return step

        iter_ = 0
        while F > tol and iter_ < max_iters:
            iter_ += 1

            # find descent direction by Newton

            # derivative of l2g
            dx_l2g = np.einsum(
                "dkl,k,l->d",
                x_poly[:, 1:, :],
                np.arange(1, Np1) * local[0] ** (np.arange(Np1 - 1)),
                local[1] ** np.arange(Np1),
            )
            dy_l2g = np.einsum(
                "dkl,k,l->d",
                x_poly[:, :, 1:],
                local[0] ** np.arange(Np1),
                np.arange(1, Np1) * local[1] ** (np.arange(Np1 - 1)),
            )

            # check badness
            defgrad_badness = np.abs(
                np.linalg.det([dx_l2g, dy_l2g]) / (char_x * char_y)  # type: ignore
            )
            if defgrad_badness < def_grad_badness_tol:
                raise DeformationGradient2DBadnessException(
                    defgrad_badness, local[0], local[1]
                )

            # grad of err function (ex^2 + ey^2)/2
            dxF = np.dot(e, dx_l2g)
            dyF = np.dot(e, dy_l2g)

            # solve hessian and descent dir
            hxx = np.sum(
                dx_l2g * dx_l2g
                + e
                * np.einsum(
                    "dkl,k,l->d",
                    x_poly[:, 2:, :],
                    np.arange(1, Np1 - 1)
                    * np.arange(2, Np1)
                    * local[0] ** (np.arange(Np1 - 2)),
                    local[1] ** (np.arange(Np1)),
                )
            )
            hxy = np.sum(
                dx_l2g * dy_l2g
                + e
                * np.einsum(
                    "dkl,k,l->d",
                    x_poly[:, 1:, 1:],
                    np.arange(1, Np1) * local[0] ** (np.arange(Np1 - 1)),
                    np.arange(1, Np1) * local[1] ** (np.arange(Np1 - 1)),
                )
            )
            hyy = np.sum(
                dy_l2g * dy_l2g
                + e
                * np.einsum(
                    "dkl,k,l->d",
                    x_poly[:, :, 2:],
                    local[0] ** (np.arange(Np1)),
                    np.arange(1, Np1 - 1)
                    * np.arange(2, Np1)
                    * local[1] ** (np.arange(Np1 - 2)),
                )
            )

            # target newton_rhaphson step
            r_nr = -np.linalg.solve([[hxx, hxy], [hxy, hyy]], [dxF, dyF])

            # target grad desc step
            r_gd = -np.array((dxF, dyF))
            r_gd /= np.linalg.norm(r_gd)  # scale gd step

            # take the better step between Newton-Raphson and grad desc
            s_nr = clamp_and_maxstep(r_nr)
            s_gd = clamp_and_maxstep(r_gd)

            # descent direction -- if dF . r == 0, we found minimum
            drF = dxF * r_gd[0] + dyF * r_gd[1]
            if drF > -dmin:
                break
            F_gd, s_gd = linsearch(r_gd, s_gd)

            # compare to NR
            F_nr = 0.5 * np.sum((l2g(local + r_nr * s_nr) - target) ** 2)
            if F_nr < F_gd:
                local += r_nr * s_nr
                e[:] = l2g(local) - target
                F = F_nr
            else:
                local += r_gd * s_gd
                e[:] = l2g(local) - target
                F = F_gd

            # nudge back in bounds in case of truncation error
            if not ignore_out_of_bounds:
                for dim in range(2):
                    local[dim] = min(1, max(-1, local[dim]))
        return (local, F < tol)

    def lagrange_eval1D(
        self,
        deriv_order: int,
        lag_index: ArrayLike | None = None,
        x: ArrayLike | None = None,
    ) -> NDArray:
        """
        Calculates the derivative
        $[(\\frac{\\partial d}{dx})^{deriv_order} L_{lag_index}(x)]_{x}$

        Note that "lagrange" refers to the lagrange interpolation polynomial,
        not lagrangian coordinates. This is a one-dimension helper function.

        deriv_order is taken as an integer.

        lag_index and x must be broadcastable to the same shape, following
        standard numpy broadcasting rules.

        Since the polynomial coefficient matrix is indexed by lag_index,
        that is, P[lag_index,:] is stored, it is advised that lag_index should
        not have more than one element per index. In other words, lag_index
        should be some subset, reshaping, and/or permutation of arange().

        Args:
            deriv_order (int): the order of the derivative to compute
            lag_index (ArrayLike | None, optional): an array of indices for sampling
                the Lagrange polynomials. If None, then
                `np.arange(degree+1)[:,...]` is used.
                Defaults to None.
            x (ArrayLike | None, optional): an array of points to sample the Lagrange
                polynomials. If None, then `self.knots` is used. Defaults to None.

        Returns:
            NDArray: the result of the evaluation.
        """

        if x is None:
            if deriv_order not in self._lagrange_at_knots:
                self._lagrange_at_knots[deriv_order] = np.einsum(
                    "ik,jk->ij",
                    self.lagrange_poly1D(deriv_order),
                    self.knots[:, np.newaxis]
                    ** np.arange(self.num_nodes - deriv_order),
                )
            if lag_index is None:
                return self._lagrange_at_knots[deriv_order]
            # second index of arange should enforce the broadcastibility

            if not isinstance(lag_index, np.ndarray):
                lag_index = np.array(lag_index)
            return self._lagrange_at_knots[deriv_order][
                lag_index, np.arange(self.num_nodes)
            ]
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        if lag_index is None:
            lag_index = np.arange(self.num_nodes)[
                :, *(np.newaxis for _ in range(len(x.shape)))
            ]
        # out = sum_{k=deriv_order}^{degree}(
        #   k*(k-1)*...*(k-deriv_order+1) * L_poly[lag_index,k] * x^{k-deriv_order}
        #   )

        if not isinstance(lag_index, np.ndarray):
            lag_index = np.array(lag_index)
        return np.einsum(
            "...k,...k->...",
            self.lagrange_poly1D(deriv_order)[lag_index, :],
            np.expand_dims(x, -1) ** np.arange(self.num_nodes - deriv_order),
        )

    @typing.override
    def compute_field_gradient(
        self,
        field: Field,
        pos_matrix: Field | None = None,
    ) -> Field:
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

        x_deriv = np.einsum(
            "ij...,ia->aj...",
            field.coefficients,
            self.lagrange_eval1D(1),
        )
        y_deriv = np.einsum(
            "ij...,jb->ib...",
            field.coefficients,
            self.lagrange_eval1D(1),
        )
        grad = np.stack([x_deriv, y_deriv], -1)

        if pos_matrix is not None:
            # (*pointshape,*stackshape,2{i},2{j}): dX^i/dx_j
            # must pad after stackshape
            field_pad = (np.newaxis,) * len(field.field_shape)
            def_grad = self.compute_field_gradient(pos_matrix)
            # grad is (*pointshape,*fieldshape,j): dF/dx_j
            # so we need the inverse
            return self.Field(
                np.einsum(
                    "...ji,...j->...i",
                    np.linalg.inv(def_grad.coefficients)[..., *field_pad, :, :],
                    grad,
                ),
                False,
                field.field_shape + (2,),
            )

        return self.Field(grad, False, field.field_shape + (2,))

    @typing.override
    def interpolate_field_gradient(
        self,
        field: Field,
        X: NDArray,
        Y: NDArray,
        pos_matrix: Field | None = None,
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
        field, pos_matrix = Field.broadcast_field_compatibility(field, pos_matrix)
        field_pad = (np.newaxis,) * len(field.field_shape)
        X = X[..., *field_pad]
        Y = Y[..., *field_pad]

        x_deriv = np.einsum(
            "ij...,i...,j...->...",
            field.coefficients,
            self.lagrange_eval1D(1, x=X),
            self.lagrange_eval1D(0, x=Y),
        )
        y_deriv = np.einsum(
            "ij...,i...,j...->...",
            field.coefficients,
            self.lagrange_eval1D(0, x=X),
            self.lagrange_eval1D(1, x=Y),
        )
        grad = np.stack([x_deriv, y_deriv], -1)

        if pos_matrix is not None:
            # (*pointshape,*stackshape*,2{i},2{j}): dX^i/dx_j
            def_grad = self.interpolate_field_gradient(pos_matrix, X, Y)
            # grad is (*pointshape,*fieldshape,j): dF/dx_j
            # so we need the inverse
            return np.einsum("...ji,...j->...i", np.linalg.inv(def_grad), grad)

        return grad

    @typing.override
    def integrate_field(
        self,
        pos_matrix: Field,
        field: Field,
        jacobian_scale: Field = Field(tuple(), tuple(), 1),
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
        pos_matrix, field, jacobian_scale = Field.broadcast_field_compatibility(
            pos_matrix, field, jacobian_scale
        )
        field_pad = (np.newaxis,) * len(field.field_shape)
        Jw = np.einsum(
            "i,j,ij...,ij...->ij...",
            self.weights,
            self.weights,
            np.abs(np.linalg.det(self.compute_field_gradient(pos_matrix).coefficients)),
            jacobian_scale.coefficients,
        )[..., *field_pad]
        return np.einsum("ij...,ij...->...", Jw, field.coefficients)

    @typing.override
    def integrate_basis_times_field(
        self,
        pos_matrix: Field,
        field: Field,
        indices: colltypes.Sequence[ArrayLike] | None = None,
        jacobian_scale: Field = Field(tuple(), tuple(), None),
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
        pos_matrix, field, jacobian_scale = Field.broadcast_field_compatibility(
            pos_matrix, field, jacobian_scale
        )
        field_pad = (np.newaxis,) * len(field.field_shape)
        Jw = np.einsum(
            "i,j,ij...,ij...->ij...",
            self.weights,
            self.weights,
            np.abs(np.linalg.det(self.compute_field_gradient(pos_matrix).coefficients)),
            jacobian_scale.coefficients,
        )[..., *field_pad]
        if indices is None:
            return np.einsum("ij...,ij...->ij...", Jw, field.coefficients)
        return np.einsum("ij...,ij...->ij...", Jw, field.coefficients)[*indices, ...]

    @typing.override
    def mass_matrix(
        self,
        pos_matrix: Field,
        indices: colltypes.Sequence[np.ndarray] | None = None,
        jacobian_scale: Field = Field(tuple(), tuple(), None),
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
        # compute using GLL quadrature

        # int (phi1 * phi2) = sum_i(phi1(x_i) phi2(x_i) w_i * J(x_i))

        # weights [i,j] times jacobian

        pos_matrix, jacobian_scale = Field.broadcast_field_compatibility(
            pos_matrix, jacobian_scale
        )
        Jw = np.einsum(
            "i,j,ij...,ij...->ij...",
            self.weights,
            self.weights,
            np.abs(np.linalg.det(self.compute_field_gradient(pos_matrix).coefficients)),
            jacobian_scale.coefficients,
        )
        aux_pad = tuple(np.newaxis for _ in range(len(Jw.shape) - 2))
        basis_dims = len(self.basis_shape())
        if indices is None:
            return (
                self.basis_fields() * Jw[*(np.newaxis for _ in range(basis_dims)), ...]
            )
        indsI = np.array(indices[:basis_dims], dtype=int)
        indsJ = np.array(indices[basis_dims:], dtype=int)
        Jw = Jw[*indsJ]
        return np.where(
            np.logical_and(
                indsI[0, ...] == indsJ[0, ...], indsI[1, ...] == indsJ[1, ...]
            )[..., *aux_pad],
            Jw,
            0,
        )

    @typing.override
    def integrate_grad_basis_dot_field(
        self,
        pos_matrix: Field,
        field: Field,
        is_field_upper_index: bool,
        indices: colltypes.Sequence[ArrayLike] | None = None,
        jacobian_scale: Field = Field(tuple(), tuple(), 1),
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
        pos_matrix, field, jacobian_scale = Field.broadcast_field_compatibility(
            pos_matrix, field, jacobian_scale
        )
        field_pad = (np.newaxis,) * len(field.field_shape)
        def_grad = self.compute_field_gradient(pos_matrix).coefficients
        # int (grad phi1 . grad field)
        w = np.einsum(
            "i,j,ij...,ij...->ij...",
            self.weights,
            self.weights,
            np.abs(np.linalg.det(def_grad)),
            jacobian_scale.coefficients,
        )[..., *field_pad]
        # [i,k] L_k'(x_i)
        lag_div = self.lagrange_eval1D(1)
        # [i,j,...,dim] partial_dim field(xi,xj)

        if not is_field_upper_index:
            def_grad_inv = np.linalg.inv(def_grad)[..., *field_pad, :, :]
            field = np.einsum(
                "ij...ab,ij...db,ij...d->ij...a",
                def_grad_inv,
                def_grad_inv,
                field.coefficients,
            )

        # integrand is
        # [ field(xi,xj) ] * [ partial_dim (L_m(xi)L_n(xj)) ]
        # where partial_dim (L_m(xi)L_n(xj)) = {dim==0: L_m'(xi)L_n(xj), dim==1: L_m(xi)L_n'(xj)}
        #                  = delta_{dim,0} delta_{nj} L_m'(xi) + delta_{dim,1} delta_{mi} L_n'(xj)

        # full integral is
        # [ field(xi,xj) ] * [ partial_dim (L_m(xi)L_n(xj)) ] * w_{ij}
        # = field(xi,xj) (delta_{dim,0} delta_{nj} L_m'(xi) + delta_{dim,1} delta_{mi} L_n'(xj)) w_{ij}
        # = (field(xi,xj)) delta_{dim,0} delta_{nj} L_m'(xi)w_{ij}
        #           + (field(xi,xj)) delta_{dim,1} delta_{mi} L_n'(xj) w_{ij}
        # = (field_0(xi,xn)) L_m'(xi)w_{in} + (field_1(xm,xj)) L_n'(xj) w_{mj}

        KF = np.empty(np.broadcast_shapes(pos_matrix.shape, field.shape))  # type: ignore
        KF[:, :] = np.einsum(
            "in...,im,in->mn...", field[..., 0], lag_div, w  # type: ignore
        )  # type: ignore
        KF[:, :] += np.einsum(
            "mj...,jn,mj->mn...", field[..., 1], lag_div, w  # type: ignore
        )  # type: ignore
        if indices is None:
            return KF
        return KF[*indices, ...]  # type: ignore

    @typing.override
    def integrate_grad_basis_dot_grad_field(
        self,
        pos_matrix: Field,
        field: Field,
        indices: colltypes.Sequence[ArrayLike] | None = None,
        jacobian_scale: Field = Field(tuple(),tuple(),1),
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
        # compute using GLL quadrature
        pos_matrix, field, jacobian_scale = Field.broadcast_field_compatibility(
            pos_matrix, field, jacobian_scale
        )

        def_grad = self.compute_field_gradient(pos_matrix).coefficients
        # int (grad phi1 . grad field)
        w = np.einsum(
            "i,j,ij...,ij...->ij...",
            self.weights,
            self.weights,
            np.abs(np.linalg.det(def_grad)),
            jacobian_scale.coefficients,
        )
        # [i,k] L_k'(x_i)
        lag_div = self.lagrange_eval1D(1)
        # [i,j,...,dim] partial_dim field(xi,xj)

        grad_field_form = self.compute_field_gradient(field).coefficients
        def_grad_inv = np.linalg.inv(def_grad)
        grad_field = np.einsum(
            "ij...ab,ij...db,ij...d->ij...a",
            def_grad_inv,
            def_grad_inv,
            grad_field_form,
        )

        # integrand is
        # [ partial_dim field(xi,xj) ] * [ partial_dim (L_m(xi)L_n(xj)) ]
        # where partial_dim (L_m(xi)L_n(xj)) = {dim==0: L_m'(xi)L_n(xj), dim==1: L_m(xi)L_n'(xj)}
        #                  = delta_{dim,0} delta_{nj} L_m'(xi) + delta_{dim,1} delta_{mi} L_n'(xj)

        # full integral is
        # [ partial_dim field(xi,xj) ] * [ partial_dim (L_m(xi)L_n(xj)) ] * w_{ij}
        # = partial_dim field(xi,xj) (delta_{dim,0} delta_{nj} L_m'(xi) + delta_{dim,1} delta_{mi} L_n'(xj)) w_{ij}
        # = (partial_dim field(xi,xj)) delta_{dim,0} delta_{nj} L_m'(xi)w_{ij}
        #           + (partial_dim field(xi,xj)) delta_{dim,1} delta_{mi} L_n'(xj) w_{ij}
        # = (partial_0 field(xi,xn)) L_m'(xi)w_{in} + (partial_1 field(xm,xj)) L_n'(xj) w_{mj}

        KF = np.empty(field.basis_shape + field.stack_shape + field.field_shape)
        KF[...] = np.einsum("in...,im,in...->mn...", grad_field[..., 0], lag_div, w)
        KF[...] += np.einsum("mj...,jn,mj...->mn...", grad_field[..., 1], lag_div, w)
        if indices is None:
            return KF
        return KF[*indices, ...]  # type: ignore

