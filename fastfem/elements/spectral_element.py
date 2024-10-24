import numpy as np
import fastfem.elements.element as element
import typing


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


class SpectralElement2D(element.Element):
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


    @typing.override
    def basis_shape(self) -> tuple[int,...]:
        """Returns a tuple representing the shape of the array corresponding to the
        basis coefficients. A scalar field `f`, given as an array is expected to have
        shape `f.shape == element.basis_shape()`
        """
        return (self.num_nodes,self.num_nodes)

    def lagrange_poly1D(self, deriv_order: int = 0) -> np.ndarray:
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

    def interp_field(
            self, field: np.ndarray, X: np.ndarray | float, Y: np.ndarray | float,
            fieldshape: tuple[int,...] = tuple()) -> np.ndarray:
        """Evaluates field at (X,Y) in reference coordinates.
        The result is an array of values `field(X,Y)`.

        `field` is of shape `(*basis_shape,...,*fieldshape)`, where the internal
        ellipses `...` broadcasts with X and Y
        using numpy's rules for broadcasting.

        X and Y must have compatible shape, broadcasting to pointshape.

        Args:
            field (np.ndarray): an array of shape (degree+1,degree+1,*fieldshape)
                representing the field to be interpolated.
            X (np.ndarray | float): x values (in reference coordinates).
            Y (np.ndarray | float): y values (in reference coordinates).
            fieldshape (tuple[int,...], optional): the shape of `field` at each point.
                Defaults to tuple() for a scalar field.

        Returns:
            np.ndarray: The interpolated values `field(X,Y)`
        """
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if not isinstance(Y, np.ndarray):
            Y = np.array(Y)
        field_pad = (np.newaxis,) * len(fieldshape)
        X = X[...,*field_pad]
        Y = Y[...,*field_pad]
        # F^{i,j} L_{i,j}(X,Y)
        # lagrange_polys[i,k] : component c in term cx^k of poly i
        return np.einsum(
            "ij...,ia,...a,jb,...b->...",
            field,
            self._lagrange_polys,
            np.expand_dims(X, -1) ** np.arange(self.num_nodes),
            self._lagrange_polys,
            np.expand_dims(Y, -1) ** np.arange(self.num_nodes),
        )

    @typing.override
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
            pos_matrix (np.ndarray): an array representing the positions of the
                    element nodes. This is of the shape `(*basis_shape,...,2)`.
            X (np.ndarray | float): X coordinates of the points to map. The
                    shape of X must be compatible with the shape of Y.
            Y (np.ndarray | float): Y coordinates of the points to map. The
                    shape of Y must be compatible with the shape of X.

        Returns:
            np.ndarray: An array of real coordinates that (X,Y) map to.
        """
        return self.interp_field(pos_matrix, X, Y,fieldshape=(2,))

    #@warnings.deprecated("This can be done (probably cleaner) with JAX.")
    def locate_point(
        self,
        pos_matrix: np.ndarray,
        posx: float,
        posy: float,
        tol: float = 1e-8,
        dmin: float = 1e-7,
        max_iters: int = 1000,
        def_grad_badness_tol: float =1e-4,
        ignore_out_of_bounds: bool =False,
        char_x: float|None = None,
        char_y: float|None = None) -> tuple[np.ndarray,bool]:
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

        while F > tol:
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
                np.linalg.det([dx_l2g, dy_l2g]) / (char_x * char_y) # type: ignore
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

    def lagrange_eval1D(self, deriv_order: int, lag_index, x):
        """Calculates the derivative
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
            lag_index (np.ndarray): an array of indices for sampling the Lagrange
                polynomials.
            x (np.ndarray): an array of points to sample the Lagrange polynomials.

        Returns:
            np.ndarray: the result of the evaluation.
        """
        # out = sum_{k=deriv_order}^{degree}(
        #   k*(k-1)*...*(k-deriv_order+1) * L_poly[lag_index,k] * x^{k-deriv_order}
        #   )

        return np.einsum(
            "...k,...k->...",
            self.lagrange_poly1D(deriv_order)[lag_index, :],
            np.expand_dims(x, -1) ** np.arange(self.num_nodes - deriv_order),
        )

    def field_grad(self, field: np.ndarray, X: np.ndarray | float,
            Y: np.ndarray | float,
            pos_matrix: np.ndarray | None = None,
            fieldshape: tuple[int,...] = tuple()) -> np.ndarray:
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
            field (np.ndarray): an array of shape (degree+1,degree+1,*fieldshape)
                representing the field to be interpolated.
            X (np.ndarray | float): x values (in reference coordinates).
            Y (np.ndarray | float): y values (in reference coordinates).
            pos_matrix (np.ndarray | None, optional): If set, `pos_matrix` specifies
                the position fields of the element, and the gradient will be computed in
                Cartesian coordinates. Defaults to None.
            fieldshape (tuple[int,...], optional): the shape of `field` pointwise.
                Defaults to tuple(), representing a scalar field.

        Returns:
            np.ndarray: an array representing the gradient of the field evaluated
                at each point.
        """

        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if not isinstance(Y, np.ndarray):
            Y = np.array(Y)
        field_pad = (np.newaxis,) * len(fieldshape)
        X = X[...,*field_pad]
        Y = Y[...,*field_pad]

        x_deriv = np.einsum(
            "ij...,ia,...a,jb,...b->...",
            field,
            self.lagrange_poly1D(1),
            np.expand_dims(X, -1) ** np.arange(self.num_nodes - 1),
            self.lagrange_poly1D(0),
            np.expand_dims(Y, -1) ** np.arange(self.num_nodes),
        )
        y_deriv = np.einsum(
            "ij...,ia,...a,jb,...b->...",
            field,
            self.lagrange_poly1D(0),
            np.expand_dims(X, -1) ** np.arange(self.num_nodes),
            self.lagrange_poly1D(1),
            np.expand_dims(Y, -1) ** np.arange(self.num_nodes - 1),
        )
        grad = np.stack([x_deriv, y_deriv], -1)

        if pos_matrix is not None:
            # (*pointshape,*fieldshape,i,j): dX^i/dx_j
            def_grad = self.def_grad(pos_matrix, X, Y)

            # grad is (*pointshape,*fieldshape,j): dF/dx_j
            # so we need the inverse
            return np.einsum("...ji,...j->...i", np.linalg.inv(def_grad), grad)

        return grad

    def def_grad(self, pos_matrix: np.ndarray, X: np.ndarray | float,
            Y: np.ndarray | float) -> np.ndarray:
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
            pos_matrix (np.ndarray): an array representing the positions of the
                    element nodes. This is of the shape `(*basis_shape,...,2)`.
            X (np.ndarray | float): X coordinates of the points to map. The
                    shape of X must be compatible with the shape of Y.
            Y (np.ndarray | float): Y coordinates of the points to map. The
                    shape of Y must be compatible with the shape of X.


        Returns:
            np.ndarray: the evaluated deformation gradient.
        """
        return self.field_grad(pos_matrix, X, Y, fieldshape=(2,))

    def _def_grad(self, pos_matrix, i, j):
        """Calculates the deformation gradient matrix dX/(dxi)
        at the reference coordinates xi = (x_i,y_j).
        i and j must be broadcastable to the same shape.
        The result is an array with shape (*X.shape,2,2) where
        the first new index specifies the coordinate in global space and the
        second index specifies the coordinate in reference space.
        """
        raise NotImplementedError(
            "Can delegate to def_grad using knot slicing. Do we need this?"
        )
        if not isinstance(i, np.ndarray):
            i = np.array(i)
        if not isinstance(j, np.ndarray):
            j = np.array(j)
        indims = max(i.ndim, j.ndim)
        i = np.expand_dims(i, tuple(range(i.ndim, indims)))
        j = np.expand_dims(j, tuple(range(j.ndim, indims)))

        grad = np.einsum(
            "abl,ak...,bk...->...lk",
            pos_matrix,
            self._lagrange_deriv(
                np.arange(self.degree + 1),  # a
                np.array([1, 0])[np.newaxis, :],  # (*,k)
                i[np.newaxis, np.newaxis],
            ),  # (*,*,indims)
            self._lagrange_deriv(
                np.arange(self.degree + 1),  # b
                np.array([0, 1])[np.newaxis, :],  # (*,k)
                j[np.newaxis, np.newaxis],
            ),  # (*,*,indims)
        )
        return grad

    def _lagrange_grads(self, pos_matrix, a, b, i, j, cartesian=False):
        """Writing phi_{a,b}(x_i,y_j) = l_a(x_i)l_b(y_j),
        calculates for arrays a,b,i,j:
            grad phi_{a,b}(x_i,y_j)
        where the last index specifies the dimension, and the rest
        match the shape of a,b,i, and j

        cartesian==True specifies that this gradient is partial_{x}.
        Otherwise, it is in Lagrangian coordinates, so partial_{xi}.
        """
        raise NotImplementedError(
            "Can delegate to field_grad using knot slicing. Do we need this?"
        )
        if not isinstance(a, np.ndarray):
            a = np.array(a)
        if not isinstance(b, np.ndarray):
            b = np.array(b)
        if not isinstance(i, np.ndarray):
            i = np.array(i)
        if not isinstance(j, np.ndarray):
            j = np.array(j)

        # TODO this is for sure broken

        dims = max(i.ndim, j.ndim, a.ndim, b.ndim) + 1
        i = np.expand_dims(i, tuple(range(i.ndim + 1, dims)))
        j = np.expand_dims(j, tuple(range(j.ndim + 1, dims)))
        a = np.expand_dims(a, tuple(range(a.ndim + 1, dims)))
        b = np.expand_dims(b, tuple(range(b.ndim + 1, dims)))
        # nabla_I L(...)
        lagrangian = self._lagrange_deriv(  # l_a^k(x)
            a, np.array([1, 0]), i
        ) * self._lagrange_deriv(  # l_b^k(y)
            b, np.array([0, 1]), j
        )
        if cartesian:
            # deformation gradient:
            # [dX/dxi1, dX/dxi2]
            # [dY/dxi1, dY/dxi2]
            grad = self.def_grad(pos_matrix, i, j)[:, :, 0]  # collapse the dim for k
            # we need to get d\xi/dx
            gradinv = np.linalg.inv(grad.T)
            # [dxi1/dX, dxi1/dY] T   [dxi1/dX, dxi2/dX]
            # [dxi2/dX, dxi2/dY]   = [dxi1/dY, dxi2/dY]

            # lagrangian is [dL/dxi1, dL/xi2]^T
            return (gradinv @ np.expand_dims(lagrangian.T, -1)).T[0]
        return lagrangian


    @typing.override
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
        # compute using GLL quadrature

        # int (phi1 * phi2) = sum_i(phi1(x_i) phi2(x_i) w_i * J(x_i))

        # weights [i,j] times jacobian
        w = (
            self.weights[:, np.newaxis]
            * self.weights[np.newaxis, :]
            * np.abs(
                np.linalg.det(
                    self.def_grad(
                        pos_matrix,
                        self.knots[:, np.newaxis],
                        self.knots[np.newaxis, :],
                    )
                )
            )
        )
        return w


    @typing.override
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
            pos_matrix (np.ndarray): _description_
            field (np.ndarray): _description_
            fieldshape (tuple[int,...], optional): The shape of field pointwise.
                    Defaults to tuple() for scalar fields.

        Returns:
            np.ndarray: an array representing the stiffness matrix
        """
        # compute using GLL quadrature

        # int (grad phi1 . grad field)
        w = (
            self.weights[:, np.newaxis]
            * self.weights[np.newaxis, :]
            * np.abs(
                np.linalg.det(
                    self.def_grad(
                        pos_matrix,
                        np.arange(self.num_nodes),
                        np.arange(self.num_nodes)[np.newaxis, :],
                    )
                )
            )
        )
        # [i,k] L_k'(x_i)
        lag_div = self.lagrange_eval1D(
            1, np.arange(self.num_nodes), self.knots[:, np.newaxis]
        )
        # [i,j,...,dim] partial_dim field(xi,xj)

        grad_field_form = self.field_grad(field, self.knots[:, np.newaxis], self.knots,
            fieldshape=fieldshape)
        def_grad_inv = np.linalg.inv(
            self.def_grad(pos_matrix, self.knots[:, np.newaxis], self.knots)
        )
        grad_field_vec = np.einsum(
            "ijab,ijdb,ij...d->ij...a", def_grad_inv, def_grad_inv, grad_field_form
        )
        grad_field = np.split(grad_field_vec, 2, axis=-1)

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

        KF = np.empty((self.num_nodes, self.num_nodes, *field.shape[2:]))
        KF[:, :] = np.einsum(
            "in...,im,in->mn...", grad_field[0].squeeze(-1), lag_div, w
        )
        KF[:, :] += np.einsum(
            "mj...,jn,mj->mn...", grad_field[1].squeeze(-1), lag_div, w
        )

        return KF

    #@warnings.deprecated("No support for vectorized elements; out of scope.")
    def basis_stiffness_matrix_diagonal(self, pos_matrix: np.ndarray) -> np.ndarray:
        """
        Computes the diagonal of the stiffness matrix
        $$\\int (\\nabla \\phi_i) \\cdot (\\nabla \\phi_i) ~ dV$$
        Args:
            pos_matrix (np.ndarray): an array representing the positions of the
                    element nodes. This is of the shape `(*basis_shape,2)`.


        Returns:
            np.ndarray: an array of the evaluated points of shape basis_shape.
        """
        w = (
            self.weights[:, np.newaxis]
            * self.weights[np.newaxis, :]
            * np.abs(
                np.linalg.det(
                    self.def_grad(
                        pos_matrix,
                        np.arange(self.num_nodes),
                        np.arange(self.num_nodes)[np.newaxis, :],
                    )
                )
            )
        )

        # [i,k] L_k'(x_i)
        lag_div = self.lagrange_eval1D(
            1, np.arange(self.num_nodes), self.knots[:, np.newaxis]
        )
        lag_div2 = lag_div**2

        def_grad_inv = np.linalg.inv(
            self.def_grad(pos_matrix, self.knots[:, np.newaxis], self.knots)
        )
        # to compute, join gradients with a,d and sum along i,j
        inner_prod = np.einsum("ijab,ijdb,ij->ijad", def_grad_inv, def_grad_inv, w)

        return (
            np.einsum("im,in->mn", lag_div2, inner_prod[:, :, 0, 0])
            + np.einsum("jn,mj->mn", lag_div2, inner_prod[:, :, 1, 1])
            + 2 * np.einsum("mm,nn,mn->mn", lag_div, lag_div, inner_prod[:, :, 0, 1])
        )

    def _bdry_normalderiv(self, pos_matrix, edge_index, field):
        """Computes the gradient of 'field' in the normal
        direction along the given edge. The returned value is an
        array of the gradient at points along the specified edge,
        in the direction of increasing coordinate.

        edge starts at 0 for the +x side, and increases as you
        go counterclockwise.
        """
        # build d/dn in the (+x)-direction
        # (partial_1 phi_a)(x = x_N) * phi_b(y = x_j) ; [a,b,j] expected,
        # but this is just [a] * delta_{b,j}, so we only need [a]:
        edge_inds = self._get_edge_inds(edge_index)
        pos_bdry = pos_matrix[edge_inds[:, 0], edge_inds[:, 1], :]
        def_grad = self.def_grad(pos_matrix, pos_bdry[:, 0], pos_bdry[:, 1])
        inv_scale = np.linalg.norm(
            def_grad[:, :, (edge_index + 1) % 2], axis=-1
        ) * np.abs(np.linalg.det(def_grad))
        if edge_index == 0 or edge_index == 3:
            # comparevec scaling factor so CCW rotation makes normal
            inv_scale *= -1

        comparevec = np.einsum(
            "jis,js->si", def_grad, def_grad[:, (edge_index + 1) % 2, :]
        )

        # 90 CCW rot
        comparevec = np.flip(comparevec, axis=1) * np.array((-1, 1))[np.newaxis, :]

        raise NotImplementedError("")

        # return np.einsum("sk,ksab,ab->s",
        #     comparevec,
        #     self._lagrange_grads(np.arange(self.degree+1)[np.newaxis,:],
        #             np.arange(self.degree+1)[np.newaxis,np.newaxis,:],
        #             edge_inds[:,0],edge_inds[:,1]),
        #     field) / inv_scale

    def _get_edge_inds(self, edgeID):
        """Returns a (N+1) x 2 array of indices for the
        specifice edge. The i-indices (x) are inds[:,0]
        and j-indices (y) are inds[:,1]
        """
        Np1 = self.degree + 1
        inds = np.empty((Np1, 2), dtype=np.uint32)
        if edgeID == 0:
            inds[:, 0] = Np1 - 1
            inds[:, 1] = np.arange(Np1)
        elif edgeID == 1:
            inds[:, 1] = Np1 - 1
            inds[:, 0] = np.arange(Np1)
        elif edgeID == 2:
            inds[:, 0] = 0
            inds[:, 1] = np.arange(Np1)
        elif edgeID == 3:
            inds[:, 1] = 0
            inds[:, 0] = np.arange(Np1)
        return inds
