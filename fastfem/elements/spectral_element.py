import numpy as np
import fastfem.elements._poly_util as GLL_UTIL
import fastfem.elements.element as element
import scipy as sp

class SpectralElement2D(element.Element):
    """A spectral element in 2 dimensions of order N, leading to (N+1)^2 nodes.
    GLL quadrature is used to diagonalize the mass matrix.
    """
    def __init__(self,degree: int):
        super().__init__()
        self.degree = degree
        self.knots = GLL_UTIL.get_GLL_knots(degree)
        self.weights = GLL_UTIL.get_GLL_weights(degree)
        self.lagrange_polys = np.array(GLL_UTIL.build_GLL_polys(degree))

    
    def reference_to_real(self,pos_matrix,X,Y):
        """Maps the points (X,Y) from reference coordinates
        to real positions. The result is an array of shape
        (*X.shape,2), where the first index is the dimension.
        X and Y must have compatible shape.
        
        pos_matrix is an array of shape (degree+1,degree+1,2) where
        pos_matrix[i,j] = [x,y] when (x,y) is the position of node i (along
        the x-axis) and j (along the y-axis)"""
        if not isinstance(X,np.ndarray):
            X = np.array(X)
        if not isinstance(Y,np.ndarray):
            Y = np.array(Y)
        Np1 = self.degree + 1
        # x^{i,j} L_{i,j}(X,Y)
        #lagrange_polys[i,k] : component c in term cx^k of poly i
        return np.einsum("ijk,ia,...a,jb,...b->...k",pos_matrix,
            self.lagrange_polys,np.expand_dims(X,-1) ** np.arange(Np1),
            self.lagrange_polys,np.expand_dims(Y,-1) ** np.arange(Np1))

    def _lagrange_deriv(self, lag_index, deriv_order, knot_index):
        """Calculates the derivative
        [(d/dx)^{deriv_order} L_{lag_index}(x)]_{x=x_{knot_index}}

        Note that "lagrange" refers to the lagrange interpolation polynomial,
        not lagrangian coordinates. This is a one-dimension helper function.
        """
        if not isinstance(lag_index,np.ndarray):
            lag_index = np.array(lag_index)
        if not isinstance(deriv_order,np.ndarray):
            deriv_order = np.array(deriv_order)
        if not isinstance(knot_index,np.ndarray):
            knot_index = np.array(knot_index)
        #dims of input arrays
        indims = max(lag_index.ndim,deriv_order.ndim,knot_index.ndim)
        lag_index = lag_index.reshape((1,*lag_index.shape,
                    *[1 for _ in range(indims-lag_index.ndim)]))
        deriv_order = deriv_order.reshape((1,*deriv_order.shape,
                    *[1 for _ in range(indims-deriv_order.ndim)]))
        knot_index = knot_index.reshape((1,*knot_index.shape,
                    *[1 for _ in range(indims-knot_index.ndim)]))
        
        N = self.degree
        shape = (N+1,*[1 for _ in range(indims)])
        arangeshape = np.arange(N+1).reshape(shape)
        L = self.lagrange_polys[lag_index,arangeshape]
        filter = arangeshape >= deriv_order
        return np.sum(L * sp.special.perm(arangeshape,deriv_order)
            * self.knots[knot_index]
            **(filter * (arangeshape-deriv_order))\
            * (filter),axis=0)

    def def_grad(self,pos_matrix,X,Y):
        """Calculates the deformation gradient matrix dX/(dxi)
        at the reference coordinates (X,Y).
        X and Y must be broadcastable to the same shape.
        The result is an array with shape (*X.shape,2,2) where
        the first new index specifies the coordinate in global space and the
        second index specifies the coordinate in reference space.

        pos_matrix is an array of shape (degree+1,degree+1,2) where
        pos_matrix[i,j] = [x,y] when (x,y) is the position of node i (along
        the x-axis) and j (along the y-axis)
        """
        raise NotImplementedError("Not yet (is it needed?)")
        if not isinstance(i,np.ndarray):
            i = np.array(i)
        if not isinstance(j,np.ndarray):
            j = np.array(j)
        indims = max(i.ndim,j.ndim)
        i = np.expand_dims(i,tuple(range(i.ndim,indims)))
        j = np.expand_dims(j,tuple(range(j.ndim,indims)))
        
        grad = np.einsum( "abl,a...,b...->l...",
            self.fields["positions"],
            self.lagrange_deriv(np.arange(self.degree+1), # a
                            np.array([1,0])[np.newaxis,:],# (*,k)
                            i[np.newaxis,np.newaxis]),    # (*,*,indims)
            self.lagrange_deriv(np.arange(self.degree+1), # b
                            np.array([0,1])[np.newaxis,:],# (*,k)
                            j[np.newaxis,np.newaxis])     # (*,*,indims)
            )
        return grad
    
    def _def_grad(self,pos_matrix,i,j):
        """Calculates the deformation gradient matrix dX/(dxi)
        at the reference coordinates xi = (x_i,y_j).
        i and j must be broadcastable to the same shape.
        The result is an array with shape (*X.shape,2,2) where
        the first new index specifies the coordinate in global space and the
        second index specifies the coordinate in reference space.
        """
        if not isinstance(i,np.ndarray):
            i = np.array(i)
        if not isinstance(j,np.ndarray):
            j = np.array(j)
        indims = max(i.ndim,j.ndim)
        i = np.expand_dims(i,tuple(range(i.ndim,indims)))
        j = np.expand_dims(j,tuple(range(j.ndim,indims)))
        
        grad = np.einsum( "abl,ak...,bk...->...lk",
            pos_matrix,
            self._lagrange_deriv(np.arange(self.degree+1), # a
                            np.array([1,0])[np.newaxis,:],# (*,k)
                            i[np.newaxis,np.newaxis]),    # (*,*,indims)
            self._lagrange_deriv(np.arange(self.degree+1), # b
                            np.array([0,1])[np.newaxis,:],# (*,k)
                            j[np.newaxis,np.newaxis])     # (*,*,indims)
            )
        return grad
    
    def _lagrange_grads(self,pos_matrix,a,b,i,j, cartesian = False):
        """Writing phi_{a,b}(x_i,y_j) = l_a(x_i)l_b(y_j),
        calculates for arrays a,b,i,j:
            grad phi_{a,b}(x_i,y_j)
        where the last index specifies the dimension, and the rest
        match the shape of a,b,i, and j

        cartesian==True specifies that this gradient is partial_{x}.
        Otherwise, it is in Lagrangian coordinates, so partial_{xi}.
        """
        if not isinstance(a,np.ndarray):
            a = np.array(a)
        if not isinstance(b,np.ndarray):
            b = np.array(b)
        if not isinstance(i,np.ndarray):
            i = np.array(i)
        if not isinstance(j,np.ndarray):
            j = np.array(j)

        #TODO this is for sure broken

        dims = max(i.ndim,j.ndim,a.ndim,b.ndim)+1
        i = np.expand_dims(i,tuple(range(i.ndim+1,dims)))
        j = np.expand_dims(j,tuple(range(j.ndim+1,dims)))
        a = np.expand_dims(a,tuple(range(a.ndim+1,dims)))
        b = np.expand_dims(b,tuple(range(b.ndim+1,dims)))
        #nabla_I L(...)
        lagrangian= (self._lagrange_deriv(#l_a^k(x)
                a,np.array([1,0]),i)
            * self._lagrange_deriv(       #l_b^k(y)
                b,np.array([0,1]),j))
        if cartesian:
            #deformation gradient:
            # [dX/dxi1, dX/dxi2]
            # [dY/dxi1, dY/dxi2]
            grad = self.def_grad(pos_matrix,i,j)[:,:,0] #collapse the dim for k
            #we need to get d\xi/dx
            gradinv = np.linalg.inv(grad.T)
            # [dxi1/dX, dxi1/dY] T   [dxi1/dX, dxi2/dX]
            # [dxi2/dX, dxi2/dY]   = [dxi1/dY, dxi2/dY]

            #lagrangian is [dL/dxi1, dL/xi2]^T
            return (gradinv @ np.expand_dims(lagrangian.T,-1)).T[0]
        return lagrangian
        
    def basis_mass_matrix(self,pos_matrix):

        #compute using GLL quadrature

        # int (phi1 * phi2) = sum_i(phi1(x_i) phi2(x_i) w_i * J(x_i))

        #weights [i,j] times jacobian
        w = self.weights[:,np.newaxis] * self.weights[np.newaxis,:] * \
            np.abs(np.linalg.det(
                self.def_grad(pos_matrix,np.arange(self.degree+1),
                              np.arange(self.degree+1)[np.newaxis,:])
            ))
        return w
    
    def basis_stiffness_matrix(self, pos_matrix):

        #compute using GLL quadrature

        raise NotImplementedError("TODO")
    
    def _bdry_normalderiv(self,pos_matrix,edge_index,field):
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
        def_grad = self._def_grad(pos_matrix,edge_inds[:,0],edge_inds[:,1])
        inv_scale = (
            np.linalg.norm(def_grad[:,(edge_index+1) % 2,:],axis=0)
            * np.abs(np.linalg.det(def_grad.T))
        )
        if edge_index == 0 or edge_index == 3:
            #comparevec scaling factor so CCW rotation makes normal
            inv_scale *= -1

        comparevec = np.einsum("jis,js->si",def_grad,
                    def_grad[:,(edge_index+1) % 2,:])
        
        #90 CCW rot
        comparevec = np.flip(comparevec,axis=1) * np.array((-1,1))[np.newaxis,:]
        


        return np.einsum("sk,ksab,ab->s",
            comparevec,
            self._lagrange_grads(np.arange(self.degree+1)[np.newaxis,:],
                    np.arange(self.degree+1)[np.newaxis,np.newaxis,:],
                    edge_inds[:,0],edge_inds[:,1]),
            field) / inv_scale

    def _get_edge_inds(self,edgeID):
        """Returns a (N+1) x 2 array of indices for the
        specifice edge. The i-indices (x) are inds[:,0]
        and j-indices (y) are inds[:,1]
        """
        Np1 = self.degree + 1 
        inds = np.empty((Np1,2),dtype=np.uint32)
        if edgeID == 0:
            inds[:,0] = Np1-1
            inds[:,1] = np.arange(Np1)
        elif edgeID == 1:
            inds[:,1] = Np1-1
            inds[:,0] = np.arange(Np1)
        elif edgeID == 2:
            inds[:,0] = 0
            inds[:,1] = np.arange(Np1)
        elif edgeID == 3:
            inds[:,1] = 0
            inds[:,0] = np.arange(Np1)
        return inds