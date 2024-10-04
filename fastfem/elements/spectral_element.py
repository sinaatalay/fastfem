import numpy as np
import fastfem.elements._poly_util as GLL_UTIL
import fastfem.elements.element as element
import scipy as sp


class DeformationGradient2DBadnessException(Exception):
    def __init__(self, val,x,y):            
        super().__init__("Element has too poor of a shape!\n"+
                    f"   def_grad badness = {val:e} at "+
                    f"({x:.6f},{y:.6f})")
            
        # Now for your custom code...
        self.x = x; self.y = y; self.val = val

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

    def locate_point(self,pos_matrix, posx,posy, tol=1e-8, dmin = 1e-7,
                     def_grad_badness_tol = 1e-4, ignore_out_of_bounds = False):
        """Attempts to find the local coordinates corresponding to the
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

        def_grad_badness_tol parameterizes the angle between coordinate
        vectors. If the coordinate vectors line up close enough
        ( abs(sin(theta)) > def_grad_badness_tol ), then an exception is raised.
        This is to ensure that the elements are regularly shaped enough.
        """
        
        Np1 = self.degree + 1
        target = np.array((posx,posy))
        node_errs = np.sum((pos_matrix-target)**2,axis=-1)
        mindex = np.unravel_index(np.argmin(node_errs),(Np1,Np1))
        
        #local coords of guess
        local = np.array((self.knots[mindex[0]],self.knots[mindex[1]]))

        # position poly
        # sum(x^{i,j} L_{i,j}) -> [dim,k,l] (coefficient of cx^ky^l for dim)
        x_poly = np.einsum("ijd,ik,jl->dkl",pos_matrix,
                           self.lagrange_polys,self.lagrange_polys)
        
        #local to global
        def l2g(local):
            return np.einsum("dkl,k,l->d",x_poly,
                    local[0]**np.arange(Np1),local[1]**np.arange(Np1))
        e = l2g(local) - target
        F = 0.5*np.sum(e**2)

            
        # linsearch on gradient descent
        def linsearch(r,step):
            ARMIJO = 0.25
            err = l2g(local + r*step) - target
            F_new = 0.5*np.sum(err**2)
            while F_new > F + (ARMIJO * step) * drF:
                step *= 0.5
                err[:] = l2g(local + r*step) - target
                F_new = 0.5*np.sum(err**2)
            return F_new,step
        
        def clamp_and_maxstep(r):
            #out of bounds check; biggest possible step is to the boundary;
            #note: uses local[]. Additionally r[] is pass-by reference since it
            #can be modified. This is a feature, since that is the "clamp" part

            step = 1
            if ignore_out_of_bounds:
                return step
            for dim in range(2): #foreach dim
                if (r[dim] < 0 and local[dim] == -1) or \
                        (r[dim] > 0 and local[dim] == 1):
                    #ensure descent direction does not point outside domain
                    #this allows constrained minimization
                    r[dim] = 0
                elif r[dim] != 0:
                    # prevent out-of-bounds step by setting maximum step;
                    if r[dim] > 0:
                        step = min(step,(1-local[dim])/r[dim])
                    else:
                        step = min(step,(-1-local[dim])/r[dim])
            return step

        while F > tol:
            #find descent direction by Newton

            #derivative of l2g
            dx_l2g = np.einsum("dkl,k,l->d",x_poly[:,1:,:],
                    np.arange(1,Np1)*local[0]**(np.arange(Np1-1)),
                    local[1]**np.arange(Np1))
            dy_l2g = np.einsum("dkl,k,l->d",x_poly[:,:,1:],
                    local[0]**np.arange(Np1),
                    np.arange(1,Np1)*local[1]**(np.arange(Np1-1)))
            
            #check badness
            defgrad_badness = np.abs(np.linalg.det([dx_l2g,dy_l2g])
                    /(np.linalg.norm(dx_l2g) * np.linalg.norm(dy_l2g)))
            if defgrad_badness < def_grad_badness_tol:
                raise DeformationGradient2DBadnessException(
                    defgrad_badness,local[0],local[1])
            
            #grad of err function (ex^2 + ey^2)/2
            dxF = np.dot(e,dx_l2g)
            dyF = np.dot(e,dy_l2g)

            #solve hessian and descent dir
            hxx = np.sum(dx_l2g*dx_l2g+e*np.einsum("dkl,k,l->d",
                                                   x_poly[:,2:,:],
                    np.arange(1,Np1-1)*np.arange(2,Np1)*
                    local[0]**(np.arange(Np1-2)),
                    local[1]**(np.arange(Np1))))
            hxy = np.sum(dx_l2g*dy_l2g+e*np.einsum("dkl,k,l->d",
                                                   x_poly[:,1:,1:],
                    np.arange(1,Np1)*local[0]**(np.arange(Np1-1)),
                    np.arange(1,Np1)*local[1]**(np.arange(Np1-1))))
            hyy = np.sum(dy_l2g*dy_l2g+e*np.einsum("dkl,k,l->d",
                                                   x_poly[:,:,2:],
                    local[0]**(np.arange(Np1)),
                    np.arange(1,Np1-1)*np.arange(2,Np1)*
                    local[1]**(np.arange(Np1-2))))

            #target newton_rhaphson step
            r_nr = -np.linalg.solve([[hxx,hxy],[hxy,hyy]],[dxF,dyF])
            
            #target grad desc step
            r_gd = -np.array((dxF,dyF))
            r_gd /= np.linalg.norm(r_gd) #scale gd step


            #take the better step between Newton-Raphson and grad desc
            s_nr = clamp_and_maxstep(r_nr)
            s_gd = clamp_and_maxstep(r_gd)
            
            #descent direction -- if dF . r == 0, we found minimum
            drF = dxF * r_gd[0] + dyF * r_gd[1]
            if drF > -dmin:
                break
            F_gd,s_gd = linsearch(r_gd,s_gd)

            #compare to NR
            F_nr = 0.5*np.sum((l2g(local + r_nr*s_nr) - target)**2)
            if F_nr < F_gd:
                local += r_nr*s_nr
                e[:] = l2g(local) - target
                F = F_nr
            else:
                local += r_gd*s_gd
                e[:] = l2g(local) - target
                F = F_gd

            #nudge back in bounds in case of truncation error
            if not ignore_out_of_bounds:
                for dim in range(2):
                    local[dim] = min(1,max(-1,local[dim]))
        return (local, F < tol)

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