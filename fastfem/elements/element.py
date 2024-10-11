import abc

class Element(abc.ABC):
    """Handles the management of element operations. TODO elaborate"""

    @abc.abstractmethod
    def reference_to_real(self,pos_matrix,X,Y):
        """Maps the points (X,Y) from reference coordinates
        to real positions. The result is an array of shape
        (*point_shape,2), where the last index is the dimension, and
        `point_shape = np.broadcast_shapes(X.shape,Y.shape)`.
        
        pos_matrix is an array of shape (*ref_shape,2) for the element's
        expected shape ref_shape, which corresponds to the element's basis.
        pos_matrix[...,i] is the i^th coordinate of the position relevant to
        the basis function indexed in (...).

        Args:
            pos_matrix (ndarray): an array representing the positions of the
                    element nodes. This is of the shape basis_shape
            X (ndarray or numeric): X coordinates of the points to map. The
                    shape of X must be compatible with the shape of Y.
            Y (ndarray or numeric): Y coordinates of the points to map. The
                    shape of Y must be compatible with the shape of X.
        """
        pass
    
    @abc.abstractmethod
    def basis_mass_matrix(self,pos_matrix):
        """Recovers the mass matrix entries = ∫ ɸᵢɸⱼ dx² for this element.

        Args:
            pos_matrix (ndarray): an array representing the positions of the
                    element nodes. This is of the shape basis_shape
        """
        pass
    
    @abc.abstractmethod
    def basis_stiffness_matrix_times_field(self, pos_matrix, field):
        """Computes the vector ∫ ∇ɸᵢ • ∇f dx² of the H¹ products on the element
        over the whole basis on the element. Using the multi-index 'n' for the
        field f, we can write these terms as ∫(∂ⱼɸᵢ)gⱼₖ(∂ₖfₙ)dx², where gⱼₖ
        represents the inner product on the dual space (since ∂ adds a
        covariant index). The result's index order is (i,n), where i may be a
        multi-index, based on the exact element, and is placed into a numpy
        array of shape (*basis_shape,*field_shape).

        Args:
            pos_matrix (ndarray): an array representing the positions of the
                    element nodes. This is of the shape basis_shape
            field (ndarray): an array of shape (*basis_shape,*field_shape)
                    representing the field to integrate with.
        """
        pass

    #TODO boundary integrals, other field * basis integral configurations