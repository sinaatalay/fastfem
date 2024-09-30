class Element:
    """Handles the management of element operations. TODO elaborate
    TODO consider setting this as an ABC"""

    def reference_to_real(self,pos_matrix,X,Y):
        """Maps the points (X,Y) from reference coordinates
        to real positions. The result is an array of shape
        (*X.shape,2), where the first index is the dimension
        
        pos_matrix is an array of shape (*ref_shape,2) for the element's
        expected shape ref_shape, which corresponds to the element's basis.
        pos_matrix[...,i] is the i^th coordinate of the position relevant to
        the basis function indexed in (...)."""
        raise NotImplementedError("Element base class called.")
    
    def basis_mass_matrix(self,pos_matrix):
        """Recovers the mass matrix M_{ij} = int(phi_i*phi_j)
        """
        raise NotImplementedError("Element base class called.")
    
    def basis_stiffness_matrix(self,pos_matrix):
        """Recovers the stiffness matrix M_{ij} = int(del phi_i  dot  del phi_j)
        """
        # TODO consider skipping this, and instead have something like mult_stiffness_matrix
        # to compute integral int(del phi_i  dot  del F) for a field F
        raise NotImplementedError("Element base class called.")

    #TODO boundary integrals