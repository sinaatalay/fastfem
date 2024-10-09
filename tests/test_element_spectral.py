import pytest
import numpy as np

import fastfem
from fastfem.elements import _poly_util
from fastfem.elements import spectral_element



def get_ref_posmatrix(order: int):
  knots = _poly_util.get_GLL_knots(order)
  out = np.empty((order+1,order+1,2))
  out[:,:,0] = knots[:,np.newaxis]
  out[:,:,1] = knots[np.newaxis,:]
  return out

@pytest.fixture(scope="module", params=[3,4,5])
def element(request):
  order = request.param
  elem = spectral_element.SpectralElement2D(order)
  out = np.empty((order+1,order+1,2))
  out[:,:,0] = elem.knots[:,np.newaxis]
  out[:,:,1] = elem.knots[np.newaxis,:]
  return (elem,out)


def transform_posmatrix(pos_matrix,mod,*args):
  if mod == "translate":
    if len(args) < 2:
      raise ValueError(f"modifier '{mod}' expects 2 arguments! (dx,dy)")
    vec = np.array([args[0],args[1]])
    pos_matrix = pos_matrix + vec #no in-place op, since we want a copy
  elif mod == "rotate":
    if len(args) < 1:
      raise ValueError(f"modifier '{mod}' expects 1 argument! (angle)")
    t = args[0]
    rotmat = np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]])
    pos_matrix = (rotmat @ np.expand_dims(pos_matrix,-1)).squeeze(-1)
  elif mod == "scale":
    if len(args) < 2:
      raise ValueError(f"modifier '{mod}' expects 2 arguments! (scalex,scaley)")
    vec = np.array([args[0],args[1]])
    pos_matrix = pos_matrix * vec #no in-place op, since we want a copy
  elif mod == "lin_trans":
    if len(args) < 4:
      raise ValueError(f"modifier '{mod}' expects 4 arguments! (m00,m01,m10,m11)")
    A = np.array([[args[0],args[1]],[args[2],args[3]]])
    pos_matrix = (A @ np.expand_dims(pos_matrix,-1)).squeeze(-1)
  else:
    raise ValueError(f"'{mod}' not acceptable element modifier!")
  return pos_matrix

@pytest.fixture(scope="module",params=["ref","translated","rotated","x-scaled","y-scaled","combo1","combo2"])
def transformation(request):
  name = request.param
  if name == "ref":
    return lambda x:x
  if name == "translated":
    return lambda x: transform_posmatrix(x,"translate",5,-2)
  if name == "rotated":
    return lambda x: transform_posmatrix(x,"rotate",1)
  if name == "x-scaled":
    return lambda x: transform_posmatrix(x,"scale",2,1)
  if name == "y-scaled":
    return lambda x: transform_posmatrix(x,"scale",1,2)
  if name == "combo1":
    return lambda x: transform_posmatrix(
      transform_posmatrix(x,"lin_trans",2,1,-1,1),
      "translate",-4,2)
  if name == "combo2":
    return lambda x: transform_posmatrix(
      transform_posmatrix(x,"lin_trans",0.5,1.3,10,0.3),
      "translate",300,600)

@pytest.fixture(scope="module")
def transformed_element(element,transformation):
  return element[0],transformation(element[1]),transformation

@pytest.fixture(params=[(0,0),(-1,-1),(1,-1),(1,1),(-1,1),(0.5,0.5),(-0.33,0.84)])
def ref_coords(request):
  return np.array(request.param)

@pytest.fixture(params=[np.array((0,0)),np.array(((-1,-1),(1,-1),(1,1),(-1,1))),np.array((((-0.5,-0.3),(0.7,-0.2)),((0.2,0.8),(-0.1,1))))])
def ref_coords_arr(request):
  return request.param

@pytest.fixture(params=[
  ((5,),(5,)),
  ((2,6),(2,6)),
  ((2,4,3),(2,4,3)),
  ((3,),(4,3)),
  ((4,3),(3,)),
  ((1,3),(4,3)),
  ((4,3),(1,3)),
  ((4,1,2),(2,1)),
  ((2,1),(4,1,2)),
])
def broadcastable_shapes(request):
  return request.param[0],request.param[1],np.broadcast_shapes(*request.param)

@pytest.fixture
def broadcastable_shape_triples(broadcastable_shapes):
  a,b,target = broadcastable_shapes

  num_entries = 3 #for triples


  #what each entry can be (this fixture fails if this is >= 10)
  candidates = [tuple(),a,b]
  #the tuple needs at least these
  requires = [1,2]

  def is_valid(inds):
    #do checks:
    if any(r not in inds for r in requires):
      #we do not have a required entry
      return False

    #placeholder for more checks

    return True
  ind_gen = (
      [int(c) for c in np.base_repr(i,len(candidates))]
      for i in range(len(candidates)**num_entries)
  )
  return target,([candidates[i] for i in inds] for inds in ind_gen if is_valid(inds))

#===================

def test_lagrange_poly_coefs1D(element):
  elem = element[0]
  elem._lagrange_derivs = dict()
  elem._lagrange_derivs[0] = elem._lagrange_polys

  sigfigs = 5
  #we will use central finite difference which has O(h^2) error
  h = 10**-((sigfigs+3)//2)

  for deriv_order in range(0,elem.degree+1):
    #verify that the lagrange derivatives are being set in the dictionary.
    P = elem.lagrange_poly1D(deriv_order)
    np.testing.assert_almost_equal(elem._lagrange_derivs[deriv_order],P,
        err_msg=f"derivative L^{({deriv_order})} is not being set properly in the dictionary!")
    assert set(elem._lagrange_derivs) == set(np.arange(deriv_order+1)),\
        f"Expected dictionary keys not found! Are they being improperly stored?"
  
  for deriv_order in range(1,elem.degree+1):
    num_terms = elem.degree+1-deriv_order
    #polynomials of degree p are uniquely defined by their values at p+1 unique points. This is how we check equality.
    test_x = np.linspace(-1,1,num_terms)

    #L_i^(deriv_order), compare to L_i^(...-1) with finite difference
    # polys are stored as P[i,k] : component c in term cx^k of poly P_i
    #L_i^(deriv_order-1)
    L = lambda x: np.einsum("ia,...a",elem.lagrange_poly1D(deriv_order-1),np.expand_dims(x,-1) ** np.arange(num_terms+1))
    #L_i^(deriv_order)
    Lp = lambda x: np.einsum("ia,...a",elem.lagrange_poly1D(deriv_order),np.expand_dims(x,-1) ** np.arange(num_terms))
    np.testing.assert_almost_equal(Lp(test_x),(L(test_x + h) - L(test_x - h))/(2*h),decimal=sigfigs,
        err_msg=f"derivative L^({deriv_order}) does not match the central difference on L^({deriv_order-1})!")
    


def test_lagrange_evals1D(element,broadcastable_shapes):
  elem = element[0]

  for deriv_order in range(0,elem.degree+1):
    #verify that the lagrange derivatives are evaluated correctly
    P = elem.lagrange_poly1D(deriv_order)
    test_points = np.linspace(-1,1,np.prod(broadcastable_shapes[1])).reshape(broadcastable_shapes[1])
    test_indices = np.arange(np.prod(broadcastable_shapes[0])).reshape(broadcastable_shapes[0]) % P.shape[0]

    eval_pts = elem.lagrange_eval1D(deriv_order,test_indices,test_points)
    assert eval_pts.shape == broadcastable_shapes[2], "Did not broadcast into the right shape!"

    test_points  = np.broadcast_to(test_points, broadcastable_shapes[2])
    test_indices = np.broadcast_to(test_indices,broadcastable_shapes[2])
    
    it = np.nditer(eval_pts,flags=["multi_index"])
    degp1 = elem.degree+1 - deriv_order #degree+1 of P
    for Px in it:
      ind = test_indices[it.multi_index]
      x = test_points[it.multi_index]
      np.testing.assert_almost_equal(Px,np.dot(
          P[ind,:],x**np.arange(degp1)),
          err_msg=f"index {it.multi_index}: L_{ind}^({deriv_order}) ({x}) disagreement")


#test depends on reference_to_real
def test_def_grad_eval(transformed_element,ref_coords_arr):
  elem = transformed_element[0]; points = transformed_element[1]; transformation = transformed_element[2]
  X,Y = np.split(ref_coords_arr,2,axis=-1)
  X = X.squeeze(-1)
  Y = Y.squeeze(-1)
  
  grads = elem.def_grad(points,X,Y)

  sigfigs = 5
  #we will use central finite difference which has O(h^2) error
  h = 10**-((sigfigs+3)//2)

  x_derivs = (elem.reference_to_real(points, X+h, Y) - elem.reference_to_real(points, X-h, Y))/(2*h)
  y_derivs = (elem.reference_to_real(points, X, Y+h) - elem.reference_to_real(points, X, Y-h))/(2*h)

  np.testing.assert_almost_equal(grads,np.stack((x_derivs,y_derivs),-1),decimal=sigfigs)


#@pytest.mark.skip
def test_reference_to_real(transformed_element,ref_coords_arr):
  elem = transformed_element[0]; points = transformed_element[1]; transformation = transformed_element[2]
  
  true_pos = transformation(ref_coords_arr)
  np.testing.assert_almost_equal(elem.reference_to_real(points,*[v.squeeze(-1) for v in np.split(ref_coords_arr,2,axis=-1)]),true_pos)


def test_real_to_reference_interior(transformed_element,ref_coords):
  elem = transformed_element[0]; points = transformed_element[1]; transformation = transformed_element[2]
  
  true_pos = transformation(ref_coords)

  recover_ref = elem.locate_point(points,true_pos[0],true_pos[1],tol=1e-10,ignore_out_of_bounds = True)
  assert recover_ref[1], "Test with ignore_out_of_bounds flag. Should be True for point being found (loss(recover_ref) < tol)."
  np.testing.assert_almost_equal(recover_ref[0],ref_coords,err_msg="Test with ignore_out_of_bounds flag")


  recover_ref = elem.locate_point(points,true_pos[0],true_pos[1],tol=1e-10,ignore_out_of_bounds = False)
  assert recover_ref[1], "Test without ignore_out_of_bounds flag. Should be True for point being found (loss(recover_ref) < tol)."
  np.testing.assert_almost_equal(recover_ref[0],ref_coords,err_msg="Test without ignore_out_of_bounds flag")

def test_field_grad(transformed_element):
  elem, points, transformation = transformed_element
  X = np.linspace(-1,1,elem.num_nodes)[:,np.newaxis]
  Y = np.linspace(-1,1,elem.num_nodes)

  sigfigs = 5
  #we will use central finite difference which has O(h^2) error
  h = 10**-((sigfigs+3)//2)

  #this hinges on linearity of field values
  field = np.zeros((elem.num_nodes,elem.num_nodes,elem.num_nodes,elem.num_nodes))
  enumeration = (np.arange(elem.num_nodes**2) % elem.num_nodes,
                 np.arange(elem.num_nodes**2) //elem.num_nodes)
  field[enumeration[0],enumeration[1],enumeration[0],enumeration[1]] = 1
  grads = elem.field_grad(field,X,Y)
  cartgrads = elem.field_grad(field,X,Y,pos_matrix=points)
  x_derivs = (elem.interp_field(field, X+h, Y) - elem.interp_field(field, X-h, Y))/(2*h)
  y_derivs = (elem.interp_field(field, X, Y+h) - elem.interp_field(field, X, Y-h))/(2*h)
  grads_comp = np.stack((x_derivs,y_derivs),-1)
  np.testing.assert_almost_equal(grads,grads_comp,decimal=sigfigs,
      err_msg="Local-coordinate gradient disagreement")
  
  def_grad = elem.def_grad(points,X,Y)
  grads_cart_to_lag = np.einsum("...ij,...i->...j",def_grad,cartgrads)
  np.testing.assert_almost_equal(grads_cart_to_lag,grads,decimal=sigfigs,
      err_msg="Local->global->local disagrees with local")

def meshquad(X,Y,F):
  """
  In case this is ever needed, this is a function to compute the integral on a 2D mesh.
  Integrals are estimated by estimating integrals on triangles, which are defined by
  indices ([i,j] - [i+1,j] - [i,j+1]) ([i+1,j+1] - [i+1,j] - [i,j+1])
  """
  def cross_mag(XA,YA,XB,YB):
    return np.abs(XA*YB - XB*YA)
  F00 = F[:-1,:-1];F01 = F[:-1,1:];F10 = F[1:,:-1];F11 = F[1:,1:]
  X00 = X[:-1,:-1];X01 = X[:-1,1:];X10 = X[1:,:-1];X11 = X[1:,1:]
  Y00 = Y[:-1,:-1];Y01 = Y[:-1,1:];Y10 = Y[1:,:-1];Y11 = Y[1:,1:]
  axes = np.arange(X.ndim)
  return np.sum(cross_mag(X10-X00,Y10-Y00,X01-X00,Y01-Y00) * (F00 + F01 + F10)/6,axes) +\
         np.sum(cross_mag(X10-X11,Y10-Y11,X01-X11,Y01-Y11) * (F11 + F01 + F10)/6,axes)

def test_stiffness_matrix(transformed_element):
  elem, points, transformation = transformed_element
  #weights [i,j] times jacobian
  w = elem.weights[:,np.newaxis] * elem.weights[np.newaxis,:] * \
      np.abs(np.linalg.det(
          elem.def_grad(points,np.arange(elem.num_nodes),
                        np.arange(elem.num_nodes)[np.newaxis,:])
      ))
  
  # F(x_i,x_j) = delta_{im}delta_{jn}

  #equiv: # L_m(x_i)L_n(x_j)
  field = np.zeros((elem.num_nodes,elem.num_nodes,elem.num_nodes,elem.num_nodes))
  enumeration = (np.arange(elem.num_nodes**2) % elem.num_nodes,
                 np.arange(elem.num_nodes**2) //elem.num_nodes)
  field[enumeration[0],enumeration[1],enumeration[0],enumeration[1]] = 1

  # [i,j, m,n, dim] partial_dim phi_{mn}(xi,xj)
  field_grad = elem.field_grad(field,elem.knots[:,np.newaxis],elem.knots[np.newaxis,:],pos_matrix=points)
  
  # sum_{ij} w_{ij}( partial_dim phi_{mn}(xi,xj) )( partial_dim F_{ab}(xi,xj) )
  stiff = np.einsum("ij,ijmnd,ijabd->mnab",w,field_grad,field_grad)

  np.testing.assert_almost_equal(elem.basis_stiffness_matrix_times_field(points,field),stiff)
  
  np.testing.assert_almost_equal(elem.basis_stiffness_matrix_diagonal(points),
      np.einsum("mnmn->mn",stiff))

  

  

if __name__ == "__main__":
  pass

