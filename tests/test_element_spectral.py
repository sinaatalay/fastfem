import pytest
import numpy as np

import fastfem
from fastfem.elements import _poly_util
from fastfem.elements import spectral_element



def get_ref_posmatrix(order: int):
  knots = _poly_util.get_GLL_knots(order)
  out = np.empty((order+1,order+1,2))
  out[:,:,0] = elem.knots[:,np.newaxis]
  out[:,:,1] = elem.knots[np.newaxis,:]
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


#===================

@pytest.mark.skip
def test_transformations_shape(transformation,ref_coords_arr):
  assert transformation(ref_coords_arr).shape == ref_coords_arr.shape

@pytest.mark.skip
def test_reference_to_real_not_arr(transformed_element,ref_coords):
  elem = transformed_element[0]; points = transformed_element[1]; transformation = transformed_element[2]
  
  true_pos = transformation(ref_coords)
  np.testing.assert_almost_equal(elem.reference_to_real(points,ref_coords[0],ref_coords[1]),true_pos)

#@pytest.mark.skip
def test_reference_to_real(transformed_element,ref_coords_arr):
  elem = transformed_element[0]; points = transformed_element[1]; transformation = transformed_element[2]
  
  true_pos = transformation(ref_coords_arr)
  np.testing.assert_almost_equal(elem.reference_to_real(points,*[v.squeeze(-1) for v in np.split(ref_coords_arr,2,axis=-1)]),true_pos)