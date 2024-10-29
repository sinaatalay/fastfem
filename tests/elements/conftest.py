import numpy as np
import pytest


def transform_posmatrix(pos_matrix, mod, *args):
    if mod == "translate":
        if len(args) < 2:
            raise ValueError(f"modifier '{mod}' expects 2 arguments! (dx,dy)")
        vec = np.array([args[0], args[1]])
        pos_matrix = pos_matrix + vec  # no in-place op, since we want a copy
    elif mod == "rotate":
        if len(args) < 1:
            raise ValueError(f"modifier '{mod}' expects 1 argument! (angle)")
        t = args[0]
        rotmat = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        pos_matrix = (rotmat @ np.expand_dims(pos_matrix, -1)).squeeze(-1)
    elif mod == "scale":
        if len(args) < 2:
            raise ValueError(f"modifier '{mod}' expects 2 arguments! (scalex,scaley)")
        vec = np.array([args[0], args[1]])
        pos_matrix = pos_matrix * vec  # no in-place op, since we want a copy
    elif mod == "lin_trans":
        if len(args) < 4:
            raise ValueError(f"modifier '{mod}' expects 4 arguments! (m00,m01,m10,m11)")
        A = np.array([[args[0], args[1]], [args[2], args[3]]])
        pos_matrix = (A @ np.expand_dims(pos_matrix, -1)).squeeze(-1)
    else:
        raise ValueError(f"'{mod}' not acceptable element modifier!")
    return pos_matrix


_PRESET_TRANSFORMS = {
    "ref": lambda x: x,
    "translated": lambda x: transform_posmatrix(x, "translate", 5, -2),
    "rotated": lambda x: transform_posmatrix(x, "rotate", 1),
    "x-scaled": lambda x: transform_posmatrix(x, "scale", 2, 1),
    "y-scaled": lambda x: transform_posmatrix(x, "scale", 1, 2),
    "combo1": lambda x: transform_posmatrix(
        transform_posmatrix(x, "lin_trans", 2, 1, -1, 1), "translate", -4, 2
    ),
    "combo2": lambda x: transform_posmatrix(
        transform_posmatrix(x, "lin_trans", 0.5, 1.3, 10, 0.3), "translate", 300, 600
    ),
}


@pytest.fixture(scope="module", params=_PRESET_TRANSFORMS.keys())
def transformation(request):
    name = request.param
    return _PRESET_TRANSFORMS[name]


@pytest.fixture(scope="module", params=[0, 1, 2, 3])
def transform_stack(request):
    transforms = _PRESET_TRANSFORMS.values()
    # param is number of dims for element position array
    ndims = request.param
    stackshape = tuple(3 for _ in range(ndims))
    stacksize = np.prod(stackshape, dtype=int)

    def stack_transform(x):
        y = np.empty(stackshape + x.shape)
        transformed = [f(x) for f in transforms]
        for i in range(stacksize):
            y[*np.unravel_index(i, stackshape), ...] = transformed[i % len(transforms)]
        return y

    return stack_transform
