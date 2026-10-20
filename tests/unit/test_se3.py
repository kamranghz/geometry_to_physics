import numpy as np

from geometry_to_physics.transforms.se3 import apply, as_homogeneous, invert


def test_se3_roundtrip():
    R = np.array([[0.36, -0.48, 0.8], [0.8, 0.6, 0], [-0.48, 0.64, 0.6]])
    t = np.array([0.1, -0.2, 0.3])
    T = as_homogeneous(R, t)
    p = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    back = apply(invert(T), apply(T, p))
    assert np.allclose(back, p, atol=1e-9)
