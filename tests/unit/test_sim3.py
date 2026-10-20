import numpy as np

from geometry_to_physics.transforms.se3 import as_homogeneous
from geometry_to_physics.transforms.sim3 import apply_sim3, compose_sim3, roundtrip_error


def test_sim3_compose_identity():
    R = np.eye(3)
    T = as_homogeneous(R, np.zeros(3))
    pts = np.random.default_rng(0).normal(size=(50, 3))
    assert roundtrip_error(T, 2.5, pts) < 1e-9
    Tc, sc = compose_sim3(T, 2.0, T, 3.0)
    assert abs(sc - 6.0) < 1e-12
    p = np.array([[1.0, 0.0, 0.0]])
    assert np.allclose(apply_sim3(Tc, sc, p), apply_sim3(T, 2.0, apply_sim3(T, 3.0, p)), atol=1e-9)
