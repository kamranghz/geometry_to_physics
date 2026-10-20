import numpy as np

from geometry_to_physics.transforms.bop import cad_local_to_recon_world, m2c_transform
from geometry_to_physics.transforms.se3 import as_homogeneous


def test_cad_to_recon_world_known_pose():
    """Independent numeric expectation: 1m shift in camera X."""
    cad = np.array([[0.0, 0.0, 0.0], [0.05, 0.0, 0.0]])
    m2c = m2c_transform(np.eye(3).tolist(), [0.0, 0.0, 0.0])
    c2w = as_homogeneous(np.eye(3), np.array([2.0, 0.0, 0.0]))
    v = cad_local_to_recon_world(cad, m2c, c2w)
    assert np.allclose(v[0], [2.0, 0.0, 0.0], atol=1e-9)
    assert np.allclose(v[1, 0], 2.05, atol=1e-9)
