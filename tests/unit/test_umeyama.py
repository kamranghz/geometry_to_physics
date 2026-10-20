import numpy as np

from geometry_to_physics.registration.camera_alignment import apply_sim3_to_points, umeyama_sim3


def test_umeyama_recovers_known_sim3():
    rng = np.random.default_rng(0)
    scale = 2.5
    R = np.array([[0.36, -0.48, 0.8], [0.8, 0.6, 0], [-0.48, 0.64, 0.6]])
    t = np.array([0.1, -0.2, 0.3])
    src = rng.normal(size=(12, 3))
    dst = apply_sim3_to_points(src, scale, R, t)
    s_hat, R_hat, t_hat = umeyama_sim3(src, dst)
    rec = apply_sim3_to_points(src, s_hat, R_hat, t_hat)
    assert abs(s_hat - scale) < 1e-6
    assert np.allclose(rec, dst, atol=1e-6)
