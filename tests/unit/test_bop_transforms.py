import numpy as np

from geometry_to_physics.transforms.bop import bop_w2c_to_c2w, m2c_transform
from geometry_to_physics.transforms.se3 import apply, invert


def test_bop_w2c_c2w_roundtrip():
    rng = np.random.default_rng(1)
    R = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    t_mm = rng.normal(size=3) * 100.0
    c2w = bop_w2c_to_c2w(R.tolist(), t_mm.tolist())
    p_w = np.array([[0.2, -0.1, 0.5]])
    p_c = apply(invert(c2w), p_w)
    back = apply(c2w, p_c)
    assert np.allclose(back, p_w, atol=1e-9)


def test_m2c_known_translation_mm():
    verts = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    m2c = m2c_transform(np.eye(3).tolist(), [1000.0, 0.0, 0.0])
    out = apply(m2c, verts)
    assert np.allclose(out[0], [1.0, 0.0, 0.0], atol=1e-9)
