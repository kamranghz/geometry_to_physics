import numpy as np

from geometry_to_physics.geometry.inertia import box_inertia, rotate_inertia, sphere_inertia


def test_box_inertia_principal_values():
    m = 12.0
    I = box_inertia(m, 2.0, 3.0, 4.0)
    expected = np.diag([m * (9 + 16) / 12, m * (4 + 16) / 12, m * (4 + 9) / 12])
    assert np.allclose(I, expected)


def test_sphere_inertia():
    m, r = 5.0, 0.2
    I = sphere_inertia(m, r)
    val = 0.4 * m * r**2
    assert np.allclose(I, np.diag([val, val, val]))


def test_inertia_rotation_trace_invariant():
    I = box_inertia(2.0, 1.0, 2.0, 3.0)
    R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
    Ir = rotate_inertia(I, R)
    assert abs(np.trace(I) - np.trace(Ir)) < 1e-9
