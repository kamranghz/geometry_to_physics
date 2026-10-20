from geometry_to_physics.constants import METHODS
from geometry_to_physics.tcsvt.scene_catalog import count_scene_jobs, iter_scene_jobs


def test_scene_catalog_7x42():
    jobs = list(iter_scene_jobs())
    assert len(jobs) == 42 * len(METHODS)
    assert count_scene_jobs() == 294
    methods = {j.method for j in jobs}
    assert methods == set(METHODS)
