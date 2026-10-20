import os
import warnings
from pathlib import Path

import pytest

from geometry_to_physics.data.cad import load_cad_vertices

GT = Path(r"C:\Users\kghol072\physical_fidelity\datasets")
BOWL = GT / "Picasso" / "model" / "bowl" / "bowl.obj"


@pytest.mark.skipif(not BOWL.is_file(), reason="local Picasso CAD missing")
def test_picasso_bowl_load_no_decompression_bomb():
    os.environ["GTP_GT_ROOT"] = str(GT)
    with warnings.catch_warnings():
        warnings.simplefilter("error", category=UserWarning)
        # DecompressionBombWarning subclasses UserWarning
        res = load_cad_vertices("picasso", "1", "bowl")
    assert res.vertices_m.shape[0] > 100
