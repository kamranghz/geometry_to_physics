# geometry_to_physics

Independent research codebase for **From Geometry to Physics: Reference-Anchored Error Propagation in Video-Based 3D Reconstruction** (TCSVT).

This repository is isolated from PhysFidelity3D historical experiment outputs. Configure dataset roots via environment variables (see `.env.example`).

## Layout

- `src/geometry_to_physics/` — library code
- `tests/` — unit, integration, and scientific tests
- `configs/` — experiment configuration
- `audit/` — provenance and validation artifacts
- `outputs/` — generated results (gitignored)

## Development

```bash
pip install -e ".[dev]"
pytest
```
