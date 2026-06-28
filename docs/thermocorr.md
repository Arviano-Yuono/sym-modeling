# ThermoCorr Formatting

ThermoCorr DIC data can be converted into the EUCLID-compatible FEM CSV layout
with `scripts/format_thermocorr_dataset.py`. The formatter normalizes
coordinates and displacements for stable deformation-gradient recovery, scales
the reaction target by the raw specimen height by default, and preserves the full
mesh boundary when sampling.

## Generate a 10k Sampled Dataset

Run from the repo root:

```bash
uv run python scripts/format_thermocorr_dataset.py \
  --input-dir dataset/fem_data/thermocorr \
  --output-dir dataset/fem_data/thermocorr/formatted_sampled_10k \
  --sample-nodes 10000 \
  --overwrite
```

The default sampler is `boundary_coarsen`, which preserves all inferred
specimen-boundary nodes plus the full top reaction boundary. Boundary
preservation has priority over an exact node count, so very small requested
counts can export more nodes than requested.

For comparison with the older exact-count random sampler, add:

```bash
--sample-method random
```

## Generate a Mesh Image

After formatting, generate a mesh PNG with:

```bash
uv run python scripts/plot_thermocorr_mesh.py \
  --data-dir dataset/fem_data/thermocorr/formatted_sampled_10k \
  --step 10 \
  --output dataset/fem_data/thermocorr/thermocorr_sampled_10k_step10.png \
  --show-nodes
```

Change `--step` to inspect another formatted load step.

## Generate a Forward FEM Mesh

To export a DOLFINx/Gmsh mesh from an existing formatted ThermoCorr dataset:

```bash
uv run python scripts/export_thermocorr_mesh.py \
  --data-dir dataset/fem_data/thermocorr_original/formatted_sampled_3k \
  --step 1 \
  --output dataset/fem_data/thermocorr_original/formatted_sampled_3k/mesh_3k.msh \
  --overwrite
```

The mesh uses physical tags `LEFT=1`, `BOTTOM=2`, `RIGHT=3`, `TOP=4`,
`OTHER=5`, and `DOMAIN=11`. `OTHER` covers non-box outer boundary edges, so
the file has the same facet-tag shape expected by the current plate-hole
forward benchmark API, even though ThermoCorr loading still needs its own
benchmark boundary-condition setup.

When regenerating formatted CSVs, the same `.msh` can be written directly:

```bash
uv run python scripts/format_thermocorr_dataset.py \
  --input-dir dataset/fem_data/thermocorr_original \
  --output-dir dataset/fem_data/thermocorr_original/formatted_sampled_3k \
  --sample-nodes 3000 \
  --mesh-path dataset/fem_data/thermocorr_original/formatted_sampled_3k/mesh_3k.msh \
  --overwrite
```

## Reaction Scaling

By default, `output_reactions.csv` stores:

```text
force_vfm / raw_y_span
```

This matches the normalized-coordinate weak-form convention used by the FEM
loaders. To write the raw measured force instead, pass `--raw-force`. To scale
by physical area-like dimensions, pass both `--physical-height` and
`--thickness`.
