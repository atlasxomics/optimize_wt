# optimize_wt

`optimize_wt` is a Latch Bio workflow for optimizing clustering parameters for
spatial whole-transcriptome RNA-seq data.

It starts from STAR/STARsolo-style gene-expression outputs plus a matching
`spatial/` directory for each run, preprocesses the data, and evaluates
multiple clustering parameter sets in parallel. The workflow supports two
clustering backends:

- `scanpy`: PCA/Harmony-based clustering
- `stagate`: STAGATE embedding followed by clustering on the learned spatial
  representation

## What The Workflow Does

For each supplied Run, the workflow:

1. Loads the count matrix, barcodes, genes/features, and spatial positions.
2. Filters to in-tissue barcodes and applies QC filtering.
3. Normalizes counts, log-transforms the matrix, and selects highly variable
   genes.
4. Builds either:
   - a Scanpy PCA/Harmony embedding (`clustering_backend="scanpy"`), or
   - a STAGATE embedding (`clustering_backend="stagate"`).
5. Iterates over clustering parameter sets in parallel and writes one
   `combined.h5ad` per successful set.
6. Aggregates UMAPs, spatial plots, medians, and spatial coherence scores into
   the final output directory.

## Input Requirements

Each Run must include:

- `run_id`: unique sample identifier
- `gex_dir`: a STAR/STARsolo output directory containing one of:
  `UniqueAndMult-EM.mtx`, `UniqueAndMult-EM.mtx.gz`, `matrix.mtx`, or
  `matrix.mtx.gz`, plus matching barcode and gene/feature tables
- `spatial_dir`: a directory containing either `tissue_positions_list.csv` or
  `tissue_positions.csv`

Optional files in `spatial_dir`, such as tissue images and scalefactors, may
also be present. The workflow will load them, but current plotting uses
coordinate-based scatter plots rather than image overlays.

## Parameter Overview

Global Parameters:

- `project_name`: output folder name under `wt_opts`
- `genome`: reference genome identifier
- `clustering_backend`: choose `scanpy` or `stagate`
- `stagate_embedding_checkpoint`: optional checkpoint to reuse a previously
  trained STAGATE embedding

Preprocessing Parameters:

- `n_top_genes`: number of highly variable features
- `hvg_flavor`: Scanpy HVG method
- `min_genes`, `min_cells`, `min_counts`, `max_counts`, `max_pct_mt`: QC
  filters
- `normalize_target_sum`: optional target sum for
  `scanpy.pp.normalize_total`. If not supplied, Scanpy normalizes to the median
  expression value

Iterative Parameters:

- `resolution`
- `n_comps` (Scanpy backend only)
- `n_neighbors`
- `min_dist`
- `spread`

Advanced Options:

- `apply_harmony`: optional batch correction for multi-sample Scanpy runs
- `merge_small_clusters`: merge undersized clusters after Leiden
- `stagate_k_cutoff`: KNN graph size used when training STAGATE
- `pt_size`, `qc_pt_size`: optional spatial plot size overrides

## Backend Behavior

`scanpy` backend:

- runs PCA inside each mapped parameter-set task
- optionally applies Harmony for multi-sample runs
- iterates over `resolution x n_comps x n_neighbors`

`stagate` backend:

- trains STAGATE once, optionally on GPU
- reuses the learned embedding across mapped parameter-set tasks
- iterates over `resolution x n_neighbors`
- ignores `n_comps` and `apply_harmony`

If `stagate_embedding_checkpoint` is provided, the workflow validates it
against the current preprocessing settings and skips retraining if it matches.

## Outputs

Results are written to `latch:///wt_opts/<project_name>` and include:

- `metadata.csv`
- `medians.csv`
- `spatial_coherence.csv` when spatial coherence can be computed
- `figures/` with UMAP, spatial clustering, and spatial QC plots
- one subdirectory per successful parameter set containing `combined.h5ad`
- `intermediates/` containing the preprocessed AnnData and, for STAGATE runs,
  the STAGATE embedding checkpoint

## Running The Workflow

1. Open `optimize_wt` in the Latch Workflows module.
2. Add one or more Runs with a STAR/STARsolo `gex_dir` and matching
   `spatial_dir`.
3. Choose a `clustering_backend`:
   - use `scanpy` for PCA/Harmony-based optimization
   - use `stagate` for spatial graph neural-network embedding
4. Set QC and HVG parameters in `Preprocessing Parameters`.
5. Set the parameter sweep in `Iterative Parameters`.
6. Launch the workflow and review the output figures and per-set
   `combined.h5ad` files to choose a preferred parameter set.

## Notes

- For multi-sample runs, sample IDs are preserved from the supplied `run_id`
  values and are used in downstream plotting and batch handling.
- STAGATE runs benefit substantially from GPU availability.
- Reusing a STAGATE checkpoint can make repeat runs much faster when only
  downstream clustering parameters are changing.
