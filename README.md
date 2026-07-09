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
5. Iterates over clustering parameter sets in parallel and writes
   `combined.h5ad` plus a reduced `combined_sm.h5ad` per successful set.
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

- `apply_harmony`: optional batch correction for multi-sample runs
- `merge_small_clusters`: merge undersized clusters after Leiden
- `compute_cluster_markers`: rank marker genes for each cluster in each
  parameter set
- `marker_top_n`: number of top marker genes per cluster to include in marker
  summaries and heatmaps
- `stagate_k_cutoff`: KNN graph size used when training STAGATE
- `pt_size`, `qc_pt_size`: optional spatial plot size overrides

## Backend Behavior

`scanpy` backend:

- runs PCA inside each mapped parameter-set task
- optionally applies Harmony for multi-sample runs
- iterates over `resolution x n_comps x n_neighbors`

`stagate` backend:

- trains STAGATE once, optionally on GPU
- optionally applies Harmony to the STAGATE embedding for multi-sample runs
- reuses the learned embedding across mapped parameter-set tasks
- iterates over `resolution x n_neighbors`
- ignores `n_comps`

## Outputs

Results are written to `latch:///wt_opts/<project_name>` and include:

Top-level summary files:

- `metadata.csv`: one-row manifest of the workflow inputs and parameter values,
  including run IDs, conditions, filtering thresholds, backend choice, and the
  optimization grid.
- `medians.csv`: per-run median QC metrics after filtering, including UMI
  counts, detected gene counts, and percent mitochondrial reads.
- `spatial_coherence.csv`: per-parameter-set spatial coherence scores when a
  spatial neighbor graph can be built.
- `svg_genes.csv`: spatially variable gene statistics when spatial
  autocorrelation completes successfully.

Top-level interactive plots:

- `all_umaps.html`: UMAP panels for each successful parameter set, colored by
  cluster and, when applicable, sample and condition.
- `all_spatialdim.html`: spatial cluster plots for each successful parameter
  set and sample.
- `spatial_qc.html`: spatial plots of QC metrics such as total counts, detected
  genes, and mitochondrial percentage.
- `svg_spatial.html`: spatial expression plots for the top spatially variable
  genes when SVG analysis completes successfully.

Static figures are written under `figures/` and mirror the interactive HTML
outputs where possible. This directory can include UMAP summaries, spatial
cluster summaries, spatial QC plots, spatial coherence plots, and spatially
variable gene plots.

Each successful parameter set also gets its own subdirectory named with the
backend and parameter values, for example
`set1_backend-scanpy_cr1-0-nc30-nn15-md0-5-sp1-0`. Each set directory
contains:

- `combined.h5ad`: full AnnData object with the clustered cells/spots,
  embeddings, metadata, layers, and clustering results for that parameter set.
- `combined_sm.h5ad`: reduced AnnData object for lightweight review and launch
  plotting. It keeps the UMAP, spatial coordinates, cluster/sample metadata,
  and a compact expression matrix while dropping large QC and intermediate
  fields.
- `Launch_Plots/artifact.json`: Latch plot artifact metadata that points to the
  reduced AnnData object.
- `deg_clusters.csv`: optional cluster marker table when
  `compute_cluster_markers` is enabled and marker ranking succeeds.
- `deg_clusters_top<marker_top_n>.csv`: optional compact marker table with the
  top marker genes per cluster.
- `figures/cluster_marker_heatmap_top<marker_top_n>.png`: optional marker-gene
  heatmap.
- `figures/deg_heatmap_top<marker_top_n>_compact_hires.pdf`: optional
  high-resolution marker-gene heatmap.

Intermediate task outputs are stored under `_intermediates/`:

- `_intermediates/preprocess/preprocessed.h5ad`: filtered, normalized,
  log-transformed, HVG-selected AnnData object used as input to optimization
  jobs.
- `_intermediates/preprocess/figures/`: pre- and post-filtering QC violin plots
  from preprocessing.
- `_intermediates/stagate_preprocess/preprocessed.h5ad`: STAGATE-embedded
  preprocessed object when `clustering_backend="stagate"`.

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
   `combined.h5ad` or `combined_sm.h5ad` files to choose a preferred parameter
   set.

## Notes

- For multi-sample runs, sample IDs are preserved from the supplied `run_id`
  values and are used in downstream plotting and batch handling.
- STAGATE runs benefit substantially from GPU availability.
