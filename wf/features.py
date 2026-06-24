import logging
import math
from pathlib import Path
from typing import List, Optional, Tuple

import anndata as ad
import numpy as np
import scipy.sparse as sp


logging.basicConfig(
    format="%(levelname)s - %(asctime)s - %(message)s",
    level=logging.INFO,
)


QC_OBS_COLUMNS = [
    "barcode",
    "on_off",
    "row",
    "col",
    "xcor",
    "ycor",
    "n_genes_by_counts",
    "log1p_n_genes_by_counts",
    "total_counts",
    "log1p_total_counts",
    "pct_counts_in_top_50_genes",
    "pct_counts_in_top_100_genes",
    "pct_counts_in_top_200_genes",
    "pct_counts_in_top_500_genes",
    "total_counts_mt",
    "log1p_total_counts_mt",
    "pct_counts_mt",
]

QC_VAR_COLUMNS = [
    "mt",
    "ribo",
    "lncrna",
    "mirna",
    "sncrna",
    "n_counts",
    "n_cells",
    "n_cells_by_counts",
    "mean_counts",
    "log1p_mean_counts",
    "pct_dropout_by_counts",
    "total_counts",
    "log1p_total_counts",
    "means",
    "dispersions",
    "dispersions_norm",
]


def _drop_columns(adata: ad.AnnData, axis: str, columns: List[str]) -> None:
    frame = adata.obs if axis == "obs" else adata.var
    to_drop = [column for column in columns if column in frame.columns]
    if len(to_drop) > 0:
        frame.drop(to_drop, axis=1, inplace=True)


def _plotting_x_matrix(
    adata: ad.AnnData,
    x_priority: List[str],
) -> Tuple[object, str]:
    for source in x_priority:
        if source == "X":
            return adata.X.copy(), "X"
        if source in adata.layers:
            return adata.layers[source].copy(), f"layers/{source}"

    return adata.X.copy(), "X"


def add_spatial_offset(
    adata: ad.AnnData,
    sample_key: str = "sample",
    spatial_key: str = "spatial",
    new_obsm_key: str = "spatial_offset",
    tile_spacing: float = 300.0,
) -> None:
    if (
        sample_key not in adata.obs
        or spatial_key not in adata.obsm
        or new_obsm_key in adata.obsm
    ):
        return

    sample_values = adata.obs[sample_key].astype(str)
    samples = sorted(sample_values.unique().tolist())
    n_samples = len(samples)
    if n_samples == 0:
        return

    n_cols = min(2, max(1, n_samples))
    n_rows = math.ceil(n_samples / n_cols)
    spatial = np.asarray(adata.obsm[spatial_key])
    offset = np.empty_like(spatial, dtype=float)
    grid_bounds = {}
    sample_positions = {}

    for idx, sample_name in enumerate(samples):
        row = idx // n_cols
        col = idx % n_cols
        sample_positions[sample_name] = (row, col)

        mask = (sample_values == sample_name).to_numpy()
        sample_spatial = spatial[mask]
        if sample_spatial.size == 0:
            continue

        max_coords = sample_spatial.max(axis=0)
        min_coords = sample_spatial.min(axis=0)
        grid_bounds[(row, col)] = {
            "width": float(max_coords[0] - min_coords[0]),
            "height": float(max_coords[1] - min_coords[1]),
            "min_x": float(min_coords[0]),
            "max_y": float(max_coords[1]),
        }

    row_heights = [
        max(
            (
                grid_bounds[(row, col)]["height"]
                for col in range(n_cols)
                if (row, col) in grid_bounds
            ),
            default=0.0,
        )
        for row in range(n_rows)
    ]
    col_widths = [
        max(
            (
                grid_bounds[(row, col)]["width"]
                for row in range(n_rows)
                if (row, col) in grid_bounds
            ),
            default=0.0,
        )
        for col in range(n_cols)
    ]

    row_y_offsets = [0.0]
    for idx in range(n_rows - 1):
        row_y_offsets.append(row_y_offsets[-1] - row_heights[idx] - tile_spacing)

    col_x_offsets = [0.0]
    for idx in range(n_cols - 1):
        col_x_offsets.append(col_x_offsets[-1] + col_widths[idx] + tile_spacing)

    for sample_name in samples:
        row, col = sample_positions[sample_name]
        bounds = grid_bounds.get((row, col))
        if bounds is None:
            continue

        mask = (sample_values == sample_name).to_numpy()
        sample_spatial = spatial[mask].copy().astype(float)
        sample_spatial[:, 0] += col_x_offsets[col] - bounds["min_x"]
        sample_spatial[:, 1] += row_y_offsets[row] - bounds["max_y"]
        offset[mask] = sample_spatial

    adata.obsm[new_obsm_key] = offset


def make_small_anndata(
    adata: ad.AnnData,
    matrix_dtype=np.float16,
    force_dense: bool = True,
    x_priority: Optional[List[str]] = None,
) -> ad.AnnData:
    """Return a reduced AnnData object for plotting and notebook use."""
    out = adata.copy()

    out.X, x_source = _plotting_x_matrix(
        out,
        x_priority or ["log1p", "normalized", "X"],
    )
    out.uns["plotting_x_source"] = x_source

    _drop_columns(out, "obs", QC_OBS_COLUMNS)
    _drop_columns(out, "var", QC_VAR_COLUMNS)

    out.varm.clear()
    out.layers.clear()
    out.raw = None

    for key in ["pca", "log1p", "neighbors"]:
        out.uns.pop(key, None)

    add_spatial_offset(out)

    keep_obsm = {"spatial_offset", "X_umap"}
    for key in list(out.obsm.keys()):
        if key not in keep_obsm:
            del out.obsm[key]
    out.obsp.clear()

    if sp.issparse(out.X):
        if force_dense:
            out.X = out.X.toarray().astype(matrix_dtype)
        else:
            out.X = out.X.astype(matrix_dtype)
    else:
        out.X = np.asarray(out.X, dtype=matrix_dtype)

    return out


def save_anndata_objects(
    adata: ad.AnnData,
    base_dir: Path,
    stem: str = "combined",
) -> None:
    """Save full and reduced AnnData objects using the shared ATX pattern."""
    base_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Saving full AnnData object.")
    adata.write(base_dir / f"{stem}.h5ad")

    logging.info("Creating reduced AnnData object.")
    sm_adata = make_small_anndata(adata)

    logging.info("Saving reduced AnnData object.")
    sm_adata.write(base_dir / f"{stem}_sm.h5ad")
