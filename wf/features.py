import logging
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

    keep_obsm = {"spatial", "X_umap"}
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
