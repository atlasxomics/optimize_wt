import anndata
import scanpy as sc
import logging
import numpy as np
import squidpy as sq

from typing import List, Union
from pathlib import Path

import pandas as pd
from scipy import io as sio
from scipy import sparse as sp

from wf.utils import gene_keys, get_LatchFile, Run, sanitize_condition


logging.basicConfig(
    format="%(levelname)s - %(asctime)s - %(message)s",
    level=logging.INFO
)


MTX_CANDIDATES = [
    "UniqueAndMult-EM.mtx",
    "UniqueAndMult-EM.mtx.gz",
    "matrix.mtx",
    "matrix.mtx.gz",
]
GENE_TABLE_CANDIDATES = [
    "features.tsv.gz",
    "features.tsv",
    "genes.tsv",
    "genes.tsv.gz",
]
BARCODE_CANDIDATES = [
    "barcodes.tsv",
    "barcodes.tsv.gz",
]
POSITION_CANDIDATES = [
    "tissue_positions_list.csv",
    "tissue_positions.csv",
]


def _resolve_local_file(directory: Union[Path, str], candidates: List[str]) -> Path:
    directory = Path(directory)
    for candidate in candidates:
        candidate_path = directory / candidate
        if candidate_path.exists():
            return candidate_path

    raise FileNotFoundError(
        f"Could not find any of {candidates} in '{directory}'."
    )


def _resolve_spatial_file(run: Run, candidates: List[str]) -> Path:
    spatial_dir = Path(run.spatial_dir.local_path)

    try:
        return _resolve_local_file(spatial_dir, candidates)
    except FileNotFoundError:
        pass

    for candidate in candidates:
        latch_file = get_LatchFile(run.spatial_dir, candidate)
        if latch_file is not None:
            return Path(latch_file.local_path)

    raise FileNotFoundError(
        f"Unable to resolve any of {candidates} for run '{run.run_id}' in "
        f"'{run.spatial_dir.remote_path}'."
    )


def _deduplicate_gene_names(gene_names: pd.Series) -> List[str]:
    seen: dict[str, int] = {}
    unique_names: List[str] = []

    for gene_name in gene_names.astype(str):
        count = seen.get(gene_name, 0) + 1
        seen[gene_name] = count
        unique_names.append(
            f"{gene_name}.{count}" if count > 1 else gene_name
        )

    return unique_names


def _read_gene_table(gex_dir: Union[Path, str]) -> pd.DataFrame:
    genes_path = _resolve_local_file(gex_dir, GENE_TABLE_CANDIDATES)
    genes = pd.read_csv(genes_path, sep="\t", header=None)

    if genes.shape[1] == 0:
        raise ValueError(f"Gene table '{genes_path}' is empty.")

    gene_ids = genes.iloc[:, 0].astype(str)
    gene_names = genes.iloc[:, 1] if genes.shape[1] >= 2 else gene_ids.copy()

    return pd.DataFrame(
        {
            "gene_id": gene_ids.to_numpy(),
            "gene_name": _deduplicate_gene_names(gene_names),
        }
    )


def _read_barcodes(gex_dir: Union[Path, str]) -> pd.Index:
    barcodes_path = _resolve_local_file(gex_dir, BARCODE_CANDIDATES)
    barcodes = pd.read_csv(barcodes_path, header=None).iloc[:, 0].astype(str)

    return pd.Index(barcodes)


def _read_count_matrix(
    gex_dir: Union[Path, str],
    n_barcodes: int,
    n_genes: int
) -> sp.csr_matrix:
    matrix_path = _resolve_local_file(gex_dir, MTX_CANDIDATES)
    matrix = sp.csr_matrix(sio.mmread(str(matrix_path)))

    if matrix.shape == (n_genes, n_barcodes):
        return matrix.T.tocsr()

    if matrix.shape == (n_barcodes, n_genes):
        return matrix.tocsr()

    raise ValueError(
        f"Count matrix '{matrix_path}' shape {matrix.shape} does not match "
        f"{n_genes} genes and {n_barcodes} barcodes."
    )


def _read_positions(positions_file: Union[Path, str]) -> pd.DataFrame:
    positions = pd.read_csv(positions_file, header=None)
    column_names = ["barcode", "on_off", "row", "col", "xcor", "ycor"]

    if positions.shape[1] < len(column_names):
        raise ValueError(
            f"Spatial positions file '{positions_file}' must have at least "
            f"{len(column_names)} columns."
        )

    first_flag = str(positions.iloc[0, 1]).strip()
    if first_flag in {"0", "1"}:
        positions = positions.iloc[:, :len(column_names)].copy()
        positions.columns = column_names
    else:
        positions.columns = positions.iloc[0]
        positions = positions.iloc[1:].reset_index(drop=True)
        positions = positions.rename(
            columns={
                "in_tissue": "on_off",
                "array_row": "row",
                "array_col": "col",
                "pxl_row_in_fullres": "xcor",
                "pxl_col_in_fullres": "ycor",
            }
        )
        missing = [col for col in column_names if col not in positions.columns]
        if missing:
            raise ValueError(
                f"Spatial positions file '{positions_file}' is missing "
                f"columns {missing}."
            )
        positions = positions[column_names].copy()

    positions["barcode"] = positions["barcode"].astype(str)
    positions["on_off"] = positions["on_off"].astype(int)
    positions["row"] = positions["row"].astype(int)
    positions["col"] = positions["col"].astype(int)
    positions["xcor"] = positions["xcor"].astype(float)
    positions["ycor"] = positions["ycor"].astype(float)

    return positions.set_index("barcode")


def _load_run_adata(run: Run) -> anndata.AnnData:
    gex_dir = Path(run.gex_dir.local_path)
    gene_table = _read_gene_table(gex_dir)
    barcodes = _read_barcodes(gex_dir)
    matrix = _read_count_matrix(gex_dir, n_barcodes=len(barcodes), n_genes=len(gene_table))

    positions_file = _resolve_spatial_file(run, POSITION_CANDIDATES)
    positions = _read_positions(positions_file)
    in_tissue = positions[positions["on_off"] == 1]

    matched_barcodes = [barcode for barcode in in_tissue.index if barcode in barcodes]
    if len(matched_barcodes) == 0:
        raise ValueError(
            f"No in-tissue barcodes from '{positions_file}' matched the count "
            f"matrix for run '{run.run_id}'."
        )

    barcode_to_idx = {barcode: idx for idx, barcode in enumerate(barcodes)}
    row_indices = [barcode_to_idx[barcode] for barcode in matched_barcodes]
    prefixed_barcodes = [f"{run.run_id}#{barcode}" for barcode in matched_barcodes]

    adata = anndata.AnnData(
        X=matrix[row_indices, :],
        obs=pd.DataFrame(
            {
                "barcode": matched_barcodes,
                "on_off": in_tissue.loc[matched_barcodes, "on_off"].to_numpy(),
                "row": in_tissue.loc[matched_barcodes, "row"].to_numpy(),
                "col": in_tissue.loc[matched_barcodes, "col"].to_numpy(),
                "xcor": in_tissue.loc[matched_barcodes, "xcor"].to_numpy(),
                "ycor": in_tissue.loc[matched_barcodes, "ycor"].to_numpy(),
                "sample": run.run_id,
                "condition": sanitize_condition(run.condition),
            },
            index=prefixed_barcodes,
        ),
        var=pd.DataFrame(
            {"gene_id": gene_table["gene_id"].to_numpy()},
            index=pd.Index(gene_table["gene_name"], dtype="object"),
        ),
    )

    return adata


def add_clusters(
    adata: anndata.AnnData,
    resolution: float,
    n_comps: int,
    n_neighbors: int,
    min_dist: float,
    spread: float,
    random_state: int = 0,
    pca_plot: bool = True,
) -> anndata.AnnData:
    """Perform dimensionality reduction, batch correction, umap, clustering.
    """

    # Dimensionality reduction
    sc.tl.pca(
        adata,
        n_comps=n_comps,
        use_highly_variable=True,
        random_state=random_state
    )
    if pca_plot:
        sc.pl.pca_variance_ratio(adata, n_pcs=n_comps, save=f"_{n_comps}_elbow")

    n_runs = 1
    try:
        n_runs = len(adata.obs["sample"].unique())
    except KeyError as e:
        logging.warning(
            f"Exception {e}: Please add metadata to combined AnnData."
        )

    if n_runs > 1:
        logging.info("Performing batch correction with Harmony...")
        sc.external.pp.harmony_integrate(adata, batch="sample")
        rep = "X_pca_harmony"
    else:
        rep = "X_pca"

    sc.pp.neighbors(
        adata,
        n_neighbors=n_neighbors,
        use_rep=rep,
        n_pcs=n_comps,
        random_state=random_state
    )

    # Add umap, nearest neightbors, clusters to .obs
    sc.tl.umap(
        adata,
        min_dist=min_dist,
        spread=spread,
        random_state=random_state
    )
    sc.tl.leiden(
        adata,
        resolution=resolution,
        key_added="cluster",
        random_state=random_state
    )

    return adata


def add_metadata(
    run: Run, adata: anndata.AnnData, positions_file: Union[Path, str]
) -> anndata.AnnData:
    """Add metadata and spatial info .obs of AnnData.
    """

    # Read in tissue_positions file from spatial/
    positions = pd.read_csv(positions_file, header=None)
    positions.columns = ["barcode", "on_off", "row", "col", "xcor", "ycor"]

    # Merge fragments file with Anndata.obs
    adata.obs["barcode"] = adata.obs.index
    adata.obs = adata.obs.merge(positions, on="barcode", how="left")

    # Set run_id, condition
    adata.obs["sample"] = run.run_id
    adata.obs["condition"] = sanitize_condition(run.condition)

    # Ensure obs_names unique
    adata.obs_names = [
        run_id + "#" + bc for
        run_id, bc in zip(adata.obs["sample"], adata.obs["barcode"])
    ]

    return adata


def add_spatial(
    adata: anndata.AnnData, x_key: str = "xcor", y_key: str = "ycor"
) -> anndata.AnnData:
    """Add move x and y coordinates from .obs to .obsm["spatial"] for squidpy.
    """
    adata.obsm["spatial"] = adata.obs[[y_key, x_key]].values

    return adata


def add_spatial_neighbors(
    adata: anndata.AnnData,
    library_key: str = "sample"
) -> anndata.AnnData:
    """Build a within-sample spatial neighbor graph for coherence scoring.

    Prefer a grid graph when the input resembles Visium-style array positions.
    Fall back to generic spatial coordinates if grid graph construction fails.
    """

    try:
        sq.gr.spatial_neighbors(
            adata,
            coord_type="grid",
            n_rings=1,
            library_key=library_key,
        )
    except Exception as grid_error:
        logging.warning(
            "Failed to build grid spatial neighbors (%s). Falling back to "
            "generic spatial coordinates.",
            grid_error,
        )
        sq.gr.spatial_neighbors(
            adata,
            coord_type="generic",
            library_key=library_key,
        )

    return adata


def morans_i(
    connectivities: Union[sp.spmatrix, np.ndarray],
    values: np.ndarray
) -> float:
    """Compute Moran's I from a connectivity matrix and numeric vector."""

    if sp.issparse(connectivities):
        weights = connectivities.tocsr().astype(float)
    else:
        weights = sp.csr_matrix(np.asarray(connectivities, dtype=float))

    row_sums = np.asarray(weights.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1.0
    weights = sp.diags(1.0 / row_sums) @ weights

    centered = np.asarray(values, dtype=float) - np.mean(values)
    denom = float(centered @ centered)
    if denom == 0.0:
        return 0.0

    weighted_centered = weights @ centered

    return float(centered @ weighted_centered / denom)


def spatial_coherence_table(adata_dict: dict[str, anndata.AnnData]) -> pd.DataFrame:
    """Summarize Moran's I spatial coherence for each clustered AnnData."""

    rows = []
    for set_name, adata in adata_dict.items():
        if "spatial_connectivities" not in adata.obsp:
            raise KeyError(
                f"AnnData '{set_name}' is missing 'spatial_connectivities'."
            )
        if "cluster" not in adata.obs:
            raise KeyError(f"AnnData '{set_name}' is missing 'cluster' labels.")

        cluster_codes = (
            adata.obs["cluster"].astype("category").cat.codes.to_numpy(dtype=float)
        )
        rows.append(
            {
                "set": set_name,
                "n_clusters": int(adata.obs["cluster"].nunique()),
                "morans_I": round(
                    morans_i(adata.obsp["spatial_connectivities"], cluster_codes),
                    4,
                ),
            }
        )

    return pd.DataFrame(rows)


def calculate_qc(adata: anndata.AnnData, genome: str) -> None:

    mito_key = gene_keys["mitochondiral"][genome]
    ribo_key = gene_keys["ribosomal"][genome]
    adata.var["mt"] = adata.var_names.str.startswith(mito_key)
    adata.var["ribo"] = adata.var_names.str.startswith(ribo_key)

    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt", "ribo"], inplace=True, log1p=True
    )

    return None


def filter_adata(
    adata: anndata.AnnData,
    min_cells: int = 1,
    min_genes: int = 1,
    min_counts: int = 0,
    max_counts: int = 0,
    max_pct_mt: float = 100.0
) -> anndata.AnnData:
    """Filter AnnData by tissue status and basic QC thresholds.

    `max_counts == 0` disables high-count filtering.
    `max_pct_mt >= 100` disables mitochondrial filtering.
    """

    n_before = adata.n_obs

    # Filter 'off tissue' tixels
    try:
        adata = adata[adata.obs["on_off"] == 1].copy()
    except KeyError as e:
        logging.warning(
            f"Exception {e}: no positions data found in AnnData.obs"
        )

    sc.pp.filter_cells(adata, min_genes=min_genes)
    if min_counts > 0:
        sc.pp.filter_cells(adata, min_counts=min_counts)

    if max_counts > 0:
        adata = adata[adata.obs["total_counts"] <= max_counts].copy()

    if max_pct_mt < 100:
        if "pct_counts_mt" in adata.obs:
            adata = adata[adata.obs["pct_counts_mt"] <= max_pct_mt].copy()
        else:
            logging.warning(
                "pct_counts_mt not found in AnnData.obs; skipping mt filter."
            )

    sc.pp.filter_genes(adata, min_cells=min_cells)
    logging.info(
        f"Filtered observations from {n_before} to {adata.n_obs} using "
        f"min_genes={min_genes}, min_counts={min_counts}, "
        f"max_counts={max_counts}, max_pct_mt={max_pct_mt}."
    )

    return adata


def make_anndatas(runs: List[Run], genome: str) -> List[anndata.AnnData]:
    """Basic preprocessing for scanpy analysis; converts raw/ gex dir into a
    list of AnnData objects. QCs, metadata and spatial data are stored in
    AnnData.obs.
    """
    return [_load_run_adata(run) for run in runs]
