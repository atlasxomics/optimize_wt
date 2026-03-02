import anndata
import scanpy as sc
import logging

from typing import List, Union
from pathlib import Path

import pandas as pd

from wf.utils import gene_keys, get_LatchFile, Run


logging.basicConfig(
    format="%(levelname)s - %(asctime)s - %(message)s",
    level=logging.INFO
)


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
    adata.obs["condition"] = run.condition

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

    adatas = [sc.read_10x_mtx(run.gex_dir.local_path) for run in runs]

    position_files = {
        run.run_id: get_LatchFile(run.spatial_dir, "tissue_positions_list.csv")
        for run in runs
    }

    adatas = [add_metadata(run, adata, position_files[run.run_id].local_path)
              for run, adata in zip(runs, adatas)]

    return adatas
