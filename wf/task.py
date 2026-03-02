import copy
import itertools
import logging
import os
import subprocess
from typing import List, Optional

import anndata as ad
import scanpy as sc

from latch import message
from latch.resources.tasks import custom_task
from latch.types import LatchDir

import wf.plotting as pl
import wf.preprocessing as pp
import wf.utils as utils

RANDOM_STATE = 42

logging.basicConfig(
    format="%(levelname)s - %(asctime)s - %(message)s",
    level=logging.INFO
)


@custom_task(cpu=8, memory=384, storage_gib=1000)
def wtOpt_task(
    runs: List[utils.Run],
    genome: utils.Genome,
    project_name: str,
    resolution: List[float] = [1.0],
    n_comps: List[int] = [30],
    n_neighbors: List[int] = [15],
    min_dist: List[float] = [0.5],
    spread: List[float] = [1.0],
    min_genes: int = 0,
    min_cells: int = 0,
    min_counts: int = 0,
    max_counts: int = 0,
    max_pct_mt: float = 100.0,
    pt_size: Optional[float] = None,
    qc_pt_size: Optional[float] = None
) -> LatchDir:
    import pandas as pd

    if min_genes == 0:
        warning = "Minimum genes set to 0"
        logging.warning(warning)
        message(typ="warning", data={"title": warning, "body": warning})

    if min_cells == 0:
        warning = "Minimum cells set to 0"
        logging.warning(warning)
        message(typ="warning", data={"title": warning, "body": warning})

    samples = [run.run_id for run in runs]

    # Get channels for specifying plot point size
    channels = max({utils.get_channels(run) for run in runs})

    # Set 'groups' list for differential analysis
    groups = utils.get_groups(runs)
    logging.info(f"Comparing features amoung groups {groups}.")

    qc_metrics = ["n_genes_by_counts", "total_counts", "pct_counts_mt"]

    genome = genome.value  # Convert to str

    out_dir = f"/root/{project_name}"
    os.makedirs(out_dir, exist_ok=True)

    for threshold in [min_cells, min_genes, min_counts]:
        if threshold == 0:
            logging.warning("Minimum fragments set to 0.")

    # Save input parameters to csv
    pd.DataFrame([locals()]).to_csv(f"{out_dir}/metadata.csv", index=False)

    # Build sets of parameters --
    sets = list(itertools.product(
        resolution, n_comps, n_neighbors, min_dist, spread
    ))
    logging.info(f"Iterating through paramter sets {sets}...")

    # Create AnnData objects --------------------------------------------------
    logging.info("Creating AnnData objects...")
    adatas = pp.make_anndatas(runs, genome)

    if len(samples) > 1:
        logging.info("Combining objects...")
        adata = ad.concat(adatas, label="sample")
    else:
        adata = adatas[0]

    # Add addtional QCs
    pp.calculate_qc(adata, genome)

    sc.settings.file_format_figs = "png"
    sc.pl.violin(
        adata,
        ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
        jitter=False,
        stripplot=False,
        multi_panel=True,
        save="_preFiltering"
    )

    adata = pp.filter_adata(
        adata,
        min_cells=min_cells,
        min_genes=min_genes,
        min_counts=min_counts,
        max_counts=max_counts,
        max_pct_mt=max_pct_mt
    )

    sc.pl.violin(
        adata,
        ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
        jitter=False,
        stripplot=False,
        multi_panel=True,
        save="_postFiltering"
    )

    adata = pp.add_spatial(adata)  # Add spatial coordinates to tixels

    # Normalize and scale
    adata.layers["counts"] = adata.X.copy()  # Save counts

    sc.pp.normalize_total(adata)
    adata.layers["normalized"] = adata.X.copy()

    sc.pp.log1p(adata)
    adata.layers["log1p"] = adata.X.copy()

    sc.pp.highly_variable_genes(
        adata, n_top_genes=2000, flavor="seurat", batch_key="sample"
    )

    # Perform scaling as in Seurat
    sc.pp.scale(adata, zero_center=True, max_value=10)

    # Iterate through parameter sets ------------------------------------------
    adata_dict = {}
    count = 1
    for set in sets:
        try:
            cr, nc, nn, md, sp = set
            logging.info(
                f"Set {count}: clustering resolution {cr}, number of "
                f"components {nc}, neighborhood size {nn}, umap minimum "
                f"{md}, umap spread {sp}."
            )
            cr_str, md_str, sp_str = [
                str(param).replace(".", "-") for param in [cr, md, sp]
            ]
            set_str = f"set{count}_cr{cr_str}-nc{nc}-nn{nn}-md{md_str}-sp{sp_str}"
            set_dir = f"{out_dir}/{set_str}"
            os.makedirs(set_dir, exist_ok=True)

            adata_i = copy.deepcopy(adata)

            logging.info(
                f"Performing dimensionality reduction with resolution {cr}, \
            number of components {nc}, neighborhood size {nn}..."
            )
            adata_i = pp.add_clusters(
                adata_i,
                cr,
                nc,
                nn,
                md,
                sp,
                random_state=RANDOM_STATE
            )

            adata_dict[set_str] = adata_i
            adata_i.write(f"{set_dir}/combined.h5ad")

        except Exception as e:
            logging.warning(f"Exception for set {count}: {e}")
            message(
                typ="warning",
                data={
                    "title": "failed set",
                    "body": f"set {count} failed with Exception '{e}'"
                }
            )

        count += 1

    # Save figures ------------------------------------------------------------
    figures_dir = f"{out_dir}/figures"
    os.makedirs(figures_dir, exist_ok=True)

    pl.combine_umaps(adata_dict, f"{figures_dir}/all_umaps.png")

    pt_size = (
        pt_size if pt_size is not None
        else utils.pt_sizes[channels]["dim"]
    )
    pl.combine_spatials(
        adata_dict,
        samples,
        f"{figures_dir}/all_spatialdim.png",
        pt_size=pt_size
    )

    qc_pt_size = (
        qc_pt_size if qc_pt_size is not None
        else utils.pt_sizes[channels]["qc"]
    )
    pl.plot_spatial_qc(
        adata,
        samples,
        qc_metrics,
        f"{figures_dir}/spatial_qc.png",
        pt_size=qc_pt_size
    )

    # Medians -----------------------------------------------------------------

    # Calculate the medians for each sample, create a DataFrame
    grouped = adata.obs.groupby("sample")
    medians_df = grouped.agg({
        "total_counts": "median",
        "n_genes_by_counts": "median",
        "pct_counts_mt": "median"
    }).reset_index()

    # Rename columns
    medians_df.rename(
        columns={
            "sample": "run_id",
            "total_counts": "umi counts",
            "n_genes_by_counts": "gene counts",
            "pct_counts_mt": "percent mitochondrial"
        }, inplace=True
    )

    medians_df.to_csv(f"{out_dir}/medians.csv", index=False)

    # Upload data -------------------------------------------------------------

    # Move scanpy plots
    subprocess.run([f"mv /root/figures/* {figures_dir}"], shell=True)

    return LatchDir(out_dir, f"latch:///wt_opts/{project_name}")
