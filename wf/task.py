import itertools
import logging
import os
from typing import List, Optional

import anndata as ad
import scanpy as sc

from latch import message
from latch.resources.tasks import custom_task
from latch.types import LatchDir

import wf.plotting as pl
import wf.preprocessing as pp
import wf.utils as utils

logging.basicConfig(
    format="%(levelname)s - %(asctime)s - %(message)s",
    level=logging.INFO
)


@custom_task(cpu=62, memory=576, storage_gib=4949)
def wtOpt_task(
    runs: List[utils.Run],
    genome: utils.Genome,
    project_name: str,
    resolution: List[float] = [1.0],
    n_comps: List[int] = [30],
    min_genes: int = 0,
    min_cells: int = 0,
    pt_size: Optional[float] = None,
    qc_pt_size: Optional[float] = None
) -> LatchDir:
    import pandas as pd

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

    for threshold in [min_cells, min_genes]:
        if threshold == 0:
            logging.warning("Minimum fragments set to 0.")

    # Save input parameters to csv
    pd.DataFrame([locals()]).to_csv("metadata.csv", index=False)

    # Build sets of parameters --
    sets = list(itertools.product(resolution, n_comps))
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
    adata = pp.filter_adata(adata, min_cells=min_cells, min_genes=min_genes)

    # Iterate through parameter sets ------------------------------------------
    adata_dict = {}
    count = 1
    for set in sets:
        try:
            cr, nc = set
            logging.info(f"Set {count}: clustering resolution {cr}, number of components {nc}")
            cr_str = str(cr).replace(".", "-")
            set_str = f"set{count}_cr{cr_str}-nc{nc}"
            set_dir = f"{out_dir}/{set_str}"
            os.makedirs(set_dir, exist_ok=True)

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

        #     logging.info(
        #         f"Performing dimensionality reduction with resolution {cr}, \
        #         number of components {nc}..."
        #     )
        #     adata = pp.add_clusters(adata, cr, nc)

        #     adata = pp.add_spatial(adata)  # Add spatial coordinates to tixels

        #     adata_dict[set_str] = adata
            adata.write(f"{set_dir}/combined.h5ad")

        except Exception as e:
            logging.warning(f"Exception for set {count}: {e}")
            message(
                typ="warning",
                data={
                    "title": "failed set",
                    "body": f"set {count} with clustering resolution {cr}, \
                        number of components {nc} failed with Exception '{e}'"
                }
            )

        count += 1

    # Save figures ------------------------------------------------------------
    figures_dir = f"{out_dir}/figures"
    os.makedirs(figures_dir, exist_ok=True)

    # pl.combine_umaps(adata_dict, f"{figures_dir}/all_umaps.pdf")

    # pt_size = (
    #     pt_size if pt_size is not None
    #     else utils.pt_sizes[channels]["dim"]
    # )
    # pl.combine_spatials(
    #     adata_dict,
    #     samples,
    #     f"{figures_dir}/all_spatialdim.pdf",
    #     pt_size=pt_size
    # )

    # qc_pt_size = (
    #     qc_pt_size if qc_pt_size is not None
    #     else utils.pt_sizes[channels]["qc"]
    # )
    # pl.plot_spatial_qc(
    #     adata,
    #     samples,
    #     qc_metrics,
    #     f"{figures_dir}/spatial_qc.pdf",
    #     pt_size=qc_pt_size
    # )
    # Violin Plots
    # Elbow plot

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
    return LatchDir(out_dir, f"latch:///snap_opts/{project_name}")
