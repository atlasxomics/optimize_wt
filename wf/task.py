import itertools
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import anndata as ad
import pandas as pd
import scanpy as sc

from latch import message
from latch.resources.tasks import custom_task
from latch.types import LatchDir, LatchFile

try:
    from latch.resources.tasks import g6e_2xlarge_task as stagate_gpu_task
except ImportError:
    from latch.resources.tasks import large_gpu_task as stagate_gpu_task

import wf.plotting as pl
import wf.preprocessing as pp
import wf.utils as utils

RANDOM_STATE = 42

logging.basicConfig(
    format="%(levelname)s - %(asctime)s - %(message)s",
    level=logging.INFO,
)


def _write_metadata_csv(output_path: Path, metadata: Dict[str, object]) -> None:
    pd.DataFrame([metadata]).to_csv(output_path, index=False)


def _write_cluster_marker_outputs(
    adata: ad.AnnData,
    out_dir: Path,
    marker_top_n: int,
) -> None:
    if marker_top_n < 1:
        raise ValueError("marker_top_n must be at least 1.")
    if "cluster" not in adata.obs:
        raise KeyError("Cannot calculate cluster markers: missing obs['cluster'].")
    if "log1p" not in adata.layers:
        raise KeyError(
            "Cannot calculate cluster markers: missing adata.layers['log1p']."
        )

    n_clusters = int(adata.obs["cluster"].nunique())
    if n_clusters < 2:
        logging.warning(
            "Skipping cluster marker calculation because only %d cluster is present.",
            n_clusters,
        )
        return

    genes = adata.var_names.astype(str)
    genes_upper = genes.str.upper()
    keep_genes = ~(
        genes_upper.str.startswith("MT-")
        | genes_upper.str.startswith("RPS")
        | genes_upper.str.startswith("RPL")
        | genes_upper.str.startswith("MTRNR")
    )
    marker_adata = adata[:, keep_genes].copy()
    marker_adata.X = adata.layers["log1p"][:, keep_genes].copy()
    marker_adata.obs["cluster"] = marker_adata.obs["cluster"].astype(str)
    clusters = sorted(
        marker_adata.obs["cluster"].unique(),
        key=lambda cluster: (
            0,
            int(cluster),
        ) if str(cluster).isdigit() else (1, str(cluster)),
    )

    sc.tl.rank_genes_groups(
        marker_adata,
        groupby="cluster",
        method="wilcoxon",
        use_raw=False,
        pts=True,
        key_added="cluster_markers",
    )
    deg_frames = []
    top_frames = []
    top_genes_per_cluster: Dict[str, List[str]] = {}
    for cluster in clusters:
        deg_df = sc.get.rank_genes_groups_df(
            marker_adata,
            group=cluster,
            key="cluster_markers",
            pval_cutoff=0.05,
            log2fc_min=0.25,
        )
        deg_df.insert(0, "cluster", cluster)
        deg_frames.append(deg_df)

        top_df = sc.get.rank_genes_groups_df(
            marker_adata,
            group=cluster,
            key="cluster_markers",
            pval_cutoff=0.05,
        ).head(marker_top_n)
        top_df.insert(0, "cluster", cluster)
        top_frames.append(top_df)
        top_genes_per_cluster[cluster] = top_df["names"].astype(str).tolist()

    if len(deg_frames) == 0:
        raise ValueError("No clusters were available for DEG output.")

    markers_df = pd.concat(deg_frames, ignore_index=True)
    markers_df.to_csv(out_dir / "deg_clusters.csv", index=False)
    top_markers_df = pd.concat(top_frames, ignore_index=True)
    top_markers_df.to_csv(
        out_dir / f"deg_clusters_top{marker_top_n}.csv",
        index=False,
    )
    adata.uns["cluster_marker_degs"] = markers_df
    adata.uns["cluster_marker_degs_params"] = {
        "groupby": "cluster",
        "method": "wilcoxon",
        "expression_layer": "log1p",
        "pval_cutoff": 0.05,
        "log2fc_min": 0.25,
        "excluded_prefixes": ["MT-", "RPS", "RPL", "MTRNR"],
        "included_gene_count": int(keep_genes.sum()),
        "excluded_gene_count": int((~keep_genes).sum()),
    }

    figures_dir = out_dir / "figures"
    os.makedirs(figures_dir, exist_ok=True)
    marker_heatmap = pl.plot_marker_heatmap(
        marker_adata,
        top_genes_per_cluster,
        str(figures_dir / f"cluster_marker_heatmap_top{marker_top_n}.png"),
        marker_top_n=marker_top_n,
    )
    pl.plot_marker_heatmap(
        marker_adata,
        top_genes_per_cluster,
        str(figures_dir / f"deg_heatmap_top{marker_top_n}_compact_hires.pdf"),
        marker_top_n=marker_top_n,
    )
    adata.uns["cluster_marker_heatmap"] = marker_heatmap
    adata.uns["cluster_marker_heatmap_params"] = {
        "included_gene_count": int(keep_genes.sum()),
        "excluded_gene_count": int((~keep_genes).sum()),
        "excluded_prefixes": ["MT-", "RPS", "RPL", "MTRNR"],
        "expression_layer": "log1p",
        "pval_cutoff": 0.05,
        "log2fc_min": 0.25,
        "marker_top_n": marker_top_n,
        "values": "column-wise z-score of mean log1p expression, clipped to [-3, 3]",
    }


@custom_task(cpu=4, memory=128, storage_gib=1000)
def preprocess_wt_task(
    runs: List[utils.Run],
    genome: utils.Genome,
    project_name: str,
    n_top_genes: int = 4000,
    hvg_flavor: str = "seurat",
    min_genes: int = 0,
    min_cells: int = 0,
    min_counts: int = 0,
    max_counts: int = 0,
    max_pct_mt: float = 100.0,
    normalize_target_sum: Optional[float] = None,
    normalization_mode: str = "total",
    gene_lengths_file: Optional[LatchFile] = None,
) -> LatchDir:
    if min_genes == 0:
        warning = "Minimum genes set to 0"
        logging.warning(warning)
        message(typ="warning", data={"title": warning, "body": warning})

    if min_cells == 0:
        warning = "Minimum cells set to 0"
        logging.warning(warning)
        message(typ="warning", data={"title": warning, "body": warning})

    genome_str = genome.value
    if hvg_flavor not in pp.ALLOWED_HVG_FLAVORS:
        raise ValueError(
            f"Invalid hvg_flavor '{hvg_flavor}'. Expected one of "
            f"{pp.ALLOWED_HVG_FLAVORS}."
        )

    out_dir = Path(f"/root/{project_name}_preprocess")
    figures_dir = out_dir / "figures"
    os.makedirs(figures_dir, exist_ok=True)
    sc.settings.file_format_figs = "png"
    sc.settings.figdir = str(figures_dir)

    logging.info("Creating AnnData objects...")
    adatas = pp.make_anndatas(runs, genome_str)
    samples = [run.run_id for run in runs]
    if len(samples) > 1:
        logging.info("Combining objects...")
        adata = ad.concat(adatas, keys=samples, label="batch")
    else:
        adata = adatas[0]

    adata.uns["spatial"] = {}
    for sample_adata in adatas:
        adata.uns["spatial"].update(sample_adata.uns.get("spatial", {}))

    pp.calculate_qc(adata, genome_str)

    sc.pl.violin(
        adata,
        ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
        jitter=False,
        stripplot=False,
        multi_panel=True,
        save="_preFiltering",
    )

    adata = pp.filter_adata(
        adata,
        min_cells=min_cells,
        min_genes=min_genes,
        min_counts=min_counts,
        max_counts=max_counts,
        max_pct_mt=max_pct_mt,
    )

    sc.pl.violin(
        adata,
        ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
        jitter=False,
        stripplot=False,
        multi_panel=True,
        save="_postFiltering",
    )

    adata = pp.add_spatial(adata)

    try:
        adata = pp.add_spatial_neighbors(adata)
    except Exception as e:
        warning = (
            "Unable to build spatial neighbor graph for coherence scoring. "
            f"Proceeding without spatial coherence outputs. Exception: {e}"
        )
        logging.warning(warning)
        message(
            typ="warning",
            data={"title": "spatial coherence skipped", "body": warning},
        )

    adata.layers["counts"] = adata.X.copy()

    if normalization_mode == "tpm":
        if gene_lengths_file is None:
            raise ValueError(
                "normalization_mode='tpm' requires gene_lengths_file to be set."
            )
        lengths_df = pd.read_csv(Path(gene_lengths_file.local_path))
        pp.add_tpm_layer(adata, lengths_df)
        adata.X = adata.layers["log1p_tpm"].copy()
        adata.layers["log1p"] = adata.X.copy()
    else:
        sc.pp.normalize_total(adata, target_sum=normalize_target_sum)
        adata.layers["normalized"] = adata.X.copy()
        sc.pp.log1p(adata)
        adata.layers["log1p"] = adata.X.copy()

    pp.select_highly_variable_genes(
        adata,
        n_top_genes=n_top_genes,
        flavor=hvg_flavor,
    )
    sc.pp.scale(adata, zero_center=True, max_value=10)

    preprocessed_path = out_dir / "preprocessed.h5ad"
    adata.write(preprocessed_path)

    return LatchDir(
        str(out_dir),
        f"latch:///wt_opts/{project_name}/_intermediates/preprocess",
    )


@stagate_gpu_task
def train_stagate_task(
    preprocessed_dir: LatchDir,
    project_name: str,
    clustering_backend: str = "scanpy",
    stagate_k_cutoff: int = 6,
    apply_harmony: bool = True,
) -> LatchDir:
    if clustering_backend not in pp.ALLOWED_CLUSTERING_BACKENDS:
        raise ValueError(
            f"Invalid clustering_backend '{clustering_backend}'. Expected one of "
            f"{pp.ALLOWED_CLUSTERING_BACKENDS}."
        )

    if clustering_backend != "stagate":
        logging.info(
            "Skipping STAGATE training because clustering_backend=%s.",
            clustering_backend,
        )
        return preprocessed_dir

    pp.require_stagate_module()

    preprocess_path = Path(preprocessed_dir.local_path)
    adata = ad.read_h5ad(preprocess_path / "preprocessed.h5ad")
    adata = pp.train_stagate_embedding(
        adata,
        k_cutoff=stagate_k_cutoff,
        apply_harmony=apply_harmony,
        random_state=RANDOM_STATE,
    )

    out_dir = Path(f"/root/{project_name}_stagate_preprocess")
    out_dir.mkdir(parents=True, exist_ok=True)
    adata.write(out_dir / "preprocessed.h5ad")

    return LatchDir(
        str(out_dir),
        f"latch:///wt_opts/{project_name}/_intermediates/stagate_preprocess",
    )


@custom_task(cpu=2, memory=4, storage_gib=50)
def build_wt_opt_jobs_task(
    project_name: str,
    preprocess_dir: LatchDir,
    clustering_backend: str,
    resolution: List[float],
    n_comps: List[int],
    n_neighbors: List[int],
    min_dist: float,
    spread: float,
    apply_harmony: bool = True,
    merge_small_clusters: Optional[int] = 200,
    compute_cluster_markers: bool = True,
    marker_top_n: int = 50,
) -> List[utils.WTOptSetInput]:
    if clustering_backend not in pp.ALLOWED_CLUSTERING_BACKENDS:
        raise ValueError(
            f"Invalid clustering_backend '{clustering_backend}'. Expected one of "
            f"{pp.ALLOWED_CLUSTERING_BACKENDS}."
        )
    if marker_top_n < 1:
        raise ValueError("marker_top_n must be at least 1.")

    merge_small_clusters_threshold = (
        0 if merge_small_clusters is None else merge_small_clusters
    )

    jobs: List[utils.WTOptSetInput] = []
    if clustering_backend == "scanpy":
        sets = list(itertools.product(resolution, n_comps, n_neighbors))
        logging.info("Creating %d scanpy optimization parameter set jobs.", len(sets))
        for i, (cr, nc, nn) in enumerate(sets, start=1):
            jobs.append(
                utils.WTOptSetInput(
                    set_index=i,
                    project_name=project_name,
                    clustering_backend=clustering_backend,
                    resolution=cr,
                    n_comps=nc,
                    n_neighbors=nn,
                    min_dist=min_dist,
                    spread=spread,
                    preprocess_dir=preprocess_dir,
                    apply_harmony=apply_harmony,
                    merge_small_clusters=merge_small_clusters_threshold,
                    compute_cluster_markers=compute_cluster_markers,
                    marker_top_n=marker_top_n,
                )
            )
        return jobs

    sets = list(itertools.product(resolution, n_neighbors))
    logging.info("Creating %d STAGATE optimization parameter set jobs.", len(sets))
    for i, (cr, nn) in enumerate(sets, start=1):
        jobs.append(
            utils.WTOptSetInput(
                set_index=i,
                project_name=project_name,
                clustering_backend=clustering_backend,
                resolution=cr,
                n_neighbors=nn,
                min_dist=min_dist,
                spread=spread,
                preprocess_dir=preprocess_dir,
                apply_harmony=apply_harmony,
                merge_small_clusters=merge_small_clusters_threshold,
                compute_cluster_markers=compute_cluster_markers,
                marker_top_n=marker_top_n,
            )
        )

    return jobs


@custom_task(cpu=4, memory=128, storage_gib=1000)
def opt_set_task(job: utils.WTOptSetInput) -> utils.WTOptSetResult:
    set_str = utils.format_wt_opt_set_str(
        set_index=job.set_index,
        clustering_backend=job.clustering_backend,
        resolution=job.resolution,
        n_neighbors=job.n_neighbors,
        min_dist=job.min_dist,
        spread=job.spread,
        n_comps=job.n_comps,
    )
    out_dir = Path(f"/root/{job.project_name}/{set_str}")
    os.makedirs(out_dir, exist_ok=True)

    try:
        preprocess_path = Path(job.preprocess_dir.local_path)
        adata = ad.read_h5ad(preprocess_path / "preprocessed.h5ad")

        if job.clustering_backend == "scanpy":
            logging.info(
                "Set %d: clustering resolution %s, number of components %s, "
                "neighborhood size %s, umap minimum %s, umap spread %s.",
                job.set_index,
                job.resolution,
                job.n_comps,
                job.n_neighbors,
                job.min_dist,
                job.spread,
            )
            adata = pp.add_clusters(
                adata,
                job.resolution,
                job.n_comps,
                job.n_neighbors,
                job.min_dist,
                job.spread,
                apply_harmony=job.apply_harmony,
                merge_small_clusters=job.merge_small_clusters,
                random_state=RANDOM_STATE,
            )
        else:
            logging.info(
                "Set %d: STAGATE resolution %s, neighborhood size %s, "
                "umap minimum %s, umap spread %s.",
                job.set_index,
                job.resolution,
                job.n_neighbors,
                job.min_dist,
                job.spread,
            )
            adata = pp.add_stagate_clusters(
                adata,
                job.resolution,
                job.n_neighbors,
                job.min_dist,
                job.spread,
                merge_small_clusters=job.merge_small_clusters,
                random_state=RANDOM_STATE,
            )

        if job.compute_cluster_markers:
            try:
                _write_cluster_marker_outputs(
                    adata,
                    out_dir,
                    marker_top_n=job.marker_top_n,
                )
            except Exception as e:
                logging.warning(
                    "Cluster marker output failed for %s: %s",
                    set_str,
                    e,
                )

        adata.write(out_dir / "combined.h5ad")
        return utils.WTOptSetResult(
            set_index=job.set_index,
            set_str=set_str,
            clustering_backend=job.clustering_backend,
            resolution=job.resolution,
            n_comps=job.n_comps,
            n_neighbors=job.n_neighbors,
            min_dist=job.min_dist,
            spread=job.spread,
            succeeded=True,
            output_dir=LatchDir(
                str(out_dir),
                f"latch:///wt_opts/{job.project_name}/{set_str}",
            ),
        )
    except Exception as e:
        logging.warning("Exception for %s: %s", set_str, e)
        return utils.WTOptSetResult(
            set_index=job.set_index,
            set_str=set_str,
            clustering_backend=job.clustering_backend,
            resolution=job.resolution,
            n_comps=job.n_comps,
            n_neighbors=job.n_neighbors,
            min_dist=job.min_dist,
            spread=job.spread,
            succeeded=False,
            error_message=str(e),
        )


@custom_task(cpu=4, memory=512, storage_gib=1000)
def wtOpt_task(
    preprocess_dir: LatchDir,
    runs: List[utils.Run],
    genome: utils.Genome,
    project_name: str,
    results: List[utils.WTOptSetResult],
    resolution: List[float] = [1.0],
    n_comps: List[int] = [30],
    n_top_genes: int = 4000,
    hvg_flavor: str = "seurat",
    stagate_k_cutoff: int = 6,
    n_neighbors: List[int] = [15],
    clustering_backend: str = "scanpy",
    min_dist: float = 0.5,
    spread: float = 1.0,
    apply_harmony: bool = True,
    min_genes: int = 0,
    min_cells: int = 0,
    min_counts: int = 0,
    max_counts: int = 0,
    max_pct_mt: float = 100.0,
    merge_small_clusters: Optional[int] = 200,
    compute_cluster_markers: bool = True,
    marker_top_n: int = 50,
    normalize_target_sum: Optional[float] = None,
    pt_size: Optional[float] = None,
    qc_pt_size: Optional[float] = None,
) -> LatchDir:
    samples = [run.run_id for run in runs]
    channels = max({utils.get_channels(run) for run in runs})
    groups = utils.get_groups(runs)
    logging.info("Comparing features amoung groups %s.", groups)

    if hvg_flavor not in pp.ALLOWED_HVG_FLAVORS:
        raise ValueError(
            f"Invalid hvg_flavor '{hvg_flavor}'. Expected one of "
            f"{pp.ALLOWED_HVG_FLAVORS}."
        )
    if clustering_backend not in pp.ALLOWED_CLUSTERING_BACKENDS:
        raise ValueError(
            f"Invalid clustering_backend '{clustering_backend}'. Expected one of "
            f"{pp.ALLOWED_CLUSTERING_BACKENDS}."
        )

    out_dir = Path(f"/root/{project_name}")
    figures_dir = out_dir / "figures"
    os.makedirs(figures_dir, exist_ok=True)

    metadata = {
        "project_name": project_name,
        "genome": genome.value,
        "resolution": resolution,
        "n_comps": n_comps,
        "n_top_genes": n_top_genes,
        "hvg_flavor": hvg_flavor,
        "stagate_k_cutoff": stagate_k_cutoff,
        "n_neighbors": n_neighbors,
        "clustering_backend": clustering_backend,
        "apply_harmony": apply_harmony,
        "min_dist": min_dist,
        "spread": spread,
        "min_genes": min_genes,
        "min_cells": min_cells,
        "min_counts": min_counts,
        "max_counts": max_counts,
        "max_pct_mt": max_pct_mt,
        "merge_small_clusters": merge_small_clusters,
        "compute_cluster_markers": compute_cluster_markers,
        "marker_top_n": marker_top_n,
        "normalize_target_sum": normalize_target_sum,
        "pt_size": pt_size,
        "qc_pt_size": qc_pt_size,
        "runs": [
            {
                "run_id": run.run_id,
                "condition": utils.sanitize_condition(run.condition),
            }
            for run in runs
        ],
    }
    _write_metadata_csv(out_dir / "metadata.csv", metadata)

    preprocess_path = Path(preprocess_dir.local_path)
    preprocessed_h5ad = preprocess_path / "preprocessed.h5ad"
    if not preprocessed_h5ad.exists():
        raise FileNotFoundError(
            f"Expected preprocessed AnnData at '{preprocessed_h5ad}'."
        )

    adata = ad.read_h5ad(preprocessed_h5ad)

    successful_results = [result for result in results if result.succeeded]
    adata_dict: Dict[str, ad.AnnData] = {}
    for result in successful_results:
        if result.output_dir is None:
            continue
        combined_path = Path(result.output_dir.local_path) / "combined.h5ad"
        if combined_path.exists():
            adata_dict[result.set_str] = ad.read_h5ad(combined_path)

    if len(successful_results) == 0:
        warning = (
            "No parameter sets completed successfully; skipping UMAP/spatial "
            "summary plots."
        )
        logging.warning(warning)
        message(
            typ="warning",
            data={
                "title": "no successful parameter sets",
                "body": warning,
            },
        )
    else:
        condition_count = len({
            utils.sanitize_condition(run.condition)
            for run in runs
        })
        umap_color_keys = ["cluster"]
        if len(set(samples)) > 1:
            umap_color_keys.append("sample")
        if condition_count > 1:
            umap_color_keys.append("condition")

        pl.combine_umaps(
            adata_dict,
            str(figures_dir / "all_umaps.png"),
            html_output_path=str(out_dir / "all_umaps.html"),
            color_keys=umap_color_keys,
        )

        pt_size = pt_size if pt_size is not None else utils.pt_sizes[channels]["dim"]
        pl.combine_spatials(
            adata_dict,
            samples,
            str(figures_dir / "all_spatialdim.png"),
            pt_size=pt_size,
            html_output_path=str(out_dir / "all_spatialdim.html"),
        )

    qc_metrics = ["n_genes_by_counts", "total_counts", "pct_counts_mt"]
    qc_pt_size = (
        qc_pt_size if qc_pt_size is not None else utils.pt_sizes[channels]["qc"]
    )
    pl.plot_spatial_qc(
        adata,
        samples,
        qc_metrics,
        str(figures_dir / "spatial_qc.png"),
        pt_size=qc_pt_size,
        html_output_path=str(out_dir / "spatial_qc.html"),
    )

    grouped = adata.obs.groupby("sample")
    medians_df = grouped.agg(
        {
            "total_counts": "median",
            "n_genes_by_counts": "median",
            "pct_counts_mt": "median",
        }
    ).reset_index()
    medians_df.rename(
        columns={
            "sample": "run_id",
            "total_counts": "umi counts",
            "n_genes_by_counts": "gene counts",
            "pct_counts_mt": "percent mitochondrial",
        },
        inplace=True,
    )
    medians_df.to_csv(out_dir / "medians.csv", index=False)

    has_spatial_graph = "spatial_connectivities" in adata.obsp
    if has_spatial_graph and len(adata_dict) > 0:
        try:
            coherence_df = pp.spatial_coherence_table(adata_dict)
            coherence_df.to_csv(out_dir / "spatial_coherence.csv", index=False)
            pl.plot_spatial_coherence(
                coherence_df,
                str(figures_dir / "spatial_coherence.png"),
            )
        except Exception as e:
            warning = (
                "Spatial coherence calculation failed after clustering. "
                f"Exception: {e}"
            )
            logging.warning(warning)
            message(
                typ="warning",
                data={"title": "spatial coherence failed", "body": warning},
            )

    return LatchDir(str(out_dir), f"latch:///wt_opts/{project_name}")
