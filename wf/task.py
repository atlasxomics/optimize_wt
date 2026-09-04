import itertools
import gc
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import anndata as ad
import pandas as pd
import scanpy as sc
from scipy import sparse as sp

from latch import message
from latch.executions import rename_current_execution
from latch.resources.tasks import custom_task
from latch.types import LatchDir
from latch.types.plots import (
    PlotsArtifact,
    PlotsArtifactBindings,
    PlotsArtifactTemplate,
    Widget,
)

try:
    from latch.resources.tasks import g6e_xlarge_task as stagate_gpu_task
except ImportError:
    from latch.resources.tasks import small_gpu_task as stagate_gpu_task

import wf.features as features
import wf.plotting as pl
import wf.preprocessing as pp
import wf.utils as utils

RANDOM_STATE = 42
PLOTS_ARTIFACT_TEMPLATE_ID = "1257"
PLOTS_ARTIFACT_DATA_TRANSFORM_ID = "462483"

logging.basicConfig(
    format="%(levelname)s - %(asctime)s - %(message)s",
    level=logging.INFO,
)


def _write_metadata_csv(output_path: Path, metadata: Dict[str, object]) -> None:
    pd.DataFrame([metadata]).to_csv(output_path, index=False)


def _write_plots_artifact(output_dir: Path, sm_remote_path: str) -> None:
    artifact = PlotsArtifact(
        bindings=PlotsArtifactBindings(
            plot_templates=[
                PlotsArtifactTemplate(
                    template_id=PLOTS_ARTIFACT_TEMPLATE_ID,
                    widgets=[
                        Widget(
                            transform_id=PLOTS_ARTIFACT_DATA_TRANSFORM_ID,
                            key="data_path",
                            value=sm_remote_path,
                        )
                    ],
                )
            ]
        )
    )

    artifacts_dir = output_dir / "Launch_Plots"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    with open(artifacts_dir / "artifact.json", "w") as f:
        json.dump(artifact.asdict(), f, indent=2)


def _write_cluster_marker_outputs(
    adata: ad.AnnData,
    out_dir: Path,
    marker_top_n: int,
) -> None:
    if marker_top_n < 1:
        raise ValueError("marker_top_n must be at least 1.")
    if "cluster" not in adata.obs:
        raise KeyError("Cannot calculate cluster markers: missing obs['cluster'].")
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
    expression = (
        adata.layers["log1p"] if "log1p" in adata.layers else adata.X
    )
    marker_adata = ad.AnnData(
        X=expression[:, keep_genes].copy(),
        obs=pd.DataFrame(
            {"cluster": adata.obs["cluster"].astype(str)},
            index=adata.obs_names.copy(),
        ),
        var=adata.var.loc[keep_genes].copy(),
    )
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
    expression_source = "layers/log1p" if "log1p" in adata.layers else "X"
    adata.uns["cluster_marker_degs_params"] = {
        "groupby": "cluster",
        "method": "wilcoxon",
        "expression_layer": expression_source,
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
        "expression_layer": expression_source,
        "pval_cutoff": 0.05,
        "log2fc_min": 0.25,
        "marker_top_n": marker_top_n,
        "values": "column-wise z-score of mean log1p expression, clipped to [-3, 3]",
    }


@custom_task(cpu=4, memory=16, storage_gib=1000)
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
) -> LatchDir:
    logging.info("Attempting to set execution name to: %s", project_name)
    try:
        rename_current_execution(str(project_name))
        logging.info("Successfully renamed execution to: %s", project_name)
    except Exception as e:
        # Execution naming is presentation-only and should never prevent the
        # analysis from running.
        logging.warning(
            "Unable to rename execution to '%s': %s",
            project_name,
            e,
        )

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
    adatas = pp.make_anndatas(
        runs,
        genome_str,
        include_spatial_images=False,
    )
    samples = [run.run_id for run in runs]
    spatial_metadata = {}
    for sample_adata in adatas:
        spatial_metadata.update(sample_adata.uns.get("spatial", {}))

    if len(samples) > 1:
        logging.info("Combining objects...")
        adata = ad.concat(adatas, keys=samples, label="batch")
    else:
        adata = adatas[0]

    adata.uns["spatial"] = spatial_metadata
    del adatas
    gc.collect()
    pp.log_matrix_storage(adata, "After sample concatenation")

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
    pp.log_matrix_storage(adata, "After cell and gene filtering")

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

    # Preserve one sparse raw-count matrix for downstream count-aware tools.
    # Log-normalized expression remains in X, so this does not reintroduce the
    # full-gene dense scaling that caused the original memory failure.
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=normalize_target_sum)
    sc.pp.log1p(adata)
    pp.select_highly_variable_genes(
        adata,
        n_top_genes=n_top_genes,
        flavor=hvg_flavor,
    )
    if sp.issparse(adata.layers["counts"]):
        adata.layers["counts"] = (
            adata.layers["counts"].tocsr().astype("float32", copy=False)
        )
    else:
        adata.layers["counts"] = adata.layers["counts"].astype(
            "float32",
            copy=False,
        )
    if sp.issparse(adata.X):
        adata.X = adata.X.tocsr().astype("float32", copy=False)
    else:
        adata.X = adata.X.astype("float32", copy=False)
    gc.collect()
    pp.log_matrix_storage(adata, "Final sparse log1p preprocessing matrix")

    preprocessed_path = out_dir / "preprocessed.h5ad"
    adata.write(preprocessed_path)

    return LatchDir(
        str(out_dir),
        f"latch:///rna_analysis/{project_name}/_intermediates/preprocess",
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
        f"latch:///rna_analysis/{project_name}/_intermediates/stagate_preprocess",
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


@custom_task(cpu=4, memory=64, storage_gib=1000)
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

        try:
            pp.run_neighborhood_enrichment(
                adata,
                cluster_key="cluster",
                library_key="sample",
            )
        except Exception as e:
            adata.uns.pop("cluster_nhood_enrichment", None)
            logging.warning(
                "Neighborhood enrichment failed for %s: %s",
                set_str,
                e,
            )
        else:
            neighborhood_group_keys = [
                key
                for key in ("sample", "condition")
                if key in adata.obs and adata.obs[key].dropna().nunique() > 1
            ]
            try:
                pp.precompute_grouped_nhood_enrichment(
                    adata,
                    group_keys=neighborhood_group_keys,
                    cluster_key="cluster",
                    library_key="sample",
                )
            except Exception as e:
                adata.uns.pop(pp.GROUPED_NHOOD_ENRICHMENT_KEY, None)
                logging.warning(
                    "Grouped neighborhood enrichment failed for %s: %s",
                    set_str,
                    e,
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

        features.save_anndata_objects(adata, out_dir)
        output_remote_path = f"latch:///rna_analysis/{job.project_name}/{set_str}"
        _write_plots_artifact(
            out_dir,
            f"{output_remote_path}/combined_sm.h5ad",
        )
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
                output_remote_path,
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


@custom_task(cpu=4, memory=64, storage_gib=1000)
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
    condition_count = len({
        utils.sanitize_condition(run.condition)
        for run in runs
    })
    umap_color_keys = ["cluster"]
    if len(set(samples)) > 1:
        umap_color_keys.append("sample")
    if condition_count > 1:
        umap_color_keys.append("condition")

    pt_size = pt_size if pt_size is not None else utils.pt_sizes[channels]["dim"]
    has_spatial_graph = "spatial_connectivities" in adata.obsp
    umap_image_paths: List[str] = []
    umap_captions: List[str] = []
    spatial_image_paths: List[str] = []
    spatial_captions: List[str] = []
    neighborhood_image_paths: List[str] = []
    neighborhood_captions: List[str] = []
    coherence_rows = []
    processed_set_count = 0

    # Read, summarize, and release one parameter set at a time. The reduced
    # object contains every field needed for UMAP/spatial plotting and cluster
    # coherence, without duplicating the full expression object in memory.
    for result in successful_results:
        if result.output_dir is None:
            continue

        result_dir = Path(result.output_dir.local_path)
        combined_path = result_dir / "combined_sm.h5ad"
        if not combined_path.exists():
            logging.warning(
                "Skipping aggregation for %s: missing %s.",
                result.set_str,
                combined_path,
            )
            continue

        logging.info("Aggregating parameter set sequentially: %s", result.set_str)
        set_adata = ad.read_h5ad(combined_path)
        # Older reduced objects omit raw per-sample spatial coordinates. Restore
        # them from the shared preprocessed object so sequential aggregation can
        # still produce spatial pages. New reduced objects retain this field.
        if "spatial" not in set_adata.obsm and "spatial" in adata.obsm:
            base_spatial = pd.DataFrame(
                adata.obsm["spatial"],
                index=adata.obs_names,
            ).reindex(set_adata.obs_names)
            if base_spatial.isna().to_numpy().any():
                logging.warning(
                    "Unable to align spatial coordinates for %s.",
                    result.set_str,
                )
            else:
                set_adata.obsm["spatial"] = base_spatial.to_numpy()
        if "spatial" not in set_adata.uns and "spatial" in adata.uns:
            set_adata.uns["spatial"] = adata.uns["spatial"]

        # Final set-level summaries only need obs, UMAP/spatial metadata, and
        # the small neighborhood matrices in uns. Release both expression
        # matrices before per-sample plotting creates temporary AnnData subsets.
        set_adata.layers.clear()
        set_adata.raw = None
        set_adata.X = None
        gc.collect()
        processed_set_count += 1
        try:
            set_stem = f"set_{result.set_index:04d}"
            try:
                set_umap_paths = pl.combine_umaps(
                    {result.set_str: set_adata},
                    str(figures_dir / f"{set_stem}_umap.png"),
                    color_keys=umap_color_keys,
                    write_gallery=False,
                )
                umap_image_paths.extend(set_umap_paths)
                umap_captions.extend(
                    [f"Set: {result.set_str}"] * len(set_umap_paths)
                )
            except Exception as e:
                logging.warning(
                    "UMAP aggregation failed for %s: %s",
                    result.set_str,
                    e,
                )

            try:
                set_spatial_paths = pl.combine_spatials(
                    {result.set_str: set_adata},
                    samples,
                    str(figures_dir / f"{set_stem}_spatial.png"),
                    pt_size=pt_size,
                    write_gallery=False,
                )
                spatial_image_paths.extend(set_spatial_paths)
                spatial_captions.extend(
                    [
                        f"Sample {samples[idx % len(samples)]} | "
                        f"Set: {result.set_str}"
                        for idx in range(len(set_spatial_paths))
                    ]
                )
            except Exception as e:
                logging.warning(
                    "Spatial aggregation failed for %s: %s",
                    result.set_str,
                    e,
                )

            try:
                neighborhood_path = (
                    figures_dir / f"{set_stem}_neighborhood.png"
                )
                plotted = pl.plot_neighborhood_enrichment(
                    set_adata,
                    str(neighborhood_path),
                    title=f"{result.set_str}: Neighborhood enrichment",
                )
                if plotted:
                    neighborhood_image_paths.append(str(neighborhood_path))
                    neighborhood_captions.append(
                        f"Set: {result.set_str}"
                    )
            except Exception as e:
                logging.warning(
                    "Neighborhood enrichment plotting failed for %s: %s",
                    result.set_str,
                    e,
                )

            if has_spatial_graph:
                try:
                    if "cluster" not in set_adata.obs:
                        raise KeyError(
                            f"AnnData '{result.set_str}' is missing cluster labels."
                        )
                    aligned_clusters = set_adata.obs["cluster"].reindex(
                        adata.obs_names
                    )
                    if aligned_clusters.isna().any():
                        missing_count = int(aligned_clusters.isna().sum())
                        raise ValueError(
                            f"AnnData '{result.set_str}' is missing "
                            f"{missing_count} preprocessed observations."
                        )
                    cluster_codes = (
                        aligned_clusters.astype(str)
                        .astype("category")
                        .cat.codes.to_numpy(dtype=float)
                    )
                    coherence_rows.append(
                        {
                            "set": result.set_str,
                            "n_clusters": int(aligned_clusters.nunique()),
                            "morans_I": round(
                                pp.morans_i(
                                    adata.obsp["spatial_connectivities"],
                                    cluster_codes,
                                ),
                                4,
                            ),
                        }
                    )
                except Exception as e:
                    logging.warning(
                        "Spatial coherence failed for %s: %s",
                        result.set_str,
                        e,
                    )
        finally:
            del set_adata
            gc.collect()

    if processed_set_count == 0:
        warning = (
            "No successful parameter-set objects were available; skipping "
            "UMAP, spatial, and neighborhood summary plots."
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
        pl.write_html_gallery(
            str(figures_dir / "all_umaps.png"),
            title="Combined UMAPs by " + ", ".join(umap_color_keys),
            image_paths=umap_image_paths,
            captions=umap_captions,
            html_output_path=str(out_dir / "all_umaps.html"),
        )
        pl.write_html_gallery(
            str(figures_dir / "all_spatialdim.png"),
            title="Combined Spatial Cluster Plots",
            image_paths=spatial_image_paths,
            captions=spatial_captions,
            html_output_path=str(out_dir / "all_spatialdim.html"),
        )
        pl.write_html_gallery(
            str(figures_dir / "all_neighborhoods.png"),
            title="Neighborhood Enrichment by Parameter Set",
            image_paths=neighborhood_image_paths,
            captions=neighborhood_captions,
            html_output_path=str(out_dir / "all_neighborhoods.html"),
        )

    if len(coherence_rows) > 0:
        coherence_df = pd.DataFrame(coherence_rows)
        coherence_df.to_csv(out_dir / "spatial_coherence.csv", index=False)
        pl.plot_spatial_coherence(
            coherence_df,
            str(figures_dir / "spatial_coherence.png"),
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

    effective_pt_size = (
        pt_size if pt_size is not None else utils.pt_sizes[channels]["dim"]
    )
    if has_spatial_graph:
        try:
            expression_source = "log1p" if "log1p" in adata.layers else "X"
            svg_df = pp.run_spatial_autocorr(
                adata,
                layer=expression_source,
                n_jobs=4,
            )
            svg_df.to_csv(out_dir / "svg_genes.csv")
            pl.plot_svg_spatial(
                adata,
                samples,
                str(figures_dir / "svg_spatial.png"),
                top_n=10,
                pt_size=effective_pt_size,
                layer=expression_source,
                html_output_path=str(out_dir / "svg_spatial.html"),
            )
        except Exception as e:
            warning = f"Spatial autocorrelation failed: {e}"
            logging.warning(warning)
            message(
                typ="warning",
                data={"title": "SVG analysis failed", "body": warning},
            )

    return LatchDir(str(out_dir), f"latch:///rna_analysis/{project_name}")
