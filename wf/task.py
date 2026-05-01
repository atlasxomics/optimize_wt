import itertools
import json
import logging
import os
import shutil
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


def _build_stagate_checkpoint_metadata(
    runs: List[utils.Run],
    genome: str,
    min_genes: int,
    min_cells: int,
    min_counts: int,
    max_counts: int,
    max_pct_mt: float,
    normalize_target_sum: Optional[float],
    n_top_genes: int,
    hvg_flavor: str,
    stagate_k_cutoff: int,
    apply_harmony: bool,
) -> Dict[str, object]:
    return {
        "runs": [
            {
                "run_id": run.run_id,
                "condition": utils.sanitize_condition(run.condition),
            }
            for run in runs
        ],
        "genome": genome,
        "min_genes": min_genes,
        "min_cells": min_cells,
        "min_counts": min_counts,
        "max_counts": max_counts,
        "max_pct_mt": max_pct_mt,
        "normalize_target_sum": normalize_target_sum,
        "n_top_genes": n_top_genes,
        "hvg_flavor": hvg_flavor,
        "stagate_k_cutoff": stagate_k_cutoff,
        "apply_harmony": apply_harmony,
    }


def _write_metadata_csv(output_path: Path, metadata: Dict[str, object]) -> None:
    pd.DataFrame([metadata]).to_csv(output_path, index=False)


def _copytree_contents(src: Path, dst: Path) -> None:
    if not src.exists():
        return

    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)


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
    runs: List[utils.Run],
    genome: utils.Genome,
    clustering_backend: str = "scanpy",
    min_genes: int = 0,
    min_cells: int = 0,
    min_counts: int = 0,
    max_counts: int = 0,
    max_pct_mt: float = 100.0,
    normalize_target_sum: Optional[float] = None,
    n_top_genes: int = 4000,
    hvg_flavor: str = "seurat",
    stagate_k_cutoff: int = 6,
    apply_harmony: bool = True,
    stagate_embedding_checkpoint: Optional[LatchFile] = None,
) -> Optional[LatchFile]:
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
        return stagate_embedding_checkpoint

    if stagate_embedding_checkpoint is not None:
        logging.info(
            "Reusing provided STAGATE embedding checkpoint from %s.",
            stagate_embedding_checkpoint.remote_path,
        )
        return stagate_embedding_checkpoint

    pp.require_stagate_module()

    preprocess_path = Path(preprocessed_dir.local_path)
    adata = ad.read_h5ad(preprocess_path / "preprocessed.h5ad")
    adata = pp.train_stagate_embedding(
        adata,
        k_cutoff=stagate_k_cutoff,
        apply_harmony=apply_harmony,
        random_state=RANDOM_STATE,
    )

    metadata = _build_stagate_checkpoint_metadata(
        runs=runs,
        genome=genome.value,
        min_genes=min_genes,
        min_cells=min_cells,
        min_counts=min_counts,
        max_counts=max_counts,
        max_pct_mt=max_pct_mt,
        normalize_target_sum=normalize_target_sum,
        n_top_genes=n_top_genes,
        hvg_flavor=hvg_flavor,
        stagate_k_cutoff=stagate_k_cutoff,
        apply_harmony=apply_harmony,
    )

    out_path = Path(f"/root/{project_name}_stagate_embedding_checkpoint.h5ad")
    pp.save_stagate_embedding_checkpoint(adata, out_path, metadata=metadata)
    return LatchFile(
        str(out_path),
        f"latch:///wt_opts/{project_name}/_intermediates/stagate_embedding_checkpoint.h5ad",
    )


@custom_task(cpu=2, memory=4, storage_gib=50)
def build_wt_opt_jobs_task(
    runs: List[utils.Run],
    genome: utils.Genome,
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
    min_genes: int = 0,
    min_cells: int = 0,
    min_counts: int = 0,
    max_counts: int = 0,
    max_pct_mt: float = 100.0,
    normalize_target_sum: Optional[float] = None,
    n_top_genes: int = 4000,
    hvg_flavor: str = "seurat",
    stagate_k_cutoff: int = 6,
    stagate_embedding_checkpoint: Optional[LatchFile] = None,
) -> List[utils.WTOptSetInput]:
    if clustering_backend not in pp.ALLOWED_CLUSTERING_BACKENDS:
        raise ValueError(
            f"Invalid clustering_backend '{clustering_backend}'. Expected one of "
            f"{pp.ALLOWED_CLUSTERING_BACKENDS}."
        )

    merge_small_clusters_threshold = (
        0 if merge_small_clusters is None else merge_small_clusters
    )
    stagate_expected_metadata_json = None
    if clustering_backend == "stagate":
        stagate_expected_metadata_json = json.dumps(
            _build_stagate_checkpoint_metadata(
                runs=runs,
                genome=genome.value,
                min_genes=min_genes,
                min_cells=min_cells,
                min_counts=min_counts,
                max_counts=max_counts,
                max_pct_mt=max_pct_mt,
                normalize_target_sum=normalize_target_sum,
                n_top_genes=n_top_genes,
                hvg_flavor=hvg_flavor,
                stagate_k_cutoff=stagate_k_cutoff,
                apply_harmony=apply_harmony,
            ),
            sort_keys=True,
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
                    stagate_expected_metadata_json=stagate_expected_metadata_json,
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
                stagate_embedding_checkpoint=stagate_embedding_checkpoint,
                stagate_expected_metadata_json=stagate_expected_metadata_json,
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
            if job.stagate_embedding_checkpoint is None:
                raise ValueError(
                    "STAGATE optimization set requires an embedding checkpoint."
                )
            expected_metadata = (
                json.loads(job.stagate_expected_metadata_json)
                if job.stagate_expected_metadata_json is not None
                else None
            )
            if expected_metadata is None:
                raise ValueError(
                    "STAGATE optimization set is missing checkpoint validation "
                    "metadata."
                )
            adata = pp.load_stagate_embedding_checkpoint(
                adata,
                job.stagate_embedding_checkpoint.local_path,
                expected_metadata=expected_metadata,
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
                f"latch:///wt_opts/{job.project_name}/_intermediates/_mapped_sets/{set_str}",
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
    normalize_target_sum: Optional[float] = None,
    stagate_embedding_checkpoint: Optional[LatchFile] = None,
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
    intermediates_dir = out_dir / "intermediates"
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(intermediates_dir, exist_ok=True)

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
    shutil.copy2(preprocessed_h5ad, intermediates_dir / "preprocessed.h5ad")
    _copytree_contents(preprocess_path / "figures", figures_dir)

    if stagate_embedding_checkpoint is not None:
        shutil.copy2(
            Path(stagate_embedding_checkpoint.local_path),
            intermediates_dir / "stagate_embedding_checkpoint.h5ad",
        )

    successful_results = [result for result in results if result.succeeded]
    adata_dict: Dict[str, ad.AnnData] = {}
    for result in successful_results:
        if result.output_dir is None:
            continue
        combined_path = Path(result.output_dir.local_path) / "combined.h5ad"
        if combined_path.exists():
            adata_dict[result.set_str] = ad.read_h5ad(combined_path)
            set_dir = out_dir / result.set_str
            os.makedirs(set_dir, exist_ok=True)
            shutil.copy2(combined_path, set_dir / "combined.h5ad")

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
        pl.combine_umaps(
            adata_dict,
            str(figures_dir / "all_umaps.png"),
            html_output_path=str(out_dir / "all_umaps.html"),
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
