from typing import List, Optional

from latch import map_task
from latch.resources.workflow import workflow
from latch.types import LatchFile
from latch.types.metadata import (LatchAuthor, LatchMetadata, LatchParameter,
                                  LatchRule, Params, Section, Spoiler, Text)

from wf.task import (build_wt_opt_jobs_task, opt_set_task, preprocess_wt_task,
                     train_stagate_task, wtOpt_task)
from wf.utils import Genome, Run

flow = [
    Params("runs"),
    Section(
        "Global Parameters",
        Params("project_name"),
        Params("genome"),
        Params("clustering_backend"),
        Section(
            "STAGATE Parameters",
            Text(
                "STAGATE-specific options only affect runs where the clustering "
                "backend is set to `stagate`."
            ),
            Params("stagate_embedding_checkpoint"),
            Params("stagate_k_cutoff"),
        )
    ),
    Section(
        "Preprocessing Parameters",
        Section(
            "Filtering",
            Params("min_genes"),
            Params("min_cells"),
            Params("min_counts"),
            Params("max_counts"),
            Params("max_pct_mt"),
        ),
        Section(
            "Select Variable Features",
            Params("normalize_target_sum"),
            Params("n_top_genes"),
            Params("hvg_flavor"),
        ),
    ),
    Section(
        "Iterative Parameters",
        Params("resolution"),
        Params("n_neighbors"),
        Params("n_comps"),
    ),
    Spoiler(
        "Advanced Options",
        Params("merge_small_clusters"),
        Params("apply_harmony"),
        Section(
            "UMAP Display",
            Params("min_dist"),
            Params("spread"),
        ),
        Section(
            "Figure Display",
            Params("pt_size"),
            Params("qc_pt_size"),
        ),
    ),
]

metadata = LatchMetadata(
    display_name="optimize_wt",
    author=LatchAuthor(
        name="James McGann",
        email="jamesm@atlasxomics.com",
        github="github.com/atlasxomics"
    ),
    repository="https://github.com/atlasxomics/optimize_wt",
    license="MIT",
    parameters={
        "runs": LatchParameter(
            display_name="runs",
            description="List of runs to be analyzed; each run must contain a \
                run_id and a path to outputs from an alignment Workflow \
                (i.e., STAR); optional: condition.  Note that multiple \
                Conditions must be separted by '_' (i.e., Female-control).",
            batch_table_column=True,
            samplesheet=True
        ),
        "project_name": LatchParameter(
            display_name="project name",
            description="Name of output directory in wt_opts/",
            batch_table_column=True,
            rules=[
                LatchRule(
                    regex="^[^/].*",
                    message="project name cannot start with a '/'"
                )
            ]
        ),
        "genome": LatchParameter(
            display_name="genome",
            description="Reference genome for runs.",
            batch_table_column=True,
        ),
        "n_comps": LatchParameter(
            display_name="number of components (scanpy only)",
            description="Number of components/dimensions to keep during \
                dimensionality reduction with `scanpy.pp.pca`.",
            batch_table_column=True
        ),
        "n_top_genes": LatchParameter(
            display_name="highly variable features",
            description="Number of highly variable genes/features to retain \
                for downstream dimensionality reduction.",
            batch_table_column=True
        ),
        "hvg_flavor": LatchParameter(
            display_name="hvg flavor: seurat, cell_ranger, seurat_v3, seurat_v3_paper",
            description="Flavor argument passed to \
                `scanpy.pp.highly_variable_genes`.",
            batch_table_column=True,
            rules=[
                LatchRule(
                    regex="^(seurat|cell_ranger|seurat_v3|seurat_v3_paper)$",
                    message="hvg_flavor must be one of: seurat, cell_ranger, seurat_v3, seurat_v3_paper",
                )
            ]
        ),
        "stagate_k_cutoff": LatchParameter(
            display_name="STAGATE k cutoff",
            description="Number of spatial nearest neighbors per spot when \
                building the STAGATE KNN graph. Ignored unless \
                `clustering_backend` is `stagate`.",
            batch_table_column=True
        ),
        "n_neighbors": LatchParameter(
            display_name="neighborhood sizes",
            description="The size of local neighborhood (number of cells) \
                in `scanpy.pp.neighbors`.",
            batch_table_column=True
        ),
        "resolution": LatchParameter(
            display_name="clustering resolution",
            description="Clustering resolution for Leiden algorithm; higher \
                values result in more clusters.",
            batch_table_column=True
        ),
        "clustering_backend": LatchParameter(
            display_name="clustering backend: scanpy or stagate",
            description="Choose the clustering backend. `scanpy` uses the \
                current PCA/Harmony pipeline; `stagate` trains a STAGATE \
                embedding and clusters on that representation.",
            batch_table_column=True,
            rules=[
                LatchRule(
                    regex="^(scanpy|stagate)$",
                    message="clustering_backend must be one of: scanpy, stagate",
                )
            ]
        ),
        "apply_harmony": LatchParameter(
            display_name="apply harmony integration (scanpy only)",
            description="Apply Harmony batch correction across samples before \
                neighbor graph construction. Ignored for single-sample runs.",
            batch_table_column=True
        ),
        "min_dist": LatchParameter(
            display_name="umap minimum distance",
            description="'The effective minimum distance between embedded \
                points. Smaller values will result in a more \
                clustered/clumped embedding where nearby points on the \
                manifold are drawn closer together, while larger values \
                will result on a more even dispersal of points. The value \
                should be set relative to the spread value, which determines \
                the scale at which embedded points will be spread out.' - \
                Scanpy docs",
            batch_table_column=True
        ),
        "spread": LatchParameter(
            display_name="umap spread",
            description="'The effective scale of embedded points. In \
                combination with min_dist this determines how \
                clustered/clumped the embedded points are.' - Scanpy docs",
            batch_table_column=True
        ),
        "min_genes": LatchParameter(
            display_name="minimum genes",
            description="Threshold for filtering cell from AnnData object.",
            batch_table_column=True
        ),
        "min_cells": LatchParameter(
            display_name="minimum cells",
            description="Threshold for filtering genes from AnnData object.",
            batch_table_column=True
        ),
        "min_counts": LatchParameter(
            display_name="minimum counts",
            description="Minimum UMI counts required for each spot/cell.",
            batch_table_column=True
        ),
        "max_counts": LatchParameter(
            display_name="maximum counts",
            description="Maximum UMI counts allowed for each spot/cell. Set to \
                0 to disable.",
            batch_table_column=True
        ),
        "max_pct_mt": LatchParameter(
            display_name="maximum mitochondrial percent",
            description="Maximum percent mitochondrial counts per spot/cell. \
                Set to 100 to disable.",
            batch_table_column=True
        ),
        "merge_small_clusters": LatchParameter(
            display_name="merge small clusters",
            description="Minimum cluster size after Leiden clustering. Any \
                cluster smaller than this threshold is iteratively merged into \
                its nearest cluster in embedding space. Set to 0 to disable.",
            batch_table_column=True
        ),
        "normalize_target_sum": LatchParameter(
            display_name="normalize target sum",
            description="Optional target sum for `scanpy.pp.normalize_total`. \
                Leave unset to use Scanpy's default behavior.",
            batch_table_column=True
        ),
        "stagate_embedding_checkpoint": LatchParameter(
            display_name="STAGATE embedding checkpoint",
            description="Optional checkpoint file created by a previous \
                STAGATE workflow run. If provided, the workflow validates and \
                reuses `X_stagate` to skip STAGATE training.",
            batch_table_column=True
        ),
        "pt_size": LatchParameter(
            display_name="Override point size",
            description="Point size for spatial plot of clustering. \
                Recommendations: 50x:75, 96x:5, 220:5.",
            batch_table_column=True,
            hidden=True
        ),
        "qc_pt_size": LatchParameter(
            display_name="Override spatial QC point size",
            description="Point size for spatial plot of clustering. \
                Recommendations: 50x:25, 96x:1, 220:0.5.",
            batch_table_column=True,
            hidden=True
        ),
    },
    flow=flow
)


@workflow(metadata)
def wtOpt_workflow(
    runs: List[Run],
    genome: Genome,
    project_name: str,
    resolution: List[float] = [1.0],
    n_comps: List[int] = [30],
    n_top_genes: int = 4000,
    hvg_flavor: str = "seurat",
    stagate_k_cutoff: int = 6,
    n_neighbors: List[int] = [15],
    clustering_backend: str = "scanpy",
    apply_harmony: bool = True,
    min_dist: float = 0.5,
    spread: float = 1.0,
    min_genes: int = 1,
    min_cells: int = 1,
    min_counts: int = 0,
    max_counts: int = 0,
    max_pct_mt: float = 100.0,
    merge_small_clusters: Optional[int] = 200,
    normalize_target_sum: Optional[float] = None,
    stagate_embedding_checkpoint: Optional[LatchFile] = None,
    pt_size: Optional[float] = None,
    qc_pt_size: Optional[float] = None,
) -> None:
    """Determine optimal input parameters for spatial whole transcriptome
    analysis.
    """
    preprocess_dir = preprocess_wt_task(
        runs=runs,
        genome=genome,
        project_name=project_name,
        n_top_genes=n_top_genes,
        hvg_flavor=hvg_flavor,
        min_genes=min_genes,
        min_cells=min_cells,
        min_counts=min_counts,
        max_counts=max_counts,
        max_pct_mt=max_pct_mt,
        normalize_target_sum=normalize_target_sum,
    )

    stagate_embedding_checkpoint = train_stagate_task(
        preprocessed_dir=preprocess_dir,
        project_name=project_name,
        clustering_backend=clustering_backend,
        runs=runs,
        genome=genome,
        min_genes=min_genes,
        min_cells=min_cells,
        min_counts=min_counts,
        max_counts=max_counts,
        max_pct_mt=max_pct_mt,
        normalize_target_sum=normalize_target_sum,
        n_top_genes=n_top_genes,
        hvg_flavor=hvg_flavor,
        stagate_k_cutoff=stagate_k_cutoff,
        stagate_embedding_checkpoint=stagate_embedding_checkpoint,
    )

    opt_jobs = build_wt_opt_jobs_task(
        runs=runs,
        genome=genome,
        project_name=project_name,
        preprocess_dir=preprocess_dir,
        clustering_backend=clustering_backend,
        resolution=resolution,
        n_comps=n_comps,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        spread=spread,
        apply_harmony=apply_harmony,
        merge_small_clusters=merge_small_clusters,
        min_genes=min_genes,
        min_cells=min_cells,
        min_counts=min_counts,
        max_counts=max_counts,
        max_pct_mt=max_pct_mt,
        normalize_target_sum=normalize_target_sum,
        n_top_genes=n_top_genes,
        hvg_flavor=hvg_flavor,
        stagate_k_cutoff=stagate_k_cutoff,
        stagate_embedding_checkpoint=stagate_embedding_checkpoint,
    )
    mapped_results = map_task(opt_set_task)(job=opt_jobs)

    results = wtOpt_task(
        preprocess_dir=preprocess_dir,
        runs=runs,
        genome=genome,
        project_name=project_name,
        results=mapped_results,
        resolution=resolution,
        n_comps=n_comps,
        n_top_genes=n_top_genes,
        hvg_flavor=hvg_flavor,
        stagate_k_cutoff=stagate_k_cutoff,
        n_neighbors=n_neighbors,
        clustering_backend=clustering_backend,
        apply_harmony=apply_harmony,
        min_genes=min_genes,
        min_cells=min_cells,
        min_counts=min_counts,
        max_counts=max_counts,
        max_pct_mt=max_pct_mt,
        merge_small_clusters=merge_small_clusters,
        normalize_target_sum=normalize_target_sum,
        stagate_embedding_checkpoint=stagate_embedding_checkpoint,
        min_dist=min_dist,
        spread=spread,
        pt_size=pt_size,
        qc_pt_size=qc_pt_size
    )

    return results


if __name__ == "__main__":
    wtOpt_workflow(
        runs=[Run(
            run_id="test",
            gex_dir="latch://13502.account/star_outputs/D02042_NG06104/STAR_outputsGeneFull/raw",
            spatial_dir="latch://atx-illumina.mount/Images_spatial/D2042/spatial",
            condition="None"
        )],
        genome=Genome.mm10,
        project_name="wf_test",
    )
