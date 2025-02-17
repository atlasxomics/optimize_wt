from typing import List, Optional

from latch.resources.workflow import workflow
from latch.types.metadata import (LatchAuthor, LatchMetadata, LatchParameter,
                                  LatchRule)

from wf.task import wtOpt_task
from wf.utils import Genome, Run

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
            display_name="number of components",
            description="Number of components/dimensions to keep during \
                dimensionality reduction with `scanpy.pp.pca`.",
            batch_table_column=True
        ),
        "resolution": LatchParameter(
            display_name="clustering resolution",
            description="Clustering resolution for Leiden algorithm; higher \
                values result in more clusters.",
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
    }
)


@workflow(metadata)
def wtOpt_workflow(
    runs: List[Run],
    genome: Genome,
    project_name: str,
    resolution: List[float] = [1.0],
    n_comps: List[int] = [30],
    min_genes: int = 0,
    min_cells: int = 0,
    pt_size: Optional[float] = None,
    qc_pt_size: Optional[float] = None,
) -> None:
    """Determine optimal input parameters for spatial whole transcriptome
    analysis.
    """

    results = wtOpt_task(
        runs=runs,
        genome=genome,
        project_name=project_name,
        resolution=resolution,
        n_comps=n_comps,
        min_genes=min_genes,
        min_cells=min_cells,
        pt_size=pt_size,
        qc_pt_size=qc_pt_size
    )

    return results


if __name__ == "__main__":
    wtOpt_task(
        runs=[Run(
            run_id="test",
            gex_dir="latch://13502.account/star_outputs/D02042_NG06104/STAR_outputsGeneFull/raw",
            spatial_dir="latch://atx-illumina.mount/Images_spatial/D2042/spatial",
            condition="None"
        )],
        genome=Genome.mm10,
        project_name="wf_test",
    )
