import logging

from dataclasses import dataclass
from typing import Optional

from latch.types import LatchDir, LatchFile

# Shared AtlasXomics utilities — re-exported for `import wf.utils as utils`
from atx_common import (  # noqa: F401
    Genome,
    filter_anndata,
    get_channels,
    get_genome_fasta,
    get_groups,
    get_LatchFile,
    pt_sizes,
    sanitize_condition,
)


logging.basicConfig(
    format="%(levelname)s - %(asctime)s - %(message)s", level=logging.INFO
)


# ---------------------------------------------------------------------------
# Workflow-specific reference data (kept local due to typo in key name
# that downstream code may depend on)
# ---------------------------------------------------------------------------
gene_keys = {
    "mitochondiral": {
        "hg38": "MT-",
        "mm10": ("Mt-", "mt-"),
        "mm39": ("Mt-", "mt-"),
        "rnor6": ("Mt-", "mt-"),
    },
    "ribosomal": {
        "hg38": ("RPS", "RPL"),
        "mm10": ("Rps", "Rpl"),
        "mm39": ("Rps", "Rpl"),
        "rnor6": ("Rps", "Rpl"),
    },
}


# ---------------------------------------------------------------------------
# Workflow-specific dataclasses
# ---------------------------------------------------------------------------
@dataclass
class Run:
    run_id: str
    gex_dir: LatchDir
    spatial_dir: LatchDir
    condition: str = "None"


@dataclass
class WTOptSetInput:
    set_index: int
    project_name: str
    clustering_backend: str
    resolution: float
    n_neighbors: int
    min_dist: float
    spread: float
    preprocess_dir: LatchDir
    n_comps: Optional[int] = None
    apply_harmony: bool = True
    merge_small_clusters: int = 0
    stagate_embedding_checkpoint: Optional[LatchFile] = None
    stagate_expected_metadata_json: Optional[str] = None


@dataclass
class WTOptSetResult:
    set_index: int
    set_str: str
    clustering_backend: str
    resolution: float
    n_neighbors: int
    min_dist: float
    spread: float
    n_comps: Optional[int] = None
    succeeded: bool = False
    error_message: Optional[str] = None
    output_dir: Optional[LatchDir] = None


# ---------------------------------------------------------------------------
# Workflow-specific helpers
# ---------------------------------------------------------------------------
def format_wt_opt_set_str(
    set_index: int,
    clustering_backend: str,
    resolution: float,
    n_neighbors: int,
    min_dist: float,
    spread: float,
    n_comps: Optional[int] = None,
) -> str:
    cr_str = str(resolution).replace(".", "-")
    md_str = str(min_dist).replace(".", "-")
    sp_str = str(spread).replace(".", "-")

    if clustering_backend == "scanpy":
        if n_comps is None:
            raise ValueError("scanpy optimization sets require n_comps.")
        return (
            f"set{set_index}_backend-scanpy_cr{cr_str}-nc{n_comps}-"
            f"nn{n_neighbors}-md{md_str}-sp{sp_str}"
        )

    return (
        f"set{set_index}_backend-stagate_cr{cr_str}-nn{n_neighbors}-"
        f"md{md_str}-sp{sp_str}"
    )
