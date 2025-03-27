import json
import logging

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List

import anndata as ad

from latch.types import LatchFile, LatchDir


logging.basicConfig(
    format="%(levelname)s - %(asctime)s - %(message)s", level=logging.INFO
)


gene_keys = {
    "mitochondiral": {
        "hg38": "MT-", "mm10": ("Mt-", "mt-"), "mm39": ("Mt-", "mt-")
    },
    "ribosomal":  {
        "hg38": ("RPS", "RPL"), "mm10": ("Rps", "Rpl"), "mm39": ("Rps", "Rpl")
    }
}

# Map DBiT channels to plot point sizes for various spatial plots
pt_sizes = {
    50: {"dim": 75, "qc": 25},
    96: {"dim": 10, "qc": 5},
    210: {"dim": 0.25, "qc": 0.25},
    220: {"dim": 0.25, "qc": 0.25}
}


class Genome(Enum):
    hg38 = "hg38"
    mm10 = "mm10"
    m39 = "mm39"
    rnor6 = "rnor6"


@dataclass
class Run:
    run_id: str
    gex_dir: LatchDir
    spatial_dir: LatchDir
    condition: str = "None"


def filter_anndata(
    adata: ad.AnnData, group: str, subgroup: List[str]
) -> ad.AnnData:
    return adata[adata.obs[group] == subgroup]


def get_channels(run: Run) -> int:

    try:
        spatial_dir = run.spatial_dir.local_path
        metadata_json = f"{spatial_dir}/metadata.json"

        with open(metadata_json, "r") as f:
            metadata = json.load(f)
            channels = metadata["numChannels"]

    except FileNotFoundError as e:
        logging.warning(f"{e}: metadata.json not found in spatial folder; \
                        defaulting to 220.")
        return 220

    return int(channels)


def get_genome_fasta(genome: str) -> LatchFile:
    """Download reference genome fasta files from latch-public"""

    fasta_paths = {
        "mm10": "s3://latch-public/test-data/13502/GRCm38_genome.fa",
        "hg38":  "s3://latch-public/test-data/13502/GRCh38_genome.fa",
        "rnor6": "s3://latch-public/test-data/13502/Rnor6_genome.fa"
    }

    return LatchFile(fasta_paths[genome])


def get_groups(runs: List[Run]) -> List[str]:
    """Set 'groups' list for differential analysis"""

    samples = [run.run_id for run in runs]
    conditions = list({run.condition for run in runs})

    groups = ["cluster"]
    if len(samples) > 1:
        groups.append("sample")
    if len(conditions) > 1:
        groups.append("condition")

    return groups


def get_LatchFile(directory: LatchDir, file_name: str) -> LatchFile:
    try:
        files = [file for file in directory.iterdir()
                 if isinstance(file, LatchFile) and
                 Path(file.path).name == file_name]
        if len(files) == 1:
            return files[0]
        elif len(files) == 0:
            raise FileNotFoundError(
                f"No file {file_name} found in {directory.remote_path}"
            )
        elif len(files) > 1:
            raise FileNotFoundError(
                f"Multiple files {file_name} found in {directory.remote_path}"
            )
    except Exception as e:
        logging.error(f"Failed to find file '{file_name}'; error {e}")
        return None
