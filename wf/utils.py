import json
import logging
import re
import time

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional

import anndata as ad

from latch.types import LatchFile, LatchDir


logging.basicConfig(
    format="%(levelname)s - %(asctime)s - %(message)s", level=logging.INFO
)


gene_keys = {
    "mitochondiral": {
        "hg38": "MT-",
        "mm10": ("Mt-", "mt-"),
        "mm39": ("Mt-", "mt-"),
        "rnor6": ("Mt-", "mt-"),
    },
    "ribosomal":  {
        "hg38": ("RPS", "RPL"),
        "mm10": ("Rps", "Rpl"),
        "mm39": ("Rps", "Rpl"),
        "rnor6": ("Rps", "Rpl"),
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


def sanitize_condition(condition: Optional[str]) -> str:
    """Normalize condition labels for downstream grouping."""
    if condition is None:
        return "None"

    condition_str = str(condition).strip()
    if condition_str == "":
        return "None"

    return re.sub(r"\s+", "_", condition_str)


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
    conditions = list({sanitize_condition(run.condition) for run in runs})

    groups = ["cluster"]
    if len(samples) > 1:
        groups.append("sample")
    if len(conditions) > 1:
        groups.append("condition")

    return groups


def get_LatchFile(
    directory: LatchDir,
    file_name: str,
    retries: int = 3,
    retry_delay_s: float = 5.0
) -> Optional[LatchFile]:
    transient_markers = [
        "remote end closed connection",
        "connection aborted",
        "connection reset",
        "timed out",
        "timeout",
        "temporarily unavailable",
        "service unavailable",
        "too many requests",
    ]

    def _is_transient(err: Exception) -> bool:
        cur = err
        while cur is not None:
            msg = f"{type(cur).__name__}: {cur}".lower()
            if any(marker in msg for marker in transient_markers):
                return True
            cur = cur.__cause__ or cur.__context__
        return False

    for attempt in range(1, retries + 1):
        try:
            files = [
                file for file in directory.iterdir()
                if isinstance(file, LatchFile)
                and Path(file.path).name == file_name
            ]
        except Exception as e:
            if not _is_transient(e):
                logging.error(
                    "Failed to list '%s' in '%s' (non-retryable): %s",
                    file_name,
                    directory.remote_path,
                    e,
                )
                return None

            is_last_attempt = attempt == retries
            if is_last_attempt:
                logging.error(
                    "Failed to list '%s' in '%s' after %d transient attempt(s): %s",
                    file_name,
                    directory.remote_path,
                    retries,
                    e,
                )
                return None

            logging.warning(
                "Attempt %d/%d failed to find file '%s' in '%s': %s. "
                "Retrying in %.1f seconds.",
                attempt,
                retries,
                file_name,
                directory.remote_path,
                e,
                retry_delay_s,
            )
            time.sleep(retry_delay_s)
            continue

        if len(files) == 1:
            return files[0]
        if len(files) == 0:
            logging.error(
                "No file '%s' found in '%s'.",
                file_name,
                directory.remote_path,
            )
            return None

        logging.error(
            "Multiple files named '%s' found in '%s'.",
            file_name,
            directory.remote_path,
        )
        return None

    return None
