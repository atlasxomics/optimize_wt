import anndata
import matplotlib.pyplot as plt
import scanpy as sc
import squidpy as sq

import os
from matplotlib.backends.backend_pdf import PdfPages
from typing import Callable
from typing import List


def _get_page_saver(output_path: str) -> tuple[Callable, Callable]:
    """Return a page saver and closer for PDF or image outputs.

    For non-PDF outputs, figures are saved as numbered pages:
    `<output_stem>_001.<ext>`, `<output_stem>_002.<ext>`, ...
    """

    ext = os.path.splitext(output_path)[1].lower()
    page_idx = 1
    pdf = PdfPages(output_path) if ext == ".pdf" else None
    output_stem = os.path.splitext(output_path)[0] if ext else output_path
    output_ext = ext if ext else ".png"

    def save_page(fig):
        nonlocal page_idx
        if pdf is not None:
            pdf.savefig(fig)
            return

        fig.savefig(
            f"{output_stem}_{page_idx:03d}{output_ext}",
            dpi=200,
            bbox_inches="tight"
        )
        page_idx += 1

    def close():
        if pdf is not None:
            pdf.close()

    return save_page, close


def combine_umaps(
    adata_dict: dict[str, anndata.AnnData], output_path: str
) -> None:
    """Create a figure with UMAPs colored by categorical metadata.

    If `output_path` ends in `.pdf`, a multipage PDF is written.
    Otherwise, paginated image files are written.
    """

    sets = list(adata_dict.keys())
    save_page, close = _get_page_saver(output_path)
    try:
        for i in range(0, len(sets), 4):

            batch = sets[i:i + 4]
            fig, axs = plt.subplots(2, 2, figsize=(12, 12))
            axs = axs.flatten()

            for j, s in enumerate(batch):
                sc.pl.umap(
                    adata_dict[s],
                    s=10,
                    color="cluster",
                    ax=axs[j],
                    show=False,
                    title=s
                )

            # Ensure empty plots are not displayed
            for k in range(len(axs)):
                axs[k].axis("off")

            plt.tight_layout()

            save_page(fig)
            plt.close(fig)
    finally:
        close()


def combine_spatials(
    adata_dict: dict[str, anndata.AnnData],
    samples: List[str],
    output_path: str,
    pt_size: float = 5.0
) -> None:
    """For each sample/condition, create a spatialdimplot colored by cluster.

    If `output_path` ends in `.pdf`, a multipage PDF is written.
    Otherwise, paginated image files are written.
    """

    sets = list(adata_dict.keys())
    save_page, close = _get_page_saver(output_path)
    try:
        for sample in samples:
            for i in range(0, len(sets), 4):

                batch = sets[i:i + 4]
                fig, axs = plt.subplots(2, 2, figsize=(10, 10))
                axs = axs.flatten()

                for j, s in enumerate(batch):
                    adata = adata_dict[s]
                    sq.pl.spatial_scatter(
                        adata[adata.obs["sample"] == sample],
                        color="cluster",
                        size=pt_size,
                        shape=None,
                        library_id=sample,
                        ax=axs[j],
                        title=f"{sample}: {s}"
                    )
                    axs[j].axis("off")

                # Ensure empty plots are not displayed
                for k in range(len(axs)):
                    axs[k].axis("off")

                plt.tight_layout()

                save_page(fig)
                plt.close(fig)
    finally:
        close()


def plot_spatial_qc(
    adata: anndata.AnnData,
    samples: List[str],
    qc_metrics: List[str],
    output_path: str,
    pt_size: float = 25.0
):
    """Generates a grid of spatial scatter plots for each sample and QC metric,
    saving them into a multipage PDF or paginated image files. Each row
    corresponds to a sample and each column to a QC metric in .obs.
    """

    rows_per_page = 3
    cols_per_page = len(qc_metrics)

    save_page, close = _get_page_saver(output_path)
    try:
        for i in range(0, len(samples), rows_per_page):

            sample_batch = samples[i:i + rows_per_page]

            # Create a figure for the current page
            fig, axs = plt.subplots(
                len(sample_batch),
                cols_per_page,
                figsize=(cols_per_page * 5, len(sample_batch) * 5)
            )

            # If  one sample, make axs a list
            if len(sample_batch) == 1:
                axs = [axs]

            for row_idx, sample in enumerate(sample_batch):
                for col_idx, qc_metric in enumerate(qc_metrics):

                    ax = axs[row_idx][col_idx]
                    sq.pl.spatial_scatter(
                        adata[adata.obs['sample'] == sample],
                        color=qc_metric,
                        size=pt_size,
                        shape=None,
                        ax=ax,
                        library_id=sample,
                        title=f"{sample} : {qc_metric}",
                        colorbar=False
                    )
                    cbar = fig.colorbar(ax.collections[0], ax=ax, shrink=0.7)

            plt.tight_layout()
            save_page(fig)
            plt.close(fig)
    finally:
        close()
