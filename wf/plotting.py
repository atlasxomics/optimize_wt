import anndata
import matplotlib.pyplot as plt
import scanpy as sc
import squidpy as sq

from matplotlib.backends.backend_pdf import PdfPages
from typing import List


def combine_umaps(
    adata_dict: dict[str, anndata.AnnData], output_path: str
) -> None:
    """Create a figure with UMAPs colored categorical metadata.
    """

    sets = list(adata_dict.keys())
    with PdfPages(output_path) as pdf:
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

            pdf.savefig(fig)
        plt.close(fig)


def combine_spatials(
    adata_dict: dict[str, anndata.AnnData],
    samples: List[str],
    output_path: str,
    pt_size: float = 5.0
) -> None:
    """For each sample/condition, create a spatialdimplot colored by cluster.
    """

    sets = list(adata_dict.keys())
    with PdfPages(output_path) as pdf:
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

                pdf.savefig(fig)
        plt.close(fig)


def plot_spatial_qc(
    adata: anndata.AnnData,
    samples: List[str],
    qc_metrics: List[str],
    output_path: str,
    pt_size: float = 25.0
):
    """Generates a grid of spatial scatter plots for each sample and QC metric,
    saving them into a PDF.  Each row corresponds to a sample and each column
    to a QC metric in .obs.
    """

    rows_per_page = 3
    cols_per_page = len(qc_metrics)

    with PdfPages(output_path) as pdf:
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
            pdf.savefig(fig)
            plt.close(fig)
