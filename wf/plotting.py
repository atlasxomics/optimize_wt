import anndata
import matplotlib.pyplot as plt
import scanpy as sc
import squidpy as sq

import os
from matplotlib.backends.backend_pdf import PdfPages
from typing import Callable
from typing import List
from typing import Optional


def _subset_for_sample_plot(adata: anndata.AnnData, sample: str) -> anndata.AnnData:
    sample_adata = adata[adata.obs["sample"] == sample].copy()
    if "spatial" in adata.uns and sample in adata.uns["spatial"]:
        sample_adata.uns["spatial"] = {sample: adata.uns["spatial"][sample]}

    return sample_adata


def _has_tissue_image(adata: anndata.AnnData, sample: str) -> bool:
    spatial_uns = adata.uns.get("spatial", {}).get(sample, {})
    return len(spatial_uns.get("images", {})) > 0


def _preferred_img_key(adata: anndata.AnnData, sample: str) -> Optional[str]:
    spatial_uns = adata.uns.get("spatial", {}).get(sample, {})
    images = spatial_uns.get("images", {})
    if "hires" in images:
        return "hires"
    if "lowres" in images:
        return "lowres"

    return None


def _plot_spatial(
    adata: anndata.AnnData,
    sample: str,
    color: str,
    ax,
    title: str,
    pt_size: float,
    categorical: bool,
) -> None:
    sample_adata = _subset_for_sample_plot(adata, sample)

    if _has_tissue_image(sample_adata, sample):
        plot_kwargs = {
            "adata": sample_adata,
            "color": color,
            "library_id": sample,
            "img_key": _preferred_img_key(sample_adata, sample),
            "spot_size": pt_size,
            "ax": ax,
            "show": False,
            "title": title,
        }
        if categorical:
            plot_kwargs["legend_loc"] = "right margin"
            plot_kwargs["bw"] = True

        sc.pl.spatial(**plot_kwargs)
        ax.set_axis_off()
        return

    if categorical:
        sq.pl.spatial_scatter(
            sample_adata,
            color=color,
            size=pt_size,
            shape=None,
            library_id=sample,
            ax=ax,
            title=title,
        )
        ax.set_axis_off()
        return

    sq.pl.spatial_scatter(
        sample_adata,
        color=color,
        size=pt_size,
        shape=None,
        ax=ax,
        library_id=sample,
        title=title,
        colorbar=False,
    )
    if len(ax.collections) > 0:
        ax.figure.colorbar(ax.collections[0], ax=ax, shrink=0.7)
    ax.set_axis_off()


def _get_page_saver(output_path: str) -> tuple[Callable, Callable, bool, List[str]]:
    """Return a page saver and closer for PDF or image outputs.

    For non-PDF outputs, figures are saved as numbered pages:
    `<output_stem>_001.<ext>`, `<output_stem>_002.<ext>`, ...
    """

    ext = os.path.splitext(output_path)[1].lower()
    page_idx = 1
    pdf = PdfPages(output_path) if ext == ".pdf" else None
    output_stem = os.path.splitext(output_path)[0] if ext else output_path
    output_ext = ext if ext else ".png"
    saved_paths: List[str] = []

    def save_page(fig):
        nonlocal page_idx
        if pdf is not None:
            pdf.savefig(fig)
            return

        image_path = f"{output_stem}_{page_idx:03d}{output_ext}"
        fig.savefig(
            image_path,
            dpi=200,
            bbox_inches="tight"
        )
        saved_paths.append(image_path)
        page_idx += 1

    def close():
        if pdf is not None:
            pdf.close()

    return save_page, close, pdf is not None, saved_paths


def _write_html_gallery(
    output_path: str,
    title: str,
    image_paths: List[str],
    captions: Optional[List[str]] = None,
    html_output_path: Optional[str] = None
) -> None:
    """Write a single HTML file that displays all image pages."""

    if len(image_paths) == 0:
        return

    html_path = (
        html_output_path
        if html_output_path is not None
        else f"{os.path.splitext(output_path)[0]}.html"
    )
    html_dir = os.path.dirname(html_path) or "."
    relative_paths = [os.path.relpath(path, start=html_dir) for path in image_paths]

    blocks = []
    for idx, rel_path in enumerate(relative_paths, start=1):
        caption = f"Page {idx}"
        if captions is not None and idx - 1 < len(captions):
            caption = captions[idx - 1]

        blocks.append(
            "<section class=\"page\">"
            f"<h2>{caption}</h2>"
            f"<img loading=\"lazy\" src=\"{rel_path}\" alt=\"{caption}\" />"
            "</section>"
        )

    body = "\n".join(blocks)
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <style>
    body {{
      margin: 24px auto;
      max-width: 1600px;
      padding: 0 16px 32px;
      background: #f4f4f4;
      color: #1f1f1f;
      font-family: Arial, sans-serif;
    }}
    h1 {{
      margin: 0 0 8px;
    }}
    p {{
      margin: 0 0 20px;
      color: #404040;
    }}
    .page {{
      margin: 0 0 24px;
      background: #fff;
      border: 1px solid #ddd;
      border-radius: 8px;
      padding: 12px;
    }}
    .page h2 {{
      margin: 0 0 10px;
      font-size: 18px;
    }}
    img {{
      width: 100%;
      height: auto;
      display: block;
    }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <p>{len(image_paths)} pages</p>
  {body}
</body>
</html>
"""
    with open(html_path, "w", encoding="utf-8") as fh:
        fh.write(html)


def combine_umaps(
    adata_dict: dict[str, anndata.AnnData],
    output_path: str,
    html_output_path: Optional[str] = None
) -> None:
    """Create a figure with UMAPs colored by categorical metadata.

    If `output_path` ends in `.pdf`, a multipage PDF is written.
    Otherwise, paginated image files are written.
    """

    sets = list(adata_dict.keys())
    page_captions: List[str] = []
    save_page, close, is_pdf, image_paths = _get_page_saver(output_path)
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

            page_captions.append("Sets: " + ", ".join(batch))
            save_page(fig)
            plt.close(fig)
    finally:
        close()
    if not is_pdf:
        _write_html_gallery(
            output_path,
            title="Combined UMAPs",
            image_paths=image_paths,
            captions=page_captions,
            html_output_path=html_output_path
        )


def combine_spatials(
    adata_dict: dict[str, anndata.AnnData],
    samples: List[str],
    output_path: str,
    pt_size: float = 5.0,
    html_output_path: Optional[str] = None
) -> None:
    """For each sample/condition, create a spatialdimplot colored by cluster.

    If `output_path` ends in `.pdf`, a multipage PDF is written.
    Otherwise, paginated image files are written.
    """

    sets = list(adata_dict.keys())
    page_captions: List[str] = []
    save_page, close, is_pdf, image_paths = _get_page_saver(output_path)
    try:
        for sample in samples:
            for i in range(0, len(sets), 4):

                batch = sets[i:i + 4]
                fig, axs = plt.subplots(2, 2, figsize=(10, 10))
                axs = axs.flatten()

                for j, s in enumerate(batch):
                    adata = adata_dict[s]
                    _plot_spatial(
                        color="cluster",
                        adata=adata,
                        sample=sample,
                        ax=axs[j],
                        title=f"{sample}: {s}",
                        pt_size=pt_size,
                        categorical=True,
                    )

                # Ensure empty plots are not displayed
                for k in range(len(axs)):
                    axs[k].axis("off")

                plt.tight_layout()

                page_captions.append(f"Sample {sample} | Sets: {', '.join(batch)}")
                save_page(fig)
                plt.close(fig)
    finally:
        close()
    if not is_pdf:
        _write_html_gallery(
            output_path,
            title="Combined Spatial Cluster Plots",
            image_paths=image_paths,
            captions=page_captions,
            html_output_path=html_output_path
        )


def plot_spatial_qc(
    adata: anndata.AnnData,
    samples: List[str],
    qc_metrics: List[str],
    output_path: str,
    pt_size: float = 25.0,
    html_output_path: Optional[str] = None
):
    """Generates a grid of spatial scatter plots for each sample and QC metric,
    saving them into a multipage PDF or paginated image files. Each row
    corresponds to a sample and each column to a QC metric in .obs.
    """

    rows_per_page = 3
    cols_per_page = len(qc_metrics)

    page_captions: List[str] = []
    save_page, close, is_pdf, image_paths = _get_page_saver(output_path)
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
                    _plot_spatial(
                        adata=adata,
                        sample=sample,
                        color=qc_metric,
                        ax=ax,
                        title=f"{sample} : {qc_metric}",
                        pt_size=pt_size,
                        categorical=False,
                    )

            plt.tight_layout()
            page_captions.append("Samples: " + ", ".join(sample_batch))
            save_page(fig)
            plt.close(fig)
    finally:
        close()
    if not is_pdf:
        _write_html_gallery(
            output_path,
            title="Spatial QC Plots",
            image_paths=image_paths,
            captions=page_captions,
            html_output_path=html_output_path
        )


def plot_spatial_coherence(
    coherence_df,
    output_path: str,
) -> None:
    """Plot Moran's I spatial coherence versus number of clusters."""

    if coherence_df.empty:
        return

    plot_df = coherence_df.sort_values(["n_clusters", "morans_I"]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(
        plot_df["n_clusters"],
        plot_df["morans_I"],
        s=40,
        color="steelblue",
        edgecolors="black",
        linewidths=0.4,
    )
    ax.plot(
        plot_df["n_clusters"],
        plot_df["morans_I"],
        color="steelblue",
        alpha=0.5,
        linewidth=1.0,
    )
    ax.set_xlabel("Number of clusters")
    ax.set_ylabel("Moran's I")
    ax.set_title("Spatial coherence across parameter sets")
    ax.grid(alpha=0.2, linewidth=0.5)

    best_idx = plot_df["morans_I"].idxmax()
    best = plot_df.loc[best_idx]
    ax.annotate(
        f"Best: {best['set']}",
        (best["n_clusters"], best["morans_I"]),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=8,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
