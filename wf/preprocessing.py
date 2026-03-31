import json
import importlib
import anndata
import scanpy as sc
import logging
import matplotlib.image as mpimg
import numpy as np
import squidpy as sq

from typing import Any, Dict, List, Union
from pathlib import Path

import pandas as pd
from scipy import io as sio
from scipy import sparse as sp

from wf.utils import get_LatchFile, Run, sanitize_condition


logging.basicConfig(
    format="%(levelname)s - %(asctime)s - %(message)s",
    level=logging.INFO
)


MTX_CANDIDATES = [
    "UniqueAndMult-EM.mtx",
    "UniqueAndMult-EM.mtx.gz",
    "matrix.mtx",
    "matrix.mtx.gz",
]
GENE_TABLE_CANDIDATES = [
    "features.tsv.gz",
    "features.tsv",
    "genes.tsv",
    "genes.tsv.gz",
]
BARCODE_CANDIDATES = [
    "barcodes.tsv",
    "barcodes.tsv.gz",
]
POSITION_CANDIDATES = [
    "tissue_positions_list.csv",
    "tissue_positions.csv",
]
SCALE_FACTOR_CANDIDATES = [
    "scalefactors_json.json",
]
SPATIAL_IMAGE_FILES = {
    "hires": ["tissue_hires_image.png"],
    "lowres": ["tissue_lowres_image.png"],
}
ALLOWED_HVG_FLAVORS = (
    "seurat",
    "cell_ranger",
    "seurat_v3",
    "seurat_v3_paper",
)
ALLOWED_CLUSTERING_BACKENDS = (
    "scanpy",
    "stagate",
)


def _resolve_local_file(
    directory: Union[Path, str], candidates: List[str]
) -> Path:
    directory = Path(directory)
    for candidate in candidates:
        candidate_path = directory / candidate
        if candidate_path.exists():
            return candidate_path

    raise FileNotFoundError(
        f"Could not find any of {candidates} in '{directory}'."
    )


def _resolve_spatial_file(run: Run, candidates: List[str]) -> Path:
    spatial_dir = Path(run.spatial_dir.local_path)

    try:
        return _resolve_local_file(spatial_dir, candidates)
    except FileNotFoundError:
        pass

    for candidate in candidates:
        latch_file = get_LatchFile(run.spatial_dir, candidate)
        if latch_file is not None:
            return Path(latch_file.local_path)

    raise FileNotFoundError(
        f"Unable to resolve any of {candidates} for run '{run.run_id}' in "
        f"'{run.spatial_dir.remote_path}'."
    )


def _resolve_optional_spatial_file(
    run: Run, candidates: List[str]
) -> Union[Path, None]:
    try:
        return _resolve_spatial_file(run, candidates)
    except FileNotFoundError:
        return None


def _deduplicate_gene_names(gene_names: pd.Series) -> List[str]:
    seen: dict[str, int] = {}
    unique_names: List[str] = []

    for gene_name in gene_names.astype(str):
        count = seen.get(gene_name, 0) + 1
        seen[gene_name] = count
        unique_names.append(
            f"{gene_name}.{count}" if count > 1 else gene_name
        )

    return unique_names


def _read_gene_table(gex_dir: Union[Path, str]) -> pd.DataFrame:
    genes_path = _resolve_local_file(gex_dir, GENE_TABLE_CANDIDATES)
    genes = pd.read_csv(genes_path, sep="\t", header=None)

    if genes.shape[1] == 0:
        raise ValueError(f"Gene table '{genes_path}' is empty.")

    gene_ids = genes.iloc[:, 0].astype(str)
    gene_names = genes.iloc[:, 1] if genes.shape[1] >= 2 else gene_ids.copy()

    return pd.DataFrame(
        {
            "gene_id": gene_ids.to_numpy(),
            "gene_name": _deduplicate_gene_names(gene_names),
        }
    )


def _read_barcodes(gex_dir: Union[Path, str]) -> pd.Index:
    barcodes_path = _resolve_local_file(gex_dir, BARCODE_CANDIDATES)
    barcodes = pd.read_csv(barcodes_path, header=None).iloc[:, 0].astype(str)

    return pd.Index(barcodes)


def _read_count_matrix(
    gex_dir: Union[Path, str],
    n_barcodes: int,
    n_genes: int
) -> sp.csr_matrix:
    matrix_path = _resolve_local_file(gex_dir, MTX_CANDIDATES)
    matrix = sp.csr_matrix(sio.mmread(str(matrix_path)))

    if matrix.shape == (n_genes, n_barcodes):
        return matrix.T.tocsr()

    if matrix.shape == (n_barcodes, n_genes):
        return matrix.tocsr()

    raise ValueError(
        f"Count matrix '{matrix_path}' shape {matrix.shape} does not match "
        f"{n_genes} genes and {n_barcodes} barcodes."
    )


def _read_positions(positions_file: Union[Path, str]) -> pd.DataFrame:
    positions = pd.read_csv(positions_file, header=None)
    column_names = ["barcode", "on_off", "row", "col", "xcor", "ycor"]

    if positions.shape[1] < len(column_names):
        raise ValueError(
            f"Spatial positions file '{positions_file}' must have at least "
            f"{len(column_names)} columns."
        )

    first_flag = str(positions.iloc[0, 1]).strip()
    if first_flag in {"0", "1"}:
        positions = positions.iloc[:, :len(column_names)].copy()
        positions.columns = column_names
    else:
        positions.columns = positions.iloc[0]
        positions = positions.iloc[1:].reset_index(drop=True)
        positions = positions.rename(
            columns={
                "in_tissue": "on_off",
                "array_row": "row",
                "array_col": "col",
                "pxl_row_in_fullres": "xcor",
                "pxl_col_in_fullres": "ycor",
            }
        )
        missing = [col for col in column_names if col not in positions.columns]
        if missing:
            raise ValueError(
                f"Spatial positions file '{positions_file}' is missing "
                f"columns {missing}."
            )
        positions = positions[column_names].copy()

    positions["barcode"] = positions["barcode"].astype(str)
    positions["on_off"] = positions["on_off"].astype(int)
    positions["row"] = positions["row"].astype(int)
    positions["col"] = positions["col"].astype(int)
    positions["xcor"] = positions["xcor"].astype(float)
    positions["ycor"] = positions["ycor"].astype(float)

    return positions.set_index("barcode")


def _ensure_rgb(image: np.ndarray) -> np.ndarray:
    """Convert grayscale/RGBA images into float32 RGB arrays."""

    image = np.array(image, dtype=np.float32)
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    elif image.ndim == 3 and image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]

    if image.size > 0 and image.max() > 1.0:
        image = image / 255.0

    return image


def _load_spatial_assets(run: Run) -> dict:
    """Load optional tissue images and scalefactors for image-backed plots."""

    spatial_uns: dict = {}

    scale_factor_path = _resolve_optional_spatial_file(run, SCALE_FACTOR_CANDIDATES)
    if scale_factor_path is not None:
        with open(scale_factor_path, "r", encoding="utf-8") as handle:
            spatial_uns["scalefactors"] = json.load(handle)

    images = {}
    for image_key, candidates in SPATIAL_IMAGE_FILES.items():
        image_path = _resolve_optional_spatial_file(run, candidates)
        if image_path is None:
            continue
        images[image_key] = _ensure_rgb(mpimg.imread(str(image_path)))

    if len(images) > 0:
        spatial_uns["images"] = images

    return spatial_uns


def _load_run_adata(run: Run) -> anndata.AnnData:
    gex_dir = Path(run.gex_dir.local_path)
    gene_table = _read_gene_table(gex_dir)
    barcodes = _read_barcodes(gex_dir)
    matrix = _read_count_matrix(
        gex_dir, n_barcodes=len(barcodes), n_genes=len(gene_table)
    )

    positions_file = _resolve_spatial_file(run, POSITION_CANDIDATES)
    positions = _read_positions(positions_file)
    in_tissue = positions[positions["on_off"] == 1]

    matched_barcodes = [
        barcode for barcode in in_tissue.index if barcode in barcodes
    ]
    if len(matched_barcodes) == 0:
        raise ValueError(
            f"No in-tissue barcodes from '{positions_file}' matched the count "
            f"matrix for run '{run.run_id}'."
        )

    barcode_to_idx = {barcode: idx for idx, barcode in enumerate(barcodes)}
    row_indices = [barcode_to_idx[barcode] for barcode in matched_barcodes]
    prefixed_barcodes = [
        f"{run.run_id}#{barcode}" for barcode in matched_barcodes
    ]

    adata = anndata.AnnData(
        X=matrix[row_indices, :],
        obs=pd.DataFrame(
            {
                "barcode": matched_barcodes,
                "on_off": in_tissue.loc[matched_barcodes, "on_off"].to_numpy(),
                "row": in_tissue.loc[matched_barcodes, "row"].to_numpy(),
                "col": in_tissue.loc[matched_barcodes, "col"].to_numpy(),
                "xcor": in_tissue.loc[matched_barcodes, "xcor"].to_numpy(),
                "ycor": in_tissue.loc[matched_barcodes, "ycor"].to_numpy(),
                "sample": run.run_id,
                "condition": sanitize_condition(run.condition),
            },
            index=prefixed_barcodes,
        ),
        var=pd.DataFrame(
            {"gene_id": gene_table["gene_id"].to_numpy()},
            index=pd.Index(gene_table["gene_name"], dtype="object"),
        ),
    )

    spatial_uns = _load_spatial_assets(run)
    if len(spatial_uns) > 0:
        adata.uns["spatial"] = {run.run_id: spatial_uns}

    return adata


def annotate_biotypes(adata: anndata.AnnData, genome: str) -> None:
    """Annotate common RNA biotypes from gene symbols."""

    gene_names = pd.Index(adata.var_names.astype(str))
    gene_names_upper = gene_names.str.upper()

    # Upper-casing gene symbols makes these prefix rules work for human, mouse,
    # and rat naming conventions (for example MT-/Mt-, RPS/Rps).
    adata.var["mt"] = gene_names_upper.str.startswith(("MT-",))
    adata.var["ribo"] = gene_names_upper.str.startswith(
        ("RPS", "RPL", "MRPS", "MRPL")
    )
    adata.var["lncrna"] = (
        gene_names_upper.str.contains("-AS", regex=False)
        | gene_names_upper.str.endswith("AS1")
        | gene_names_upper.str.startswith(("LINC", "NEAT", "MALAT"))
    )
    adata.var["mirna"] = gene_names_upper.str.startswith(("MIR", "MIRLET"))
    adata.var["sncrna"] = gene_names_upper.str.startswith(
        ("SNORD", "SNORA", "SNRP")
    )


def select_highly_variable_genes(
    adata: anndata.AnnData,
    n_top_genes: int = 4000,
    flavor: str = "seurat_v3",
) -> None:
    """Select HVGs from protein-coding plus lncRNA genes only."""

    if flavor not in ALLOWED_HVG_FLAVORS:
        raise ValueError(
            "Invalid hvg flavor "
            f"'{flavor}'. Expected one of {ALLOWED_HVG_FLAVORS}."
        )
    if flavor in {"seurat_v3", "seurat_v3_paper"}:
        try:
            import skmisc  # noqa: F401
        except ImportError as e:
            raise ImportError(
                f"hvg_flavor='{flavor}' requires scikit-misc to be installed."
            ) from e

    allowed = ~(
        adata.var["mt"]
        | adata.var["ribo"]
        | adata.var["mirna"]
        | adata.var["sncrna"]
    )
    n_allowed = int(allowed.sum())
    if n_allowed == 0:
        raise ValueError(
            "No genes remain after excluding mt, ribo, mirna, and sncrna "
            "features from HVG selection."
        )

    adata_hvg = adata[:, allowed].copy()

    if flavor in {"seurat_v3", "seurat_v3_paper"}:
        if "counts" not in adata.layers:
            raise ValueError(
                f"hvg_flavor='{flavor}' requires `adata.layers['counts']` to be "
                "available."
            )
        adata_hvg.X = adata.layers["counts"][:, allowed].copy()
    else:
        if "log1p" not in adata.layers:
            raise ValueError(
                f"hvg_flavor='{flavor}' requires `adata.layers['log1p']` to be "
                "available."
            )
        adata_hvg.X = adata.layers["log1p"][:, allowed].copy()

    batch_key = "sample" if "sample" in adata_hvg.obs.columns else None
    sc.pp.highly_variable_genes(
        adata_hvg,
        n_top_genes=min(n_top_genes, n_allowed),
        flavor=flavor,
        batch_key=batch_key,
    )

    hvg_names = adata_hvg.var_names[adata_hvg.var["highly_variable"]]
    adata.var["highly_variable"] = adata.var_names.isin(hvg_names)


def _cluster_from_embedding(
    adata: anndata.AnnData,
    use_rep: str,
    resolution: float,
    n_neighbors: int,
    min_dist: float,
    spread: float,
    merge_small_clusters: int = 0,
    random_state: int = 0,
) -> anndata.AnnData:
    sc.pp.neighbors(
        adata,
        n_neighbors=n_neighbors,
        use_rep=use_rep,
        random_state=random_state,
    )
    sc.tl.umap(
        adata,
        min_dist=min_dist,
        spread=spread,
        random_state=random_state,
    )
    sc.tl.leiden(
        adata,
        resolution=resolution,
        key_added="cluster",
        random_state=random_state,
    )

    if merge_small_clusters > 0:
        cluster_codes = adata.obs["cluster"].astype("category").cat.codes.to_numpy()
        merged_codes = _merge_small_clusters(
            cluster_codes,
            adata.obsm[use_rep],
            merge_small_clusters,
        )
        adata.obs["cluster"] = pd.Categorical(merged_codes.astype(str))

    return adata


def add_clusters(
    adata: anndata.AnnData,
    resolution: float,
    n_comps: int,
    n_neighbors: int,
    min_dist: float,
    spread: float,
    apply_harmony: bool = True,
    merge_small_clusters: int = 0,
    random_state: int = 0,
    pca_plot: bool = True,
) -> anndata.AnnData:
    """Perform dimensionality reduction, batch correction, umap, clustering.
    """

    # Dimensionality reduction
    sc.tl.pca(
        adata,
        n_comps=n_comps,
        use_highly_variable=True,
        random_state=random_state
    )
    if pca_plot:
        sc.pl.pca_variance_ratio(adata, n_pcs=n_comps, save=f"_{n_comps}_elbow")

    n_runs = 1
    try:
        n_runs = len(adata.obs["sample"].unique())
    except KeyError as e:
        logging.warning(
            f"Exception {e}: Please add metadata to combined AnnData."
        )

    if n_runs > 1 and apply_harmony:
        logging.info("Performing batch correction with Harmony...")
        sc.external.pp.harmony_integrate(adata, batch="sample")
        rep = "X_pca_harmony"
    else:
        rep = "X_pca"

    return _cluster_from_embedding(
        adata,
        use_rep=rep,
        resolution=resolution,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        spread=spread,
        merge_small_clusters=merge_small_clusters,
        random_state=random_state
    )


def _merge_small_clusters(
    labels: np.ndarray,
    embedding: np.ndarray,
    min_size: int
) -> np.ndarray:
    """Iteratively merge clusters smaller than `min_size`."""

    labels = np.asarray(labels, dtype=int).copy()
    embedding = np.asarray(embedding, dtype=float)

    while True:
        unique_labels, counts = np.unique(labels, return_counts=True)
        if len(unique_labels) <= 1 or counts.min() >= min_size:
            break

        smallest_idx = int(np.argmin(counts))
        smallest_label = unique_labels[smallest_idx]
        remaining_labels = unique_labels[unique_labels != smallest_label]

        centroids = {
            label: embedding[labels == label].mean(axis=0)
            for label in unique_labels
        }
        distances = [
            np.linalg.norm(centroids[smallest_label] - centroids[label])
            for label in remaining_labels
        ]
        nearest_label = remaining_labels[int(np.argmin(distances))]
        labels[labels == smallest_label] = nearest_label

    remap = {old_label: new_label for new_label, old_label in enumerate(np.unique(labels))}
    return np.array([remap[label] for label in labels], dtype=int)


def require_stagate_module():
    try:
        return importlib.import_module("STAGATE_pyG")
    except ImportError as e:
        raise ImportError(
            "clustering_backend='stagate' requires STAGATE_pyG to be installed "
            "in the workflow image."
        ) from e


def train_stagate_embedding(
    adata: anndata.AnnData,
    random_state: int = 0,
) -> anndata.AnnData:
    """Train STAGATE once on HVG log-normalized expression and store embedding."""

    STAGATE_pyG = require_stagate_module()

    if "highly_variable" not in adata.var:
        raise ValueError(
            "STAGATE backend requires `adata.var['highly_variable']` to be set."
        )
    if "log1p" not in adata.layers:
        raise ValueError(
            "STAGATE backend requires `adata.layers['log1p']` to be available."
        )

    adata_st = adata[:, adata.var["highly_variable"]].copy()
    adata_st.X = adata_st.layers["log1p"].copy()

    if "sample" in adata_st.obs.columns:
        nets = []
        for sample in adata_st.obs["sample"].astype(str).unique():
            adata_sub = adata_st[adata_st.obs["sample"] == sample].copy()
            STAGATE_pyG.Cal_Spatial_Net(
                adata_sub,
                rad_cutoff=None,
                k_cutoff=6,
                model="KNN",
            )
            nets.append(adata_sub.uns["Spatial_Net"])

        adata_st.uns["Spatial_Net"] = pd.concat(nets, ignore_index=True)
    else:
        STAGATE_pyG.Cal_Spatial_Net(
            adata_st,
            rad_cutoff=None,
            k_cutoff=6,
            model="KNN",
        )

    STAGATE_pyG.Stats_Spatial_Net(adata_st)

    train_kwargs = {
        "random_seed": random_state,
        "save_loss": False,
    }
    try:
        import torch

        train_kwargs["device"] = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
    except Exception:
        pass

    adata_st = STAGATE_pyG.train_STAGATE(adata_st, **train_kwargs)
    adata.obsm["X_stagate"] = adata_st.obsm["STAGATE"].copy()

    return adata


def save_stagate_embedding_checkpoint(
    adata: anndata.AnnData,
    output_path: Union[Path, str],
    metadata: Dict[str, Any],
) -> Path:
    """Write a minimal checkpoint containing the trained STAGATE embedding."""

    if "X_stagate" not in adata.obsm:
        raise ValueError(
            "Cannot save STAGATE checkpoint without `adata.obsm['X_stagate']`."
        )

    obs = pd.DataFrame(index=adata.obs_names.copy())
    if "sample" in adata.obs.columns:
        obs["sample"] = adata.obs["sample"].astype(str).values

    checkpoint = anndata.AnnData(
        X=sp.csr_matrix((adata.n_obs, 0), dtype=np.float32),
        obs=obs,
        var=pd.DataFrame(index=pd.Index([], dtype=str)),
    )
    checkpoint.obsm["X_stagate"] = np.asarray(adata.obsm["X_stagate"]).copy()
    checkpoint.uns["stagate_checkpoint_metadata_json"] = json.dumps(
        metadata,
        sort_keys=True,
    )

    output_path = Path(output_path)
    checkpoint.write(output_path)
    return output_path


def load_stagate_embedding_checkpoint(
    adata: anndata.AnnData,
    checkpoint_path: Union[Path, str],
    expected_metadata: Dict[str, Any],
) -> anndata.AnnData:
    """Load and validate a STAGATE embedding checkpoint."""

    checkpoint = anndata.read_h5ad(checkpoint_path)

    if "X_stagate" not in checkpoint.obsm:
        raise ValueError(
            f"Checkpoint '{checkpoint_path}' does not contain `X_stagate`."
        )

    if len(checkpoint.obs_names) != len(adata.obs_names):
        raise ValueError(
            "STAGATE checkpoint spot count does not match the current filtered "
            "AnnData."
        )

    if list(checkpoint.obs_names) != list(adata.obs_names):
        raise ValueError(
            "STAGATE checkpoint barcodes do not match the current filtered "
            "AnnData."
        )

    if "sample" in checkpoint.obs.columns and "sample" in adata.obs.columns:
        checkpoint_samples = checkpoint.obs["sample"].astype(str).tolist()
        current_samples = adata.obs["sample"].astype(str).tolist()
        if checkpoint_samples != current_samples:
            raise ValueError(
                "STAGATE checkpoint sample assignments do not match the "
                "current filtered AnnData."
            )

    stored_metadata_json = checkpoint.uns.get("stagate_checkpoint_metadata_json")
    if stored_metadata_json is None:
        raise ValueError(
            f"Checkpoint '{checkpoint_path}' is missing "
            "`uns['stagate_checkpoint_metadata_json']`."
        )
    stored_metadata = json.loads(stored_metadata_json)
    if stored_metadata != expected_metadata:
        raise ValueError(
            "STAGATE checkpoint preprocessing metadata does not match the "
            "current workflow inputs."
        )

    adata.obsm["X_stagate"] = np.asarray(checkpoint.obsm["X_stagate"]).copy()
    return adata


def add_stagate_clusters(
    adata: anndata.AnnData,
    resolution: float,
    n_neighbors: int,
    min_dist: float,
    spread: float,
    merge_small_clusters: int = 0,
    random_state: int = 0,
) -> anndata.AnnData:
    if "X_stagate" not in adata.obsm:
        raise ValueError(
            "STAGATE clustering requires `adata.obsm['X_stagate']`."
        )

    return _cluster_from_embedding(
        adata,
        use_rep="X_stagate",
        resolution=resolution,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        spread=spread,
        merge_small_clusters=merge_small_clusters,
        random_state=random_state,
    )


def add_metadata(
    run: Run, adata: anndata.AnnData, positions_file: Union[Path, str]
) -> anndata.AnnData:
    """Add metadata and spatial info .obs of AnnData.
    """

    # Read in tissue_positions file from spatial/
    positions = pd.read_csv(positions_file, header=None)
    positions.columns = ["barcode", "on_off", "row", "col", "xcor", "ycor"]

    # Merge fragments file with Anndata.obs
    adata.obs["barcode"] = adata.obs.index
    adata.obs = adata.obs.merge(positions, on="barcode", how="left")

    # Set run_id, condition
    adata.obs["sample"] = run.run_id
    adata.obs["condition"] = sanitize_condition(run.condition)

    # Ensure obs_names unique
    adata.obs_names = [
        run_id + "#" + bc for
        run_id, bc in zip(adata.obs["sample"], adata.obs["barcode"])
    ]

    return adata


def add_spatial(
    adata: anndata.AnnData, x_key: str = "xcor", y_key: str = "ycor"
) -> anndata.AnnData:
    """Add move x and y coordinates from .obs to .obsm["spatial"] for squidpy.
    """
    adata.obsm["spatial"] = adata.obs[[y_key, x_key]].values

    return adata


def add_spatial_neighbors(
    adata: anndata.AnnData,
    library_key: str = "sample"
) -> anndata.AnnData:
    """Build a within-sample spatial neighbor graph for coherence scoring.

    Prefer a grid graph when the input resembles Visium-style array positions.
    Fall back to generic spatial coordinates if grid graph construction fails.
    """

    try:
        sq.gr.spatial_neighbors(
            adata,
            coord_type="grid",
            n_rings=1,
            library_key=library_key,
        )
    except Exception as grid_error:
        logging.warning(
            "Failed to build grid spatial neighbors (%s). Falling back to "
            "generic spatial coordinates.",
            grid_error,
        )
        sq.gr.spatial_neighbors(
            adata,
            coord_type="generic",
            library_key=library_key,
        )

    return adata


def morans_i(
    connectivities: Union[sp.spmatrix, np.ndarray],
    values: np.ndarray
) -> float:
    """Compute Moran's I from a connectivity matrix and numeric vector."""

    if sp.issparse(connectivities):
        weights = connectivities.tocsr().astype(float)
    else:
        weights = sp.csr_matrix(np.asarray(connectivities, dtype=float))

    row_sums = np.asarray(weights.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1.0
    weights = sp.diags(1.0 / row_sums) @ weights

    centered = np.asarray(values, dtype=float) - np.mean(values)
    denom = float(centered @ centered)
    if denom == 0.0:
        return 0.0

    weighted_centered = weights @ centered

    return float(centered @ weighted_centered / denom)


def spatial_coherence_table(adata_dict: dict[str, anndata.AnnData]) -> pd.DataFrame:
    """Summarize Moran's I spatial coherence for each clustered AnnData."""

    rows = []
    for set_name, adata in adata_dict.items():
        if "spatial_connectivities" not in adata.obsp:
            raise KeyError(
                f"AnnData '{set_name}' is missing 'spatial_connectivities'."
            )
        if "cluster" not in adata.obs:
            raise KeyError(f"AnnData '{set_name}' is missing 'cluster' labels.")

        cluster_codes = (
            adata.obs["cluster"].astype("category").cat.codes.to_numpy(dtype=float)
        )
        rows.append(
            {
                "set": set_name,
                "n_clusters": int(adata.obs["cluster"].nunique()),
                "morans_I": round(
                    morans_i(adata.obsp["spatial_connectivities"], cluster_codes),
                    4,
                ),
            }
        )

    return pd.DataFrame(rows)


def calculate_qc(adata: anndata.AnnData, genome: str) -> None:
    annotate_biotypes(adata, genome)

    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=["mt", "ribo", "lncrna"],
        inplace=True,
        percent_top=None,
        log1p=False,
    )

    return None


def filter_adata(
    adata: anndata.AnnData,
    min_cells: int = 1,
    min_genes: int = 1,
    min_counts: int = 0,
    max_counts: int = 0,
    max_pct_mt: float = 100.0
) -> anndata.AnnData:
    """Filter AnnData by tissue status and basic QC thresholds.

    `max_counts == 0` disables high-count filtering.
    `max_pct_mt >= 100` disables mitochondrial filtering.
    """

    n_before = adata.n_obs

    # Filter 'off tissue' tixels
    try:
        adata = adata[adata.obs["on_off"] == 1].copy()
    except KeyError as e:
        logging.warning(
            f"Exception {e}: no positions data found in AnnData.obs"
        )

    sc.pp.filter_cells(adata, min_genes=min_genes)
    if min_counts > 0:
        sc.pp.filter_cells(adata, min_counts=min_counts)

    if max_counts > 0:
        adata = adata[adata.obs["total_counts"] <= max_counts].copy()

    if max_pct_mt < 100:
        if "pct_counts_mt" in adata.obs:
            adata = adata[adata.obs["pct_counts_mt"] <= max_pct_mt].copy()
        else:
            logging.warning(
                "pct_counts_mt not found in AnnData.obs; skipping mt filter."
            )

    sc.pp.filter_genes(adata, min_cells=min_cells)
    logging.info(
        f"Filtered observations from {n_before} to {adata.n_obs} using "
        f"min_genes={min_genes}, min_counts={min_counts}, "
        f"max_counts={max_counts}, max_pct_mt={max_pct_mt}."
    )

    return adata


def make_anndatas(runs: List[Run], genome: str) -> List[anndata.AnnData]:
    """Basic preprocessing for scanpy analysis; converts raw/ gex dir into a
    list of AnnData objects. QCs, metadata and spatial data are stored in
    AnnData.obs.
    """
    return [_load_run_adata(run) for run in runs]
