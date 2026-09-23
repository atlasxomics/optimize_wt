"""Memory-aware STAGATE training on disconnected sample graphs."""

import gc
import logging
import random

TRAINING_MODES = ("auto", "full", "sample_batched")


def estimate_training_bytes(n_cells, n_genes, n_edges, hidden_dims=(512, 30)):
    """Conservative float32 working-set heuristic, not a peak-memory guarantee.

    Include dense reconstruction/backward buffers, edge messages, activations,
    parameters/Adam state, and allocator/workspace headroom. Edges include loops.
    """
    hidden, latent = hidden_dims
    floats = (
        5 * n_cells * n_genes
        + 8 * n_cells * hidden
        + 6 * n_edges * hidden
        + 4 * n_cells * latent
        + 16 * (n_genes * hidden + hidden * latent)
    )
    return int(1.25 * 4 * floats + 1024**3)


def choose_training_mode(requested, estimated_bytes, free_bytes, n_samples):
    if requested not in TRAINING_MODES:
        raise ValueError(f"STAGATE training mode must be one of {TRAINING_MODES}.")
    if requested != "auto":
        return requested
    if n_samples > 1 and free_bytes is not None and estimated_bytes > 0.8 * free_bytes:
        return "sample_batched"
    return "full"


def sample_rows(adata):
    import numpy as np
    import pandas as pd

    if not adata.obs_names.is_unique:
        raise ValueError("STAGATE requires unique cell names.")
    if adata.n_obs == 0 or adata.n_vars == 0:
        raise ValueError("STAGATE requires nonempty cells and features.")
    if "sample" not in adata.obs:
        return [np.arange(adata.n_obs)]
    if adata.obs["sample"].isna().any():
        raise ValueError("STAGATE sample labels must not be missing.")
    codes, _ = pd.factorize(adata.obs["sample"], sort=False)
    return [np.flatnonzero(codes == i) for i in range(codes.max() + 1)]


def _prepare_batches(adata, rows):
    """Retain sparse CPU expression and remap each graph to local cell indices."""
    import numpy as np
    from scipy import sparse

    names = adata.obs_names
    net = adata.uns["Spatial_Net"]
    sources = names.get_indexer(net["Cell1"])
    targets = names.get_indexer(net["Cell2"])
    if (sources < 0).any() or (targets < 0).any():
        raise ValueError("Spatial_Net contains unknown cell names.")
    group = np.empty(adata.n_obs, dtype=np.int64)
    for i, indices in enumerate(rows):
        group[indices] = i
    if (group[sources] != group[targets]).any():
        raise ValueError("Sample batching cannot discard cross-sample spatial edges.")

    batches = []
    local = np.empty(adata.n_obs, dtype=np.int64)
    for i, indices in enumerate(rows):
        local[indices] = np.arange(len(indices))
        mask = group[sources] == i
        graph = sparse.coo_matrix(
            (np.ones(mask.sum()), (local[sources[mask]], local[targets[mask]])),
            shape=(len(indices), len(indices)),
        ).tocsr()
        # Match STAGATE's Transfer_pytorch_Data: collapse duplicates, add loops.
        graph = graph + sparse.eye(len(indices), format="csr")
        edges = np.asarray(graph.nonzero(), dtype=np.int64)
        x = adata.X[indices]
        x = x.astype(np.float32)
        batches.append((indices, x, edges))
    return batches


def _clear_attention(model):
    # STAGATE caches tensors on its convolution modules between forwards.
    for layer in model.modules():
        if hasattr(layer, "attentions"):
            layer.attentions = None
        if hasattr(layer, "_alpha"):
            layer._alpha = None


def _fit_batches(model, batches, device, n_cells, n_epochs=1000):
    import numpy as np
    import torch
    from scipy import sparse
    from tqdm import trange

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    def tensors(batch):
        _, values, edges = batch
        dense = values.toarray() if sparse.issparse(values) else values
        return (
            torch.as_tensor(dense, dtype=torch.float32, device=device),
            torch.as_tensor(edges, dtype=torch.long, device=device),
        )

    for _ in trange(n_epochs, desc="STAGATE (sample batches)"):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for batch in batches:
            x, edges = tensors(batch)
            z, reconstruction = model(x, edges)
            loss = torch.nn.functional.mse_loss(reconstruction, x) * (len(batch[0]) / n_cells)
            loss.backward()
            _clear_attention(model)
            del x, edges, z, reconstruction, loss
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

    model.eval()
    embedding = None
    with torch.no_grad():
        for batch in batches:
            x, edges = tensors(batch)
            z, reconstruction = model(x, edges)
            result = z.cpu().numpy()
            if embedding is None:
                embedding = np.empty((n_cells, result.shape[1]), dtype=result.dtype)
            embedding[batch[0]] = result
            _clear_attention(model)
            del x, edges, z, reconstruction, result
    return embedding


def _train_sample_batched(adata, rows, device, random_seed):
    import numpy as np
    import torch
    from STAGATE_pyG.STAGATE import STAGATE

    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    batches = _prepare_batches(adata, rows)
    model = STAGATE(hidden_dims=[adata.n_vars, 512, 30]).to(device)
    return _fit_batches(model, batches, device, adata.n_obs)


def train_embedding(adata, stagate, training_mode="auto", random_seed=0):
    """Return (embedding, actual mode); restart from seed after a full CUDA OOM."""
    import torch

    rows = sample_rows(adata)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    free = torch.cuda.mem_get_info(device)[0] if device.type == "cuda" else None
    estimated = estimate_training_bytes(
        adata.n_obs, adata.n_vars, len(adata.uns["Spatial_Net"]) + adata.n_obs
    )
    selected = choose_training_mode(training_mode, estimated, free, len(rows))
    logging.info(
        "STAGATE requested=%s selected=%s; estimated full peak=%.2f GiB; "
        "free GPU=%s GiB; samples=%d; largest sample=%d cells",
        training_mode, selected, estimated / 1024**3,
        "n/a" if free is None else f"{free / 1024**3:.2f}",
        len(rows), max(map(len, rows)),
    )
    if selected == "full":
        try:
            trained = stagate.train_STAGATE(
                adata, random_seed=random_seed, save_loss=False, device=device
            )
            return trained.obsm["STAGATE"].copy(), "full"
        except torch.cuda.OutOfMemoryError:
            if training_mode != "auto" or device.type != "cuda" or len(rows) < 2:
                raise
            logging.warning(
                "Full STAGATE training exhausted CUDA memory. Restarting from "
                "the original seed with sample batches."
            )
        # Outside except: release traceback references to the failed GPU model.
        adata.obsm.pop("STAGATE", None)
        gc.collect()
        torch.cuda.empty_cache()

    try:
        return _train_sample_batched(adata, rows, device, random_seed), "sample_batched"
    except torch.cuda.OutOfMemoryError as exc:
        raise RuntimeError(
            "A complete sample graph still exceeds GPU memory during STAGATE "
            "training. Reduce n_top_genes or use a GPU with more memory. "
            "Sample batching does not split individual sample graphs."
        ) from exc
