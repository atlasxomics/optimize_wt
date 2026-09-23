"""CPU regressions for graph batching, numerical equivalence, and OOM routing."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
import torch

# Avoid importing the Latch workflow just to test its numerical helper.
spec = importlib.util.spec_from_file_location(
    "stagate_training", Path(__file__).parents[1] / "wf" / "stagate_training.py"
)
training = importlib.util.module_from_spec(spec)
spec.loader.exec_module(training)


def fixture_adata():
    data = ad.AnnData(
        sparse.csr_matrix(np.arange(15, dtype=np.float32).reshape(5, 3) / 10),
        obs=pd.DataFrame({"sample": ["b", "a", "b", "a", "b"]}, index=list("ABCDE")),
    )
    data.uns["Spatial_Net"] = pd.DataFrame({
        "Cell1": ["A", "C", "B", "D", "A"],
        "Cell2": ["C", "E", "D", "B", "C"],
    })
    return data


def test_estimate_routes_reported_run_to_batches():
    estimated = training.estimate_training_bytes(387827, 4000, 387827 * 5)
    assert training.choose_training_mode("auto", estimated, 44.42 * 1024**3, 11) == "sample_batched"
    assert training.choose_training_mode("auto", 1024**3, 40 * 1024**3, 2) == "full"
    assert training.choose_training_mode("auto", estimated, None, 11) == "full"
    assert training.choose_training_mode("auto", estimated, 1024**3, 1) == "full"
    assert training.choose_training_mode("full", estimated, 1024**3, 11) == "full"
    with pytest.raises(ValueError):
        training.choose_training_mode("typo", 0, None, 2)


def test_graph_remapping_loops_and_duplicate_edges():
    data = fixture_adata()
    batches = training._prepare_batches(data, training.sample_rows(data))
    np.testing.assert_array_equal(batches[0][0], [0, 2, 4])
    np.testing.assert_array_equal(batches[1][0], [1, 3])
    assert set(map(tuple, batches[0][2].T)) == {(0, 0), (1, 1), (2, 2), (0, 1), (1, 2)}
    np.testing.assert_array_equal(batches[0][1].toarray(), data.X[[0, 2, 4]].toarray())
    data.uns["Spatial_Net"].loc[0, "Cell2"] = "B"
    with pytest.raises(ValueError, match="cross-sample"):
        training._prepare_batches(data, training.sample_rows(data))


def test_invalid_labels_and_unknown_cells():
    data = fixture_adata()
    data.obs.loc["A", "sample"] = None
    with pytest.raises(ValueError, match="missing"):
        training.sample_rows(data)
    data = fixture_adata()
    data.uns["Spatial_Net"].loc[0, "Cell2"] = "unknown"
    with pytest.raises(ValueError, match="unknown"):
        training._prepare_batches(data, training.sample_rows(data))


class GraphAutoencoder(torch.nn.Module):
    """Small graph-dependent model to exercise the actual accumulation loop."""
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(3, 2)
        self.decoder = torch.nn.Linear(2, 3)

    def forward(self, x, edges):
        h = self.encoder(x)
        z = torch.zeros_like(h).index_add(0, edges[1], h[edges[0]])
        return z, self.decoder(z)


def compare_full_and_batched(model):
    import copy
    data = fixture_adata()
    batches = training._prepare_batches(data, training.sample_rows(data))
    full = training._prepare_batches(data, [np.arange(data.n_obs)])[0]
    reference = copy.deepcopy(model)
    x = torch.tensor(full[1].toarray())
    edges = torch.tensor(full[2])
    optimizer = torch.optim.Adam(reference.parameters(), lr=0.001, weight_decay=0.0001)
    for _ in range(3):
        reference.train()
        optimizer.zero_grad(set_to_none=True)
        _, out = reference(x, edges)
        torch.nn.functional.mse_loss(out, x).backward()
        torch.nn.utils.clip_grad_norm_(reference.parameters(), 5.0)
        optimizer.step()
    reference.eval()
    with torch.no_grad():
        expected = reference(x, edges)[0].numpy()
    actual = training._fit_batches(model, batches, torch.device("cpu"), data.n_obs, n_epochs=3)
    np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-5)
    for p, q in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(p, q, rtol=2e-4, atol=2e-5)


def test_accumulation_and_original_cell_order():
    torch.manual_seed(42)
    compare_full_and_batched(GraphAutoencoder())


def test_actual_stagate_equivalence():
    pytest.importorskip("STAGATE_pyG")
    from STAGATE_pyG.STAGATE import STAGATE
    torch.manual_seed(42)
    compare_full_and_batched(STAGATE(hidden_dims=[3, 8, 2]))


@pytest.mark.parametrize("mode,error,fallback", [
    ("auto", torch.cuda.OutOfMemoryError, True),
    ("full", torch.cuda.OutOfMemoryError, False),
    ("auto", ValueError, False),
])
def test_full_failure_routing(monkeypatch, mode, error, fallback):
    data = fixture_adata()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (100 * 1024**3, 100 * 1024**3))
    empty = Mock()
    monkeypatch.setattr(torch.cuda, "empty_cache", empty)
    expected = np.zeros((5, 2), dtype=np.float32)
    batched = Mock(return_value=expected)
    monkeypatch.setattr(training, "_train_sample_batched", batched)
    upstream = SimpleNamespace(train_STAGATE=Mock(side_effect=error("test failure")))
    if fallback:
        result, actual_mode = training.train_embedding(data, upstream, mode, random_seed=17)
        assert result is expected
        assert actual_mode == "sample_batched"
        empty.assert_called_once()
        assert batched.call_args.args[-1] == 17
    else:
        with pytest.raises(error):
            training.train_embedding(data, upstream, mode)
        batched.assert_not_called()
        empty.assert_not_called()


def test_explicit_batching_bypasses_full(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    batched = Mock(return_value=np.zeros((5, 2)))
    monkeypatch.setattr(training, "_train_sample_batched", batched)
    upstream = SimpleNamespace(train_STAGATE=Mock())
    _, selected = training.train_embedding(fixture_adata(), upstream, "sample_batched")
    assert selected == "sample_batched"
    upstream.train_STAGATE.assert_not_called()


def test_single_sample_oom_does_not_retry(monkeypatch):
    data = fixture_adata()
    data.obs["sample"] = "one"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (1024**3, 1024**3))
    batched = Mock()
    monkeypatch.setattr(training, "_train_sample_batched", batched)
    upstream = SimpleNamespace(train_STAGATE=Mock(side_effect=torch.cuda.OutOfMemoryError()))
    with pytest.raises(torch.cuda.OutOfMemoryError):
        training.train_embedding(data, upstream)
    batched.assert_not_called()


def test_full_success_preserves_upstream_defaults(monkeypatch):
    data = fixture_adata()
    data.obsm["STAGATE"] = np.ones((5, 2), dtype=np.float32)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    upstream = SimpleNamespace(train_STAGATE=Mock(return_value=data))
    embedding, mode = training.train_embedding(data, upstream, random_seed=23)
    assert mode == "full"
    np.testing.assert_array_equal(embedding, data.obsm["STAGATE"])
    assert embedding is not data.obsm["STAGATE"]
    upstream.train_STAGATE.assert_called_once_with(
        data, random_seed=23, save_loss=False, device=torch.device("cpu")
    )
