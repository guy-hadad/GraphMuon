import pytest
import torch
from torch import nn

from graphmuon import GraphMuon

pyg = pytest.importorskip("torch_geometric")
from torch_geometric.nn import GATConv, GCNConv, GINConv, GINEConv, SAGEConv  # noqa: E402
from torch_geometric.nn.conv.gcn_conv import gcn_norm  # noqa: E402


@pytest.mark.parametrize("kind", ["gcn", "sage", "gin", "gine", "gat"])
def test_auto_observers_train_and_ignore_evaluation(kind):
    torch.manual_seed(1)
    x = torch.randn(8, 4)
    edges = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7, 0]])
    mlp = nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.Linear(6, 3))
    layer = {
        "gcn": lambda: GCNConv(4, 3),
        "sage": lambda: SAGEConv(4, 3),
        "gin": lambda: GINConv(mlp),
        "gine": lambda: GINEConv(mlp, edge_dim=2),
        "gat": lambda: GATConv(4, 3, heads=2),
    }[kind]()
    optimizer = GraphMuon.from_model(layer)
    assert len(optimizer.graph_parameters) == 1
    args = (x, edges, torch.randn(8, 2)) if kind == "gine" else (x, edges)
    optimizer.zero_grad()
    layer(*args).square().mean().backward()
    optimizer.step()
    parameter = optimizer.graph_parameters[0]
    covariance = optimizer.state[parameter]["covariance"]
    assert covariance.shape == (4, 4)
    if kind == "gcn":
        normalized_edges, weights = gcn_norm(edges, num_nodes=8, dtype=x.dtype)
        adjacency = torch.sparse_coo_tensor(normalized_edges.flip(0), weights, (8, 8))
        propagated = torch.sparse.mm(adjacency, x)
        torch.testing.assert_close(covariance, propagated.T @ propagated / 8)
    assert all(torch.isfinite(p).all() for p in layer.parameters())
    layer.eval()
    with torch.no_grad():
        layer(*args)
    assert not optimizer._pending
    optimizer.remove_hooks()


def test_embedding_and_bias_groups_use_scheduled_adamw():
    model = nn.ModuleList([nn.Embedding(5, 3), GCNConv(3, 2)])
    optimizer = GraphMuon.from_model(model, adamw_lr=0.003)
    group = next(g for g in optimizer.param_groups if g["algorithm"] == "adamw")
    assert any(p is model[0].weight for p in group["params"])
    assert any(p is model[1].bias for p in group["params"])
    assert group["lr"] == 0.003


@pytest.mark.parametrize("edge_features", [False, True])
def test_pyg_mlp_inside_gin_and_gine(edge_features):
    from torch_geometric.nn import MLP

    mlp = MLP([4, 6, 3], norm=None)
    layer = GINEConv(mlp, edge_dim=2) if edge_features else GINConv(mlp)
    optimizer = GraphMuon.from_model(layer)
    assert len(optimizer.graph_parameters) == 1
    assert optimizer.graph_parameters[0] is mlp.lins[0].weight
    x = torch.randn(8, 4)
    edges = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7, 0]])
    args = (x, edges, torch.randn(8, 2)) if edge_features else (x, edges)
    layer(*args).square().mean().backward()
    optimizer.step()
    assert optimizer.state[mlp.lins[0].weight]["covariance"].shape == (4, 4)
    optimizer.remove_hooks()


@pytest.mark.parametrize("kind", ["gcn", "sage", "gin", "gine", "gat"])
def test_frozen_graph_layers_are_excluded(kind):
    mlp = nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.Linear(6, 3))
    frozen = {
        "gcn": lambda: GCNConv(4, 3),
        "sage": lambda: SAGEConv(4, 3),
        "gin": lambda: GINConv(mlp),
        "gine": lambda: GINEConv(mlp, edge_dim=2),
        "gat": lambda: GATConv(4, 3),
    }[kind]()
    frozen.requires_grad_(False)
    trainable = GCNConv(3, 2)
    optimizer = GraphMuon.from_model(nn.ModuleList([frozen, trainable]))
    assert len(optimizer.graph_parameters) == 1
    assert optimizer.graph_parameters[0] is trainable.lin.weight
    assert all(p.requires_grad for group in optimizer.param_groups for p in group["params"])
