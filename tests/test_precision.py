import pytest
import torch

from graphmuon import GraphMuon


def test_pyg_gcn_preserves_float64_statistics():
    pytest.importorskip("torch_geometric")
    from torch_geometric.nn import GCNConv
    from torch_geometric.nn.conv.gcn_conv import gcn_norm

    torch.manual_seed(8)
    x = torch.randn(7, 4, dtype=torch.float64)
    edges = torch.tensor([[0, 1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6, 0]])
    layer = GCNConv(4, 3).double()
    optimizer = GraphMuon.from_model(layer, ns_dtype=torch.float64)
    layer(x, edges).square().mean().backward()
    optimizer.step()
    indices, weights = gcn_norm(edges, num_nodes=7, dtype=torch.float64)
    adjacency = torch.sparse_coo_tensor(indices.flip(0), weights, (7, 7))
    propagated = torch.sparse.mm(adjacency, x)
    expected = propagated.T @ propagated / 7
    actual = optimizer.state[layer.lin.weight]["covariance"]
    torch.testing.assert_close(actual, expected, rtol=1e-14, atol=1e-14)
