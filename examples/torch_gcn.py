"""Dependency-free usage demo: python examples/torch_gcn.py --device cpu."""

import argparse

import torch
from torch import nn

from graphmuon import GraphMuon


class GCN(nn.Module):
    def __init__(self, features, hidden, classes):
        super().__init__()
        self.first = nn.Linear(features, hidden)
        self.second = nn.Linear(hidden, classes)

    def forward(self, x, adjacency):
        x = self.first(torch.sparse.mm(adjacency, x)).relu()
        return self.second(torch.sparse.mm(adjacency, x))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    torch.manual_seed(0)
    device = torch.device(args.device)
    nodes = 64
    source = torch.arange(nodes, device=device)
    edges = torch.stack((source.repeat(3), torch.cat((source, source.roll(1), source.roll(-1)))))
    adjacency = torch.sparse_coo_tensor(
        edges, torch.full((3 * nodes,), 1 / 3, device=device), (nodes, nodes)
    ).coalesce()
    x = torch.randn(nodes, 8, device=device)
    labels = (x[:, 0] > 0).long()
    model = GCN(8, 16, 2).to(device)
    optimizer = GraphMuon(model.parameters(), lr=0.01)
    optimizer.watch(model.first)
    optimizer.watch(model.second)
    for _ in range(args.steps):
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(model(x, adjacency), labels)
        loss.backward()
        optimizer.step()
    print(f"Final demo loss: {loss.item():.4f}")
    optimizer.remove_hooks()


if __name__ == "__main__":
    main()
