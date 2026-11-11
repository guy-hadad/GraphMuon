"""Train a PyG GCN: python examples/pyg_node_classification.py --dataset Cora."""

import argparse

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.transforms import NormalizeFeatures

from graphmuon import GraphMuon


class GCN(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.first = GCNConv(inputs, 64, cached=True)
        self.second = GCNConv(64, outputs, cached=True)

    def forward(self, x, edges):
        x = F.dropout(self.first(x, edges).relu(), p=0.3, training=self.training)
        return self.second(x, edges)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["Cora", "CiteSeer", "PubMed"], default="Cora")
    parser.add_argument("--root", default="data")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    torch.manual_seed(0)
    dataset = Planetoid(args.root, args.dataset, transform=NormalizeFeatures())
    data = dataset[0].to(args.device)
    model = GCN(dataset.num_features, dataset.num_classes).to(args.device)
    optimizer = GraphMuon.from_model(model, lr=0.01, adamw_lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        output = model(data.x, data.edge_index)
        loss = F.cross_entropy(output[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        scheduler.step()
        if epoch % 20 == 0 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                prediction = model(data.x, data.edge_index).argmax(-1)
                accuracy = (prediction[data.val_mask] == data.y[data.val_mask]).float().mean()
            print(
                f"Epoch {epoch:3d}: loss={loss.item():.4f}, validation accuracy={accuracy.item():.3f}"
            )
    optimizer.remove_hooks()


if __name__ == "__main__":
    main()
