"""Optional observers for PyG 2.7+; imported only by from_model()."""

from __future__ import annotations

import warnings

import torch
from torch import nn


def attach(optimizer, model):
    try:
        from torch_geometric.nn import GATConv, GCNConv, GINConv, GINEConv, SAGEConv
        from torch_geometric.nn.conv.gcn_conv import gcn_norm
        from torch_geometric.nn.dense.linear import Linear as PyGLinear
    except ImportError as error:
        raise ImportError("from_model() requires pip install 'graphmuon[pyg]'") from error

    def active(layer, parameter):
        return layer.training and torch.is_grad_enabled() and optimizer.needs_observation(parameter)

    def gcn_observer(layer, args, kwargs):
        if not active(layer, layer.lin.weight):
            return
        x = args[0] if args else kwargs["x"]
        edges = args[1] if len(args) > 1 else kwargs["edge_index"]
        weights = args[2] if len(args) > 2 else kwargs.get("edge_weight")
        if not isinstance(edges, torch.Tensor) or edges.layout != torch.strided:
            raise ValueError(
                "GCN auto-observer requires COO edge_index [2,E]; use observe() for sparse adjacency"
            )
        with torch.no_grad(), torch.autocast(device_type=x.device.type, enabled=False):
            dtype = torch.float64 if layer.lin.weight.dtype == torch.float64 else torch.float32
            x = x.detach().to(dtype)
            if layer.normalize:
                if layer._cached_edge_index is not None:
                    edges, weights = layer._cached_edge_index
                else:
                    edges, weights = gcn_norm(
                        edges,
                        weights,
                        x.shape[0],
                        layer.improved,
                        layer.add_self_loops,
                        layer.flow,
                        x.dtype,
                    )
            if weights is None:
                weights = x.new_ones(edges.shape[1])
            source, target = edges if layer.flow == "source_to_target" else edges.flip(0)
            adjacency = torch.sparse_coo_tensor(
                torch.stack((target, source)),
                weights.to(x.dtype),
                (x.shape[0], x.shape[0]),
                device=x.device,
            )
            optimizer.observe(layer.lin.weight, torch.sparse.mm(adjacency, x))

    def attach_gat(layer):
        if layer.lin is not None and layer.lin.weight not in optimizer._groups:
            return
        if layer.lin is None or layer.flow != "source_to_target":
            warnings.warn(
                "Bipartite/reversed GAT requires manual observe(); using Muon", stacklevel=3
            )
            return
        parameter = layer.lin.weight
        optimizer.register_graph(parameter)
        cache = {}

        def remember(module, args, kwargs):
            cache.clear()
            if active(module, parameter):
                x = args[0] if args else kwargs["x"]
                if not isinstance(x, torch.Tensor):
                    raise ValueError("GAT auto-observer requires homogeneous tensor inputs")
                cache["x"] = x.detach()

        def propagated(module, args):
            if "x" not in cache:
                return
            edges, _size, inputs = args
            x = cache["x"]
            if not isinstance(edges, torch.Tensor) or edges.layout != torch.strided:
                raise ValueError("GAT auto-observer requires COO edge_index [2,E]")
            # Approximation: average frozen per-head attention, including dropout.
            # The actual attention Jacobian is intentionally outside this metric.
            with torch.no_grad(), torch.autocast(device_type=x.device.type, enabled=False):
                dtype = torch.float64 if parameter.dtype == torch.float64 else torch.float32
                weights = inputs["alpha"].detach().to(dtype).mean(-1)
                adjacency = torch.sparse_coo_tensor(
                    edges.flip(0),
                    weights,
                    (x.shape[0], x.shape[0]),
                    device=x.device,
                )
                optimizer.observe(parameter, torch.sparse.mm(adjacency, x.to(dtype)))

        def release(module, args, output):
            cache.clear()

        optimizer._handles.extend(
            [
                layer.register_forward_pre_hook(remember, with_kwargs=True),
                layer.register_propagate_forward_pre_hook(propagated),
                layer.register_forward_hook(release),
            ]
        )

    for layer in model.modules():
        if isinstance(layer, GCNConv):
            if layer.lin.weight not in optimizer._groups:
                continue
            optimizer.register_graph(layer.lin.weight)
            optimizer._handles.append(
                layer.register_forward_pre_hook(gcn_observer, with_kwargs=True)
            )
        elif isinstance(layer, SAGEConv):
            if layer.lin_l.weight in optimizer._groups:
                optimizer.watch(layer.lin_l)
        elif isinstance(layer, (GINConv, GINEConv)):
            # Hook the actual matrix input, after aggregation and edge processing.
            first = next(
                (m for m in layer.nn.modules() if isinstance(m, (nn.Linear, PyGLinear))), None
            )
            if first is None:
                warnings.warn(
                    "GIN/GINE has no supported linear layer: use manual observe()", stacklevel=2
                )
            elif first.weight in optimizer._groups:
                optimizer.watch(first)
        elif isinstance(layer, GATConv):
            attach_gat(layer)
    if not optimizer.graph_parameters:
        warnings.warn(
            "No supported graph layers found; call watch() or observe() for GraphMuon", stacklevel=2
        )
