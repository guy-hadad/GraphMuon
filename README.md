# GraphMuon

**A compact PyTorch optimizer for graph neural networks.**

Official implementation accompanying [Module-Aware Optimization for Graph Neural Networks](https://arxiv.org/abs/2611.08379), NeurIPS 2026, by Guy Hadad, Haggai Roitman, Moshe Eliasof, and Bracha Shapira.

GraphMuon adapts Muon's matrix updates using the features a graph layer actually consumes. This package provides the optimizer and optional PyTorch Geometric observers, with the usual `zero_grad()`, `backward()`, `step()`, parameter groups, schedulers, and checkpoint interface.

```python
from graphmuon import GraphMuon

optimizer = GraphMuon.from_model(model, lr=0.02)  # Supported PyG model
optimizer.zero_grad()
loss = criterion(model(data.x, data.edge_index), target)
loss.backward()
optimizer.step()
```

## Installation

Install PyTorch for your hardware first using the [PyTorch installation guide](https://pytorch.org/get-started/locally/), then install from GitHub:

```bash
pip install git+https://github.com/guy-hadad/GraphMuon.git
```

For automatic PyG integration:

```bash
pip install 'graphmuon[pyg] @ git+https://github.com/guy-hadad/GraphMuon.git'
```

For development:

```bash
git clone https://github.com/guy-hadad/GraphMuon.git
cd GraphMuon
pip install -e '.[pyg,dev]'
pytest -q
```

Requires Python 3.10+ and PyTorch 2.2+. The optional adapters target PyG 2.7–2.x. The release was exercised with PyTorch 2.11.0 and PyG 2.7.0 on CPU and NVIDIA RTX 6000 Ada GPUs. GraphMuon implements its own Newton–Schulz iteration; it does not require native `torch.optim.Muon`. These commands do not assume a PyPI release.

## Choose an integration

| Model | Setup | Parameters receiving GraphMuon |
|---|---|---|
| Supported PyG GNN | `GraphMuon.from_model(model)` | Supported graph-conditioned matrices |
| Linear module after graph aggregation | `optimizer.watch(linear)` | That module's weight |
| Custom graph operator or weight layout | `optimizer.observe(weight, X_g, input_axis=...)` | The explicitly observed matrix |
| Other matrices in the same optimizer | No observation | Ordinary Muon direction |
| Biases and other non-matrix tensors | Automatic AdamW branch | AdamW |
| Embeddings | Automatic AdamW in `from_model()`; explicit group otherwise | AdamW |

**Graph information is required.** `GraphMuon(model.parameters())` cannot infer a graph from gradients. Supply propagated inputs using `from_model()`, `watch()`, or `observe()`. Declared graph matrices raise an error if a required observation is missing; unobserved matrices use the Muon branch.

Move the model to its final device before constructing the optimizer. Materialize lazy layers with a dry forward first. Keep one active set of optimizer observers per model; call `remove_hooks()` before replacing the optimizer.

## Automatic PyG setup

```python
import torch
from graphmuon import GraphMuon

model = model.to(device)
optimizer = GraphMuon.from_model(
    model, lr=0.02, adamw_lr=0.001, tau=0.5, weight_decay=0.0,
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

for epoch in range(200):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    prediction = model(data.x, data.edge_index)
    loss = criterion(prediction[train_mask], target[train_mask])
    loss.backward()
    optimizer.step()
    scheduler.step()

    model.eval()
    with torch.no_grad():
        validation_prediction = model(data.x, data.edge_index)

optimizer.remove_hooks()
```

`from_model()` creates separate matrix and auxiliary AdamW parameter groups, so a scheduler updates both rates. Observers collect only while their modules are in training mode and autograd is enabled. Validation forwards do not replace training statistics.

| PyG layer | Observed input | Scope |
|---|---|---|
| `GCNConv` | Normalized propagated features `S @ X` | Homogeneous COO indices; honors normalization, edge weights, self-loops, and flow |
| `SAGEConv` | Actual input to `lin_l`, after aggregation | Root and projection matrices remain Muon |
| `GINConv` / `GINEConv` | Input to the first `nn.Linear` in the local MLP | Includes aggregation, root term, and GINE edge processing |
| `GPSConv` | Supported local modules, discovered recursively | Unobserved attention and feed-forward matrices remain Muon |
| `GATConv` | Raw features propagated by frozen, mean-head attention | Homogeneous COO, shared projection, source-to-target flow; approximation below |

The GAT adapter averages attention across heads and freezes it when estimating feature geometry. It includes realized attention dropout, but omits the attention Jacobian and separate head-specific metrics. This is an approximation to the full attention operator. Bipartite or reversed-flow GAT requires manual observation; the constructor warns and leaves those matrices on Muon.

GCN and GAT auto-observers require dense COO `edge_index` of shape `[2, E]`. For sparse adjacency objects, unusual propagation, or other architectures, supply effective inputs explicitly. GCN's adapter performs an additional propagation to obtain `S @ X`; include that cost in timings.

```bash
python examples/pyg_node_classification.py --dataset Cora --epochs 200
```

## Custom layers: observe a linear module

A linear module that consumes already propagated features needs one hook:

```python
class GraphLayer(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.projection = torch.nn.Linear(in_features, out_features)

    def forward(self, x, normalized_adjacency):
        propagated = torch.sparse.mm(normalized_adjacency, x)
        return self.projection(propagated)

layer = GraphLayer(64, 32).to(device)
optimizer = GraphMuon(layer.parameters(), lr=0.02)
optimizer.watch(layer.projection)
```

Use the normal training loop. The hook records the input second moment, detaches it from autograd, and retains no feature matrix. A projection executed **before** aggregation needs an explicit propagated-input observation instead.

The [plain PyTorch example](examples/torch_gcn.py) requires no PyG:

```bash
python examples/torch_gcn.py --device cpu
python examples/torch_gcn.py --device cuda
```

## Custom layers: explicit observations

For a PyTorch weight with layout `[out_features, in_features]`:

```python
optimizer = GraphMuon(model.parameters(), lr=0.02)
optimizer.register_graph(model.weight)  # Optional declaration, recommended

optimizer.zero_grad()
propagated = propagation(x)
prediction = torch.nn.functional.linear(propagated, model.weight, model.bias)
optimizer.observe(model.weight, propagated)
loss = criterion(prediction, target)
loss.backward()
optimizer.step()
```

For a mathematical `[in_features, out_features]` weight used as `X_g @ W`:

```python
optimizer.register_graph(W, input_axis=0)
optimizer.zero_grad()
X_g = torch.sparse.mm(S, H)
prediction = X_g @ W
optimizer.observe(W, X_g, input_axis=0)
criterion(prediction, target).backward()
optimizer.step()
```

`observe()` accepts `[..., in_features]` and flattens leading dimensions into rows. It checks feature dimension and device. Supply the effective inputs of this particular matrix after the relevant propagation and nonlinear processing. Explicit calls are not gated by `model.eval()`; call them only for training forwards.

For shared modules and gradient accumulation, observations before the same optimizer step combine by their original row counts. They estimate the second moment of all observed rows, without extra weighting by loss weights or graph size. Remove padding and unwanted rows yourself when using explicit observations.

## Parameter groups and AdamW fallback

Group modes are `"auto"`, `"muon"`, and `"adamw"`. `auto` uses GraphMuon for observed 2D parameters, Muon for other 2D parameters, and AdamW for other shapes. A `muon` group requires 2D parameters and still becomes graph-aware when observed. An `adamw` group always uses AdamW and cannot be registered as graph-aware.

For separate rates and decay settings:

```python
matrices = [p for p in model.parameters() if p.requires_grad and p.ndim == 2]
auxiliary = [p for p in model.parameters() if p.requires_grad and p.ndim != 2]
optimizer = GraphMuon([
    {"params": matrices, "lr": 0.02, "algorithm": "muon", "weight_decay": 0.01},
    {"params": auxiliary, "lr": 0.001, "algorithm": "adamw", "weight_decay": 0.0},
])
optimizer.watch(model.graph_projection)
```

Place embeddings or selected output heads in the AdamW group regardless of rank. A parameter must belong to only one group. `from_model()` excludes frozen parameters and sends embeddings to AdamW automatically.

In a direct mixed-rank `auto` group, `adamw_lr` controls non-matrix parameters independently of `lr`. An ordinary scheduler changes the group's `lr`, not its `adamw_lr`. Use explicit AdamW groups or `from_model()` when scheduling both rates.

## Options

All options can be overridden per group.

| Option | Default | Meaning |
|---|---:|---|
| `lr` | `0.02` | Base matrix rate; explicit AdamW groups use it directly |
| `adamw_lr` | `0.001` | Auxiliary AdamW rate |
| `momentum` | `0.95` | Matrix momentum coefficient |
| `nesterov` | `True` | Use Nesterov matrix momentum |
| `weight_decay` | `0.0` | Decoupled weight decay |
| `tau` | `0.5` | Graph metric strength in `[0, 1]` |
| `ns_steps` | `5` | Quintic Newton–Schulz iterations |
| `graph_scale` | `True` | Normalize direction by raw feature energy |
| `covariance_eps` | `1e-4` | Normalized metric regularizer |
| `scale_eps` | `1e-8` | Positive constant inside the graph-scale square root |
| `max_full_dim` | `256` | Diagonal approximation above this input width; `0` always uses diagonal |
| `sample_size` | `0` | Rows sampled with replacement per observation; `0` uses all rows |
| `update_freq` | `1` | Covariance refresh interval, counted per active matrix update |
| `ns_dtype` | `torch.bfloat16` | NS arithmetic; also accepts `torch.float32` or `torch.float64` |
| `betas` | `(0.9, 0.999)` | AdamW moment coefficients |
| `eps` | `1e-8` | AdamW denominator epsilon |
| `sync_stats` | `False` | Synchronize graph statistics over an initialized distributed group |

Tune matrix and auxiliary learning rates separately. Larger `tau` is not guaranteed to improve a task. For lower overhead:

```python
optimizer = GraphMuon.from_model(
    model, lr=0.02, max_full_dim=0, sample_size=2048, update_freq=10,
)
```

This uses diagonal, sampled, periodically refreshed statistics. Cached raw covariance is reused between refreshes; direction and scale still change with momentum. Full metrics cache eigendecompositions until the next refresh. Sampling consumes PyTorch's RNG, so seed it. These approximations can change convergence; benchmark the complete training loop.

## Accumulation, clipping, and AMP

Call `zero_grad()` once per accumulated optimizer step:

```python
optimizer.zero_grad()
for microbatch in microbatches:
    loss = loss_for(microbatch) / len(microbatches)
    loss.backward()  # Automatic observers collect each microbatch
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

With modern PyTorch's AMP interface:

```python
scaler = torch.amp.GradScaler("cuda")
optimizer.zero_grad()
with torch.autocast("cuda", dtype=torch.float16):
    loss = loss_for(batch)
scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
scaler.step(optimizer)
scaler.update()
```

Statistics and optimizer accumulators use FP32 for FP16/BF16/FP32 parameters and FP64 for FP64 parameters. NS uses `ns_dtype`. Autocast is disabled for optimizer math. Standard AMP with FP32 model weights is recommended; directly storing parameters in low precision also rounds each update. After a skipped AMP step, start the next attempt with `zero_grad()` to discard pending observations.

## Checkpoints and closures

Save immediately after an optimizer step:

```python
torch.save({
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict(),
    "epoch": epoch,
}, "checkpoint.pt")
```

Recreate the same model, groups, and observers. Initialize the scheduler before loading optimizer state, following the [PyTorch optimizer API](https://docs.pytorch.org/docs/stable/optim.html):

```python
checkpoint = torch.load("checkpoint.pt", map_location=device, weights_only=True)
model.load_state_dict(checkpoint["model"])
optimizer = GraphMuon.from_model(model, lr=0.02)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
scheduler.load_state_dict(checkpoint["scheduler"])
optimizer.load_state_dict(checkpoint["optimizer"])
```

Optimizer state includes momentum, AdamW moments, covariance, eigendecompositions, refresh counters, and layout declarations. Hook handles and pending observations are excluded. A checkpoint during accumulation is therefore incomplete. Also save RNG and AMP scaler state for exact training continuation.

Closures use the ordinary PyTorch pattern:

```python
def closure():
    optimizer.zero_grad()
    loss = loss_for(batch)
    loss.backward()
    return loss

loss = optimizer.step(closure)
```

## Distributed training and compilation

For replicated DistributedDataParallel, initialize the process group and device, then construct observers on the underlying model:

```python
ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
optimizer = GraphMuon.from_model(ddp_model.module, lr=0.02, sync_stats=True)
```

DDP reduces gradients. `sync_stats=True` additionally sums second-moment totals and row counts across ranks at each refresh. Ranks must register and update the same graph matrices in the same order and refresh on the same steps. Unequal local row counts are supported. This is not FSDP, tensor-parallel, or sharded-optimizer support. Unsynchronized local graph statistics can otherwise produce different updates across ranks.

This is an eager optimizer. Hooks, Python counters, and collectives are not designed for `torch.compile(fullgraph=True)` or CUDA graph capture. Compiled models may introduce observer-related graph breaks; verify that path for your model. Reentrant activation checkpointing can suppress or repeat hooks; use explicit observations outside checkpointed regions when appropriate.

## Algorithm and conventions

For a mathematical `[in, out]` matrix, the package forms an uncentered second moment `C = X_g.T @ X_g / n`, trace-normalizes and regularizes the directional metric, computes a metric-transformed Muon direction, and normalizes its scale using raw `C`. See equations (5)–(8) and Algorithm 1 in the [paper](https://arxiv.org/pdf/2611.08379).

The shape correction is `r = sqrt(max(1, in / out))` on the mathematical matrix, independent of storage layout. Matrix decay uses effective rate `lr * r`. This differs from applying native PyTorch Muon's shape correction directly to a stored `[out, in]` tensor; adjust per-matrix baseline rates when comparing. `tau=0` disables metric-aware orthogonalization; `graph_scale=False` separately disables graph normalization. Set both for a Muon control. Numerical results need not be bitwise identical to native `torch.optim.Muon`.

Full, fresh, unsampled statistics directly implement the equations. Diagonal, sampled, or stale statistics estimate a different metric from the current full batch; the exact-covariance representation bound should not be claimed unchanged for them. Zero feature energy uses an isotropic directional fallback while keeping zero raw energy. NS is a fixed polynomial approximation, not an exact SVD polar factor.

## Benchmarks: selected positive results

<!-- BENCHMARKS:START -->
These are **fresh measurements of this package**, not a reproduction of the paper tables.
**Selection:** only completed cells faster than both AdamW and Muon are shown. This is a selected positive subset, not an overall win-rate claim. All trials, including unsuccessful and non-positive results, remain in the private experiment archive.

Current audited coverage: **4 selected positive cells out of 104 fully scored cells**. Other scheduled cells are omitted until complete.

### Protocol

- Each optimizer has three learning rates and three seeds (0–2), with matched architecture, split, epoch budget, and batch size. AdamW rates: `{0.001, 0.003, 0.01}`; Muon and GraphMuon rates: `{0.003, 0.01, 0.03}`. Each family is selected by its lowest three-epoch-sustained mean validation loss, never by crossing time or test metrics.
- The fixed target is `1.02 × max_seed(best three-epoch-sustained AdamW validation loss)` for the selected AdamW configuration. A run reaches it after three consecutive validation evaluations at or below it. AdamW and GraphMuon must reach it in all three selected seeds. If Muon misses in a completed run, its measured final-budget time is a lower bound on that seed’s crossing time: `>` marks the resulting lower bounds on its mean time and speedup. “Muon reached” shows its successful seeds; a censored row is included only when the speedup lower bound also exceeds 1.
- Times are mean cumulative **training plus validation seconds** at the third qualifying epoch, measured with CUDA synchronization. Speedup is the ratio of those mean times: `baseline seconds / GraphMuon seconds`. Loading and one state-restored forward/backward warmup are excluded; the first optimizer update is timed. Values above `1×` favor GraphMuon.
- Node models: two graph layers, width 64, dropout 0.3, 200 epochs; GPS uses two local SAGE blocks with four-head Performer attention. Graph models: three layers, width 64, mean pooling, dropout 0.1, 40 epochs, batch size 512. GraphGPS uses four-head attention and local GINE blocks. These are fixed-budget transfer experiments, not claims of fully tuned final accuracy.
- GraphMuon: `tau=0.5`, five NS iterations, diagonal metric, 2,048 sampled rows with replacement, covariance refresh every ten active updates. Muon uses native `torch.optim.Muon` with per-matrix rates adjusted to the same mathematical shape correction; both use AdamW at 0.001 for auxiliary parameters. Weight decay is zero for all methods. GAT uses the documented frozen mean-head approximation.
- PyTorch 2.11.0+cu130, PyG 2.7.0, FP32 weights, BF16 NS, TF32 disabled, one RTX 6000 Ada GPU and four CPU threads per task. Timings from different optimizer runs can be affected by shared-node contention.
- Citation datasets use their standard public split and normalized features. Heterophilous datasets use published split 0 and standardized features; non-directed variants are symmetrized. OGB uses official train/validation splits; MAG uses its paper-node citation subgraph. Molecular test sets are not used for selection.
- Products and proteins use the same two-layer GCN equations with sparse CSR propagation and observers on the post-propagation linear modules to bound memory use. Protein node features are mean incident edge features; its target loss is multilabel BCE. Dataset preparation is excluded from timing.
- Three seeds are a small sample. Ratios near 1 and sub-second measurements should be treated as noisy observations, not established statistical gains. A dagger marks any row whose mean crossing time is below one second for any method. No aggregate across this selected subset is reported.

### GCN — node classification

0 positive rows / 21 fully scored dataset cells.

| Dataset | Task | Validation target | AdamW s | Muon s | GraphMuon s | vs AdamW | vs Muon | Muon reached | Seeds |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|

No completed positive comparison is currently listed for this architecture.

### GAT — node classification

1 positive rows / 21 fully scored dataset cells.

| Dataset | Task | Validation target | AdamW s | Muon s | GraphMuon s | vs AdamW | vs Muon | Muon reached | Seeds |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cora † | Citation node classification | 0.9855 | 0.156 | 0.307 | 0.107 | 1.46× | 2.87× | 3/3 | 3 |

### GraphSAGE — node classification

2 positive rows / 21 fully scored dataset cells.

| Dataset | Task | Validation target | AdamW s | Muon s | GraphMuon s | vs AdamW | vs Muon | Muon reached | Seeds |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| citeseer † | Citation node classification | 1.083 | 0.097 | 0.158 | 0.050 | 1.95× | 3.17× | 3/3 | 3 |
| pubmed † | Citation node classification | 0.5888 | 0.226 | >0.719 | 0.110 | 2.05× | >6.52× | 2/3 | 3 |

### GIN — node classification

1 positive rows / 21 fully scored dataset cells.

| Dataset | Task | Validation target | AdamW s | Muon s | GraphMuon s | vs AdamW | vs Muon | Muon reached | Seeds |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| citeseer † | Citation node classification | 1.688 | 0.165 | 0.144 | 0.091 | 1.81× | 1.58× | 3/3 | 3 |

### GPS with Performer attention — node classification

0 positive rows / 11 fully scored dataset cells.

| Dataset | Task | Validation target | AdamW s | Muon s | GraphMuon s | vs AdamW | vs Muon | Muon reached | Seeds |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|

No completed positive comparison is currently listed for this architecture.

### GCN — graph classification and regression

0 positive rows / 3 fully scored dataset cells.

| Dataset | Task | Validation target | AdamW s | Muon s | GraphMuon s | vs AdamW | vs Muon | Muon reached | Seeds |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|

No completed positive comparison is currently listed for this architecture.

### GINE — graph classification and regression

0 positive rows / 3 fully scored dataset cells.

| Dataset | Task | Validation target | AdamW s | Muon s | GraphMuon s | vs AdamW | vs Muon | Muon reached | Seeds |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|

No completed positive comparison is currently listed for this architecture.

### GraphGPS — graph classification and regression

0 positive rows / 3 fully scored dataset cells.

| Dataset | Task | Validation target | AdamW s | Muon s | GraphMuon s | vs AdamW | vs Muon | Muon reached | Seeds |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|

No completed positive comparison is currently listed for this architecture.
<!-- BENCHMARKS:END -->

## Repository contents

```text
graphmuon/
  optimizer.py                  # Optimizer and matrix update
  pyg.py                        # Optional PyG observers
examples/
  torch_gcn.py                  # Plain PyTorch usage demo
  pyg_node_classification.py    # PyG training example
tests/                          # Numerical and integration checks
```

Experiment runners, datasets, trajectories, tuning records, checkpoints, and machine logs are maintained outside this public repository. Only aggregate benchmark results and their protocol appear above. This repository is an optimizer library, rather than the paper's full experiment harness.

## Citation

```bibtex
@article{hadad2026moduleaware,
  title={Module-Aware Optimization for Graph Neural Networks},
  author={Hadad, Guy and Roitman, Haggai and Eliasof, Moshe and Shapira, Bracha},
  journal={arXiv preprint arXiv:2611.08379},
  year={2026}
}
```

The quintic iteration follows the [original Muon implementation](https://github.com/KellerJordan/Muon). Optional adapters use [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/). Released under the [MIT license](LICENSE).
