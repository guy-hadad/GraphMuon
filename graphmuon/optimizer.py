"""Small, eager PyTorch implementation of GraphMuon, equations (5)--(8)."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.optim import Optimizer


def _zeropower(matrix: Tensor, steps: int, dtype: torch.dtype) -> Tensor:
    x = matrix.to(dtype=dtype, copy=True)
    transpose = x.shape[0] > x.shape[1]
    if transpose:
        x = x.T
    x = x / x.norm().clamp_min(1e-7)
    for _ in range(steps):
        a = x @ x.T
        b = torch.addmm(a, a, a, beta=-4.7750, alpha=2.0315)
        x = torch.addmm(x, b, x, beta=3.4445)
    return (x.T if transpose else x).to(matrix.dtype)


class GraphMuon(Optimizer):
    """GraphMuon for observed matrices, Muon for other matrices, AdamW otherwise.

    Use ``GraphMuon.from_model(model)`` for supported PyG layers, or call
    ``observe(weight, propagated_inputs)`` before ``step()`` for custom layers.
    Parameters use torch's [out, in] layout unless input_axis=0 is specified.
    See the README for parameter groups, approximations and training examples.
    """

    def __init__(
        self,
        params,
        lr=0.02,
        *,
        momentum=0.95,
        nesterov=True,
        weight_decay=0.0,
        tau=0.5,
        ns_steps=5,
        graph_scale=True,
        covariance_eps=1e-4,
        scale_eps=1e-8,
        max_full_dim=256,
        sample_size=0,
        update_freq=1,
        adamw_lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        ns_dtype=torch.bfloat16,
        sync_stats=False,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
            tau=tau,
            ns_steps=ns_steps,
            graph_scale=graph_scale,
            covariance_eps=covariance_eps,
            scale_eps=scale_eps,
            max_full_dim=max_full_dim,
            sample_size=sample_size,
            update_freq=update_freq,
            adamw_lr=adamw_lr,
            betas=betas,
            eps=eps,
            ns_dtype=ns_dtype,
            sync_stats=sync_stats,
            algorithm="auto",
        )
        self._groups = {}
        self._pending = {}
        self._handles = []
        self.graph_parameters = []
        super().__init__(params, defaults)

    def add_param_group(self, param_group):
        options = {**self.defaults, **param_group}
        for key in ("lr", "adamw_lr", "weight_decay"):
            if not math.isfinite(options[key]) or options[key] < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
        for key in ("covariance_eps", "scale_eps", "eps"):
            if not math.isfinite(options[key]) or options[key] <= 0:
                raise ValueError(f"{key} must be finite and positive")
        if not 0 <= options["momentum"] < 1 or not 0 <= options["tau"] <= 1:
            raise ValueError("momentum must be in [0, 1); tau must be in [0, 1]")
        if len(options["betas"]) != 2 or not all(0 <= b < 1 for b in options["betas"]):
            raise ValueError("betas must contain two numbers in [0, 1)")
        for key, lower in (
            ("ns_steps", 1),
            ("update_freq", 1),
            ("max_full_dim", 0),
            ("sample_size", 0),
        ):
            if not isinstance(options[key], int) or options[key] < lower:
                raise ValueError(f"{key} must be an integer >= {lower}")
        if options["ns_dtype"] not in (torch.float32, torch.float64, torch.bfloat16):
            raise ValueError("ns_dtype must be float32, float64 or bfloat16")
        if options["algorithm"] not in ("auto", "muon", "adamw"):
            raise ValueError("algorithm must be auto, muon or adamw")
        super().add_param_group(param_group)
        for p in self.param_groups[-1]["params"]:
            if not p.is_floating_point():
                raise TypeError("GraphMuon requires real floating-point parameters")
            if options["algorithm"] == "muon" and p.ndim != 2:
                raise ValueError("Muon groups require 2D parameters")
            self._groups[p] = self.param_groups[-1]

    @classmethod
    def from_model(cls, model: nn.Module, **kwargs):
        """Attach PyG input observers; embeddings use AdamW automatically.

        GCN, SAGE, GIN, GINE, and local modules nested in GPS are supported.
        GAT uses a frozen, mean-head attention approximation (see README).
        Unsupported matrices receive ordinary Muon, with a warning when no
        supported graph layer is found. PyG is an optional dependency.
        """
        from .pyg import attach

        embeddings = {
            p for m in model.modules() if isinstance(m, nn.Embedding) for p in m.parameters()
        }
        auxiliary = {p for p in model.parameters() if p.ndim != 2 or p in embeddings}
        groups = [
            {"params": [p for p in model.parameters() if p.requires_grad and p not in auxiliary]}
        ]
        if auxiliary:
            groups.append(
                {
                    "params": [p for p in model.parameters() if p.requires_grad and p in auxiliary],
                    "algorithm": "adamw",
                    "lr": kwargs.get("adamw_lr", 1e-3),
                }
            )
        optimizer = cls(groups, **kwargs)
        attach(optimizer, model)
        return optimizer

    def register_graph(self, parameter: Tensor, *, input_axis=1):
        """Declare a matrix that must receive propagated-input observations."""
        if parameter not in self._groups or parameter.ndim != 2:
            raise ValueError("parameter must be a 2D tensor owned by this optimizer")
        if input_axis not in (0, 1):
            raise ValueError("input_axis must be 0 ([in,out]) or 1 ([out,in])")
        if self._groups[parameter]["algorithm"] == "adamw":
            raise ValueError("an AdamW parameter cannot be registered as graph-aware")
        state = self.state[parameter]
        if "input_axis" in state and state["input_axis"] != input_axis:
            raise ValueError("cannot change input_axis after registration")
        state["input_axis"] = input_axis
        if all(p is not parameter for p in self.graph_parameters):
            self.graph_parameters.append(parameter)

    def needs_observation(self, parameter: Tensor) -> bool:
        """Whether this step refreshes the covariance for this matrix."""
        state, group = self.state[parameter], self._groups[parameter]
        return "covariance" not in state or state.get("step", 0) % group["update_freq"] == 0

    @torch.no_grad()
    def observe(self, parameter: Tensor, inputs: Tensor, *, input_axis=1):
        """Accumulate X_g.T @ X_g / n without retaining activations or autograd.

        Repeated observations before a step combine by row count, supporting
        shared modules and gradient accumulation. Call only on training data.
        """
        self.register_graph(parameter, input_axis=input_axis)
        if inputs.ndim < 2 or inputs.shape[-1] != parameter.shape[input_axis]:
            raise ValueError("inputs must have shape [..., parameter input dimension]")
        if inputs.device != parameter.device:
            raise ValueError("inputs and parameter must be on the same device")
        if not self.needs_observation(parameter):
            return
        group = self._groups[parameter]
        dtype = torch.float64 if parameter.dtype == torch.float64 else torch.float32
        # Disable autocast so covariance accumulation stays in the statistics dtype.
        with torch.autocast(device_type=parameter.device.type, enabled=False):
            x = inputs.detach().reshape(-1, inputs.shape[-1])
            original_n = x.shape[0]
            if original_n == 0:
                raise ValueError("cannot estimate covariance from zero rows")
            if 0 < group["sample_size"] < original_n:
                # Bounded sampling cost; samples are drawn with replacement.
                idx = torch.randint(original_n, (group["sample_size"],), device=x.device)
                x = x[idx]
            x = x.to(dtype)
            moment = (
                x.T @ x if x.shape[1] <= group["max_full_dim"] else x.square().sum(0)
            ) / x.shape[0]
            weighted = moment * original_n
            if parameter in self._pending:
                previous, count = self._pending[parameter]
                self._pending[parameter] = (previous + weighted, count + original_n)
            else:
                self._pending[parameter] = (weighted, original_n)

    def watch(self, module: nn.Module, parameter=None, *, input_axis=1):
        """Observe the first input of a module that consumes propagated features."""
        parameter = module.weight if parameter is None else parameter
        self.register_graph(parameter, input_axis=input_axis)

        def capture(layer, args, kwargs):
            if layer.training and torch.is_grad_enabled():
                x = args[0] if args else kwargs.get("input", kwargs.get("x"))
                self.observe(parameter, x, input_axis=input_axis)

        handle = module.register_forward_pre_hook(capture, with_kwargs=True)
        self._handles.append(handle)
        return handle

    def remove_hooks(self):
        """Detach automatically installed observers (e.g. after training)."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def zero_grad(self, set_to_none=True):
        self._pending.clear()
        super().zero_grad(set_to_none=set_to_none)

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self._groups = {p: g for g in self.param_groups for p in g["params"]}
        # Optimizer.load_state_dict casts floating state to the parameter dtype.
        # Restore statistics directly from the checkpoint to preserve FP32 state
        # for FP16/BF16 parameters, including covariance precision.
        for saved, current in zip(state_dict["param_groups"], self.param_groups):
            for index, parameter in zip(saved["params"], current["params"]):
                dtype = torch.float64 if parameter.dtype == torch.float64 else torch.float32
                for key in (
                    "covariance",
                    "momentum_buffer",
                    "exp_avg",
                    "exp_avg_sq",
                    "eigenvalues",
                    "eigenvectors",
                ):
                    value = state_dict["state"].get(index, {}).get(key)
                    if value is not None:
                        self.state[parameter][key] = value.to(
                            device=parameter.device, dtype=dtype, copy=True
                        )
        self.graph_parameters = [p for p in self._groups if "input_axis" in self.state[p]]
        self._pending.clear()

    @staticmethod
    def _direction(momentum, covariance, group, cache=None):
        if covariance is None or group["tau"] == 0:
            return _zeropower(momentum, group["ns_steps"], group["ns_dtype"])
        diagonal = covariance.ndim == 1
        energy = covariance.mean() if diagonal else covariance.diag().mean()
        normalized = covariance / energy.clamp_min(torch.finfo(covariance.dtype).tiny)
        if diagonal:
            normalized = torch.where(energy > 0, normalized, torch.ones_like(normalized))
            power = (normalized + group["covariance_eps"]).pow(group["tau"] / 2)
            mapped = momentum * power[:, None]
            return _zeropower(mapped, group["ns_steps"], group["ns_dtype"]) / power[:, None]
        if cache is not None and "eigenvalues" in cache:
            values, vectors = cache["eigenvalues"], cache["eigenvectors"]
        else:
            identity = torch.eye(
                covariance.shape[0], dtype=covariance.dtype, device=covariance.device
            )
            normalized = torch.where(energy > 0, (normalized + normalized.T) * 0.5, identity)
            normalized.diagonal().add_(group["covariance_eps"])
            values, vectors = torch.linalg.eigh(normalized)
            if cache is not None:
                cache["eigenvalues"], cache["eigenvectors"] = values, vectors
        power = values.clamp_min(group["covariance_eps"]).pow(group["tau"] / 2)
        mapped = vectors @ (power[:, None] * (vectors.T @ momentum))
        orthogonal = _zeropower(mapped, group["ns_steps"], group["ns_dtype"])
        return vectors @ ((vectors.T @ orthogonal) / power[:, None])

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        # Check every active parameter before changing any weight.
        for group in self.param_groups:
            if group["sync_stats"] and not torch.distributed.is_initialized():
                raise RuntimeError(
                    "sync_stats=True requires an initialized torch.distributed process group"
                )
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse or p.grad.is_complex():
                    raise RuntimeError("GraphMuon requires dense, real gradients")
                state = self.state[p]
                if "input_axis" in state and self.needs_observation(p) and p not in self._pending:
                    raise RuntimeError("Missing propagated inputs: call observe() before step()")
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                with torch.autocast(device_type=p.device.type, enabled=False):
                    self._update(p, group)
        self._pending.clear()
        return loss

    def _update(self, p, group):
        state = self.state[p]
        state["step"] = state.get("step", 0) + 1
        dtype = torch.float64 if p.dtype == torch.float64 else torch.float32
        gradient = p.grad.to(dtype)
        use_adamw = group["algorithm"] == "adamw" or p.ndim != 2
        if use_adamw:
            beta1, beta2 = group["betas"]
            if "exp_avg" not in state:
                state["exp_avg"] = torch.zeros_like(p, dtype=dtype)
                state["exp_avg_sq"] = torch.zeros_like(p, dtype=dtype)
            first = state["exp_avg"].to(dtype)
            second = state["exp_avg_sq"].to(dtype)
            first.lerp_(gradient, 1 - beta1)
            second.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
            state["exp_avg"], state["exp_avg_sq"] = first, second
            direction = first / (1 - beta1 ** state["step"])
            denominator = (second / (1 - beta2 ** state["step"])).sqrt().add_(group["eps"])
            lr = group["lr"] if group["algorithm"] == "adamw" else group["adamw_lr"]
            p.mul_(1 - lr * group["weight_decay"])
            p.add_((direction / denominator).to(p.dtype), alpha=-lr)
            return
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros_like(p, dtype=dtype)
        buffer = state["momentum_buffer"].to(dtype)
        buffer.mul_(group["momentum"]).add_(gradient)
        state["momentum_buffer"] = buffer
        momentum = gradient.add(buffer, alpha=group["momentum"]) if group["nesterov"] else buffer
        axis = state.get("input_axis", 1)
        matrix = momentum.T if axis == 1 else momentum
        if p in self._pending:
            total, count = self._pending[p]
            if group["sync_stats"]:
                count = total.new_tensor(count)
                torch.distributed.all_reduce(total)
                torch.distributed.all_reduce(count)
            state["covariance"] = total / count
            state.pop("eigenvalues", None)
            state.pop("eigenvectors", None)
        covariance = state.get("covariance")
        if covariance is not None:
            covariance = covariance.to(dtype)
        direction = self._direction(matrix, covariance, group, state)
        if covariance is not None and group["graph_scale"]:
            product = (
                covariance[:, None] * direction if covariance.ndim == 1 else covariance @ direction
            )
            squared_norm = direction.square().sum().clamp_min(torch.finfo(dtype).tiny)
            scale = (
                (direction * product).sum().clamp_min(0) / squared_norm + group["scale_eps"]
            ).sqrt()
            direction = direction / scale
        # r is defined on the mathematical [in, out] matrix, independent of storage layout.
        lr = group["lr"] * math.sqrt(max(1, matrix.shape[0] / matrix.shape[1]))
        p.mul_(1 - lr * group["weight_decay"])
        p.add_((direction.T if axis == 1 else direction).to(p.dtype), alpha=-lr)
