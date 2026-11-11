from pathlib import Path
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from graphmuon import GraphMuon


def _distributed_worker(rank, directory):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{directory}/rendezvous",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=20),
    )
    try:
        parameter = nn.Parameter(torch.zeros(2, 3))
        optimizer = GraphMuon([parameter], sync_stats=True)
        # Distinct feature distributions and unequal local row counts.
        inputs = torch.ones(rank + 1, 3) * (rank + 1)
        optimizer.observe(parameter, inputs)
        parameter.grad = torch.ones_like(parameter)  # Represents an already reduced gradient.
        optimizer.step()
        torch.save(
            {"weight": parameter.detach(), "covariance": optimizer.state[parameter]["covariance"]},
            Path(directory) / f"rank{rank}.pt",
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(), reason="Gloo unavailable"
)
def test_distributed_row_weighted_covariance(tmp_path):
    mp.spawn(_distributed_worker, args=(str(tmp_path),), nprocs=2, join=True)
    states = [torch.load(tmp_path / f"rank{rank}.pt", weights_only=True) for rank in range(2)]
    for state in states:
        # (1 * 1**2 + 2 * 2**2) / (1 + 2) == 3
        torch.testing.assert_close(state["covariance"], torch.full((3, 3), 3.0))
    torch.testing.assert_close(states[0]["weight"], states[1]["weight"], rtol=0, atol=0)
