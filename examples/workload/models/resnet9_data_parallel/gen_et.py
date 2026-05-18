## ******************************************************************************
## This source code is licensed under the MIT license found in the
## LICENSE file in the root directory of this source tree.
##
## Copyright (c) 2024 Georgia Institute of Technology
## ******************************************************************************

"""
一个“真实网络负载”的最小示例：分布式 ResNet9（数据并行，手动 All-Reduce 梯度），
并导出 ASTRA-sim 可直接读取的 Chakra 二进制 ET（每个 rank 一份 .et）。

特点：
- 真实运行 PyTorch 前向/反向/优化器 step（CPU 或 GPU）
- 通信用 torch.distributed.all_reduce（默认 gloo；如需 GPU 可用 nccl）
- 生成的 ET 只保留粗粒度节点：compute -> allreduce -> compute（每迭代）

运行（推荐从仓库根目录）：
  torchrun --standalone --nproc_per_node=4 \
    examples/workload/models/resnet9_data_parallel/gen_et.py \
    --out-dir examples/workload/models/resnet9_data_parallel/4npus_bs4_1iter \
    --prefix resnet9_dp --iters 1 --warmup-iters 1 --batch-size 4 --backend gloo

生成后，workload prefix 为：
  <out-dir>/<prefix>
例如：
  examples/workload/models/resnet9_data_parallel/4npus_bs4_1iter/resnet9_dp
对应文件：
  resnet9_dp.0.et / resnet9_dp.1.et / ...
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

# Workaround: current generated _pb2.py under extern/graph_frontend is not
# compatible with protobuf>=4 unless using pure-python implementation.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import torch
import torch.distributed as dist
import torch.nn as nn

# Ensure repo root is on sys.path so "extern.*" imports work even without PYTHONPATH.
REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from extern.graph_frontend.chakra.schema.protobuf.et_def_pb2 import (  # noqa: E402
    ALL_REDUCE,
    COMM_COLL_NODE,
    COMP_NODE,
    GlobalMetadata,
)
from extern.graph_frontend.chakra.schema.protobuf.et_def_pb2 import (  # noqa: E402
    AttributeProto as ChakraAttr,
)
from extern.graph_frontend.chakra.schema.protobuf.et_def_pb2 import Node as ChakraNode  # noqa: E402
from extern.graph_frontend.chakra.src.third_party.utils.protolib import (  # noqa: E402
    encodeMessage as encode_message,
)


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, pool: bool = False) -> None:
        super().__init__()
        layers: List[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResNet9(nn.Module):
    def __init__(self, in_ch: int = 3, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = ConvBlock(in_ch, 64, pool=False)
        self.conv2 = ConvBlock(64, 128, pool=True)
        self.res1 = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))
        self.conv3 = ConvBlock(128, 256, pool=True)
        self.conv4 = ConvBlock(256, 512, pool=True)
        self.res2 = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))
        self.classifier = nn.Sequential(
            nn.MaxPool2d(4),  # 4x4 -> 1x1 for CIFAR-like 32x32
            nn.Flatten(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + self.res1(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = out + self.res2(out)
        return self.classifier(out)


@dataclass
class _EtBuilder:
    out_path: str
    _next_id: int = 0
    _prev_id: Optional[int] = None
    _t_us: int = 0
    _nodes: List[ChakraNode] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self._nodes = []

    def _new_node(self, name: str, node_type: int, duration_us: int) -> ChakraNode:
        node = ChakraNode()
        node.id = self._next_id
        node.name = name
        node.type = node_type
        node.start_time_micros = max(0, int(self._t_us))
        node.duration_micros = max(0, int(duration_us))
        if self._prev_id is not None:
            node.ctrl_deps.append(int(self._prev_id))
        self._t_us += int(duration_us)
        self._prev_id = self._next_id
        self._next_id += 1
        return node

    def add_compute(self, name: str, duration_us: int, is_cpu_op: bool) -> None:
        node = self._new_node(name=name, node_type=COMP_NODE, duration_us=duration_us)
        node.attr.append(ChakraAttr(name="is_cpu_op", bool_val=bool(is_cpu_op)))
        self._nodes.append(node)

    def add_allreduce(self, name: str, duration_us: int, comm_size_bytes: int) -> None:
        node = self._new_node(
            name=name, node_type=COMM_COLL_NODE, duration_us=duration_us
        )
        # Comm nodes must NOT be on CPU in ASTRA-sim workload frontend.
        node.attr.append(ChakraAttr(name="is_cpu_op", bool_val=False))
        node.attr.append(ChakraAttr(name="comm_type", int64_val=int(ALL_REDUCE)))
        node.attr.append(ChakraAttr(name="comm_size", int64_val=int(comm_size_bytes)))
        self._nodes.append(node)

    def write(self) -> None:
        os.makedirs(os.path.dirname(self.out_path), exist_ok=True)
        with open(self.out_path, "wb") as f:
            encode_message(f, GlobalMetadata(version="0.0.4"))
            for node in self._nodes:
                encode_message(f, node)


def _get_rank_info() -> Tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return rank, local_rank, world_size


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _flatten_grads(params: List[torch.nn.Parameter]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    grads: List[torch.Tensor] = []
    for p in params:
        if p.grad is None:
            continue
        grads.append(p.grad)
    if not grads:
        raise RuntimeError("No gradients found to all-reduce")
    flat = torch.cat([g.contiguous().view(-1) for g in grads], dim=0)
    return flat, grads


def _assign_flat_to_grads(flat: torch.Tensor, grads: List[torch.Tensor]) -> None:
    # Use PyTorch internal helpers to avoid extra shape bookkeeping.
    unflat = torch._utils._unflatten_dense_tensors(flat, grads)
    for g, u in zip(grads, unflat):
        g.copy_(u)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--prefix", type=str, default="resnet9_dp")
    parser.add_argument("--backend", type=str, default="gloo", choices=["gloo", "nccl"])
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--compute-is-cpu-op",
        action="store_true",
        help="在导出的 ET 中将 COMP_NODE 标记为 CPU op（默认标记为 GPU op，以便在 ASTRA-sim 中计入 gpu_ops）",
    )
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.01)
    args = parser.parse_args()

    rank, local_rank, world_size = _get_rank_info()

    if world_size <= 1:
        raise RuntimeError("请使用 torchrun 启动多进程（WORLD_SIZE > 1）")

    dist.init_process_group(backend=args.backend, init_method="env://")

    # Device/backend compatibility:
    # - gloo: CPU only
    # - nccl: CUDA only
    if args.backend == "gloo":
        if args.device == "cuda":
            raise RuntimeError("gloo backend 不支持 CUDA tensor；请使用 --device cpu 或改用 --backend nccl")
        device = torch.device("cpu")
    else:  # nccl
        if args.device == "cpu":
            raise RuntimeError("nccl backend 只能用于 CUDA；请使用 --device cuda 或 --device auto")
        if not torch.cuda.is_available():
            raise RuntimeError("选择了 nccl backend 但当前没有可用的 CUDA 设备")
        device = torch.device("cuda", local_rank)

    # Avoid CPU over-subscription in multi-process runs.
    if device.type == "cpu":
        torch.set_num_threads(1)

    torch.manual_seed(1234 + rank)

    model = ResNet9(in_ch=3, num_classes=args.num_classes).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    # Sync initial parameters from rank 0 so all ranks start identical.
    for p in model.parameters():
        dist.broadcast(p.data, src=0)

    out_prefix = os.path.join(args.out_dir, args.prefix)
    out_path = f"{out_prefix}.{rank}.et"
    et = _EtBuilder(out_path=out_path)
    compute_is_cpu_op = bool(args.compute_is_cpu_op)

    def run_one_iter(record: bool, iter_idx: int) -> None:
        # Synthetic data (CIFAR-like)
        x = torch.randn(args.batch_size, 3, args.image_size, args.image_size, device=device)
        y = torch.randint(0, args.num_classes, (args.batch_size,), device=device)

        # compute: fwd + bwd
        _sync(device)
        t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        _sync(device)
        t1 = time.perf_counter()
        compute_us = int((t1 - t0) * 1e6)

        if record:
            et.add_compute(
                name=f"iter{iter_idx}_fwd_bwd",
                duration_us=compute_us,
                is_cpu_op=compute_is_cpu_op,
            )

        # comm: all-reduce flattened grads
        flat, grads = _flatten_grads(list(model.parameters()))
        comm_bytes = int(flat.numel() * flat.element_size())

        _sync(device)
        c0 = time.perf_counter()
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        _sync(device)
        c1 = time.perf_counter()
        comm_us = int((c1 - c0) * 1e6)

        # average
        flat.div_(float(world_size))
        _assign_flat_to_grads(flat, grads)

        if record:
            et.add_allreduce(
                name=f"iter{iter_idx}_allreduce_grads",
                duration_us=comm_us,
                comm_size_bytes=comm_bytes,
            )

        # compute: optimizer step (and grad unpack cost is included above)
        _sync(device)
        s0 = time.perf_counter()
        optimizer.step()
        _sync(device)
        s1 = time.perf_counter()
        step_us = int((s1 - s0) * 1e6)

        if record:
            et.add_compute(
                name=f"iter{iter_idx}_optimizer_step",
                duration_us=step_us,
                is_cpu_op=compute_is_cpu_op,
            )

    # warmup
    for i in range(max(0, int(args.warmup_iters))):
        run_one_iter(record=False, iter_idx=i)
        dist.barrier()

    # record
    for i in range(max(0, int(args.iters))):
        run_one_iter(record=True, iter_idx=i)
        dist.barrier()

    et.write()

    if rank == 0:
        print("Chakra ET prefix:", out_prefix)
        print("Example files:", f"{out_prefix}.0.et ... {out_prefix}.{world_size-1}.et")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

