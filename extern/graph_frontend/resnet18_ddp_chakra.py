#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DDP 多进程训练 ResNet18（用固定shape随机数据，免下载数据集）
同时采集：
  - PyTorch Execution Trace (host)  -> pytorch_et_rank{r}.json
  - Kineto trace (device timeline) -> kineto_rank{r}.json

后续用 chakra_trace_link 合并成 ET+，再用 chakra_converter 转成 Chakra ET（.et）
"""

import os
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def ddp_init():
    """初始化 DDP，并返回 (rank, world_size, local_rank, device)."""
    if not dist.is_available():
        raise RuntimeError("torch.distributed 不可用")

    # torchrun 会自动注入这些环境变量
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    # GPU 优先
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"

    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    return rank, world_size, local_rank, device


def ddp_cleanup():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


@torch.no_grad()
def _warmup_cuda(device):
    """小热身，减少首步 kernel 初始化/缓存对 trace 的干扰。"""
    if device.type == "cuda":
        x = torch.randn(8, 3, 224, 224, device=device)
        m = torchvision.models.resnet18(weights=None).to(device).eval()
        _ = m(x)
        torch.cuda.synchronize()


def train_steps(model, device, steps, batch_size, image_size, num_classes, lr, prof_step=None):
    """最小训练循环：固定 shape 张量重复用（trace 更干净），每步调用 prof_step()。"""
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # 固定 shape 输入（避免每步随机数/数据加载噪声影响 trace）
    x = torch.randn(batch_size, 3, image_size, image_size, device=device)
    y = torch.randint(0, num_classes, (batch_size,), device=device)

    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.time()
    for it in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        if prof_step is not None:
            prof_step()

        if (it + 1) % max(1, steps // 5) == 0:
            print(f"[iter {it+1:4d}/{steps}] loss={loss.item():.4f}", flush=True)

    if device.type == "cuda":
        torch.cuda.synchronize()
    print(f"[Done] steps={steps}, elapsed={time.time()-t0:.3f}s", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="ddp_trace_out")
    parser.add_argument("--steps", type=int, default=20, help="总训练步数（>= wait+warmup+active）")
    parser.add_argument("--trace_active_steps", type=int, default=10, help="真正记录多少步（active）")
    parser.add_argument("--trace_wait", type=int, default=1, help="等待多少步后开始 warmup/active")
    parser.add_argument("--trace_warmup", type=int, default=1, help="warmup 步数")
    parser.add_argument("--batch_size", type=int, default=8, help="每个 rank 的 batch_size")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--num_classes", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.01)
    args = parser.parse_args()

    rank, world_size, local_rank, device = ddp_init()

    # 每个 rank 单独目录/文件名，防止互相覆盖
    os.makedirs(args.out_dir, exist_ok=True)
    host_trace = os.path.join(args.out_dir, f"pytorch_et_rank{rank}.json")
    kineto_trace = os.path.join(args.out_dir, f"kineto_rank{rank}.json")

    if rank == 0:
        print(f"[DDP] world_size={world_size}", flush=True)
    print(f"[Rank {rank}] device={device}", flush=True)

    # 构建 ResNet18 + DDP
    model = torchvision.models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, args.num_classes)
    model.to(device)
    if device.type == "cuda":
        model = DDP(model, device_ids=[local_rank])
    else:
        model = DDP(model)

    # 可选热身（尤其是 CUDA）
    _warmup_cuda(device)

    # 1) PyTorch Execution Trace Observer
    from torch.profiler import ExecutionTraceObserver
    et = ExecutionTraceObserver()
    et.register_callback(host_trace)
    et.start()

    # 2) Kineto profiler（每 rank 导出自己的 chrome trace）
    def trace_handler(prof):
        prof.export_chrome_trace(kineto_trace)

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    # schedule：wait -> warmup -> active
    # 注意：总 steps 必须 >= wait+warmup+active
    total_needed = args.trace_wait + args.trace_warmup + args.trace_active_steps
    if args.steps < total_needed:
        if rank == 0:
            print(f"[WARN] steps={args.steps} < wait+warmup+active={total_needed}，将 active 可能采不满。", flush=True)

    try:
        with torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(
                wait=args.trace_wait, warmup=args.trace_warmup,
                active=args.trace_active_steps, repeat=1
            ),
            record_shapes=True,
            with_stack=False,
            on_trace_ready=trace_handler,
        ) as prof:
            train_steps(
                model=model,
                device=device,
                steps=args.steps,
                batch_size=args.batch_size,
                image_size=args.image_size,
                num_classes=args.num_classes,
                lr=args.lr,
                prof_step=prof.step,
            )
    finally:
        # 确保即使异常也会正确关闭 observer（避免 json 写不全）
        et.stop()
        et.unregister_callback()

    dist.barrier()
    if rank == 0:
        print("\n[OK] Raw per-rank traces written into:", os.path.abspath(args.out_dir), flush=True)
        print("      Next: run chakra_trace_link + chakra_converter per rank.", flush=True)

    ddp_cleanup()


if __name__ == "__main__":
    main()
