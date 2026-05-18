#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
resnet18_chakra.py
- 用随机数据跑 ResNet-18（训练若干步）
- 同时采集 PyTorch Execution Trace + Kineto trace
- 生成:
    trace_out/pytorch_et.json
    trace_out/kineto_trace.json

之后用:
    chakra_trace_link ...
    chakra_converter ...
把 trace 转成 Chakra ET
"""

import os
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision


def run_training(
    device: torch.device,
    steps: int,
    batch_size: int,
    num_classes: int,
    image_size: int,
    lr: float,
    prof_step=None,
):
    """一个最小训练循环：每个 step 用随机数据 forward/backward/step。"""

    # ResNet-18（不下载权重）
    model = torchvision.models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # 预先构造固定 shape 的随机输入（更贴近真实 workload 的稳定形状）
    x = torch.randn(batch_size, 3, image_size, image_size, device=device)
    y = torch.randint(0, num_classes, (batch_size,), device=device)

    # 可选：让 GPU “热起来”，减少首步开销对 trace 的影响
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.time()
    for it in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        # profiler 每个 iteration 末尾都要 step()
        if prof_step is not None:
            prof_step()

        if (it + 1) % max(1, steps // 5) == 0:
            print(f"[iter {it+1:4d}/{steps}] loss={loss.item():.4f}")

    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()
    print(f"[Done] steps={steps}, elapsed={t1 - t0:.3f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="trace_out", help="trace 输出目录")
    parser.add_argument("--steps", type=int, default=10, help="采集/运行多少个 step（建议先小再大）")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_classes", type=int, default=1000)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--cpu_only", action="store_true", help="强制只用 CPU（无 CUDA 也会自动退化）")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    pytorch_et_path = os.path.join(args.out_dir, "pytorch_et.json")
    kineto_path = os.path.join(args.out_dir, "kineto_trace.json")

    # 选择设备
    if (not args.cpu_only) and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[Device] {device}")

    # ---------------------------
    # 1) PyTorch Execution Trace
    # ---------------------------
    from torch.profiler import ExecutionTraceObserver

    et = ExecutionTraceObserver()
    et.register_callback(pytorch_et_path)
    et.start()

    # ---------------------------
    # 2) Kineto trace
    # ---------------------------
    def trace_handler(prof):
        prof.export_chrome_trace(kineto_path)

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    # schedule: wait=0 warmup=0 active=steps repeat=1 -> 全部 steps 都采
    with torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(wait=0, warmup=0, active=args.steps, repeat=1),
        record_shapes=True,
        with_stack=False,
        on_trace_ready=trace_handler,
    ) as prof:
        run_training(
            device=device,
            steps=args.steps,
            batch_size=args.batch_size,
            num_classes=args.num_classes,
            image_size=args.image_size,
            lr=args.lr,
            prof_step=prof.step,
        )

    et.stop()
    et.unregister_callback()

    print("\n[OK] Raw traces generated:")
    print(f"  - {pytorch_et_path}")
    print(f"  - {kineto_path}")

    print("\nNext (Chakra):")
    print("  1) Link host/device traces into one JSON:")
    print(f"     chakra_trace_link --pytorch-et-file {pytorch_et_path} --kineto-file {kineto_path} --output-file {os.path.join(args.out_dir, 'linked_host_device_trace.json')}")
    print("  2) Convert to Chakra ET:")
    print(f"     chakra_converter PyTorch --input {os.path.join(args.out_dir, 'linked_host_device_trace.json')} --output {os.path.join(args.out_dir, 'chakra_et')}")
    print("\nTip: 用 Chrome/Perfetto 打开 kineto_trace.json 可以先看时间线是否合理。")


if __name__ == "__main__":
    main()
