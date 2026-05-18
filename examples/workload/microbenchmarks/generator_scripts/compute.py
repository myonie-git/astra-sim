## ******************************************************************************
## This source code is licensed under the MIT license found in the
## LICENSE file in the root directory of this source tree.
##
## Copyright (c) 2024 Georgia Institute of Technology
## ******************************************************************************

import argparse
import os
from extern.graph_frontend.chakra.schema.protobuf.et_def_pb2 import (
    GlobalMetadata,
    COMP_NODE,
)
from extern.graph_frontend.chakra.schema.protobuf.et_def_pb2 import (
    AttributeProto as ChakraAttr,
)
from extern.graph_frontend.chakra.schema.protobuf.et_def_pb2 import Node as ChakraNode
from extern.graph_frontend.chakra.src.third_party.utils.protolib import (
    encodeMessage as encode_message,
)


def generate_compute(
    npus_count: int,
    tensor_size_mb: int,
    num_ops: int,
    path: str = "./",
) -> None:
    """
    生成仅包含计算节点的 ET（用于触发 issue_comp/roofline 路径）。

    Args:
        npus_count (int): NPU 数量。
        tensor_size_mb (int): Tensor 尺寸（MB，影响 memory bw）。
        num_ops (int): 计算操作数（影响算力利用率）。
        path (str): 输出根目录。
    """
    if npus_count <= 0:
        raise ValueError("npus_count must be > 0")
    if tensor_size_mb <= 0:
        raise ValueError("tensor_size_mb must be > 0")
    if num_ops <= 0:
        raise ValueError("num_ops must be > 0")

    et_path = os.path.join(path, "compute", f"{npus_count}npus_{tensor_size_mb}MB")
    os.makedirs(et_path, exist_ok=True)

    tensor_size_bytes = tensor_size_mb * 1024 * 1024

    for npu in range(npus_count):
        et_filename = os.path.join(et_path, f"compute.{npu}.et")
        with open(et_filename, "wb") as et:
            # metadata
            encode_message(et, GlobalMetadata(version="0.0.4"))

            node = ChakraNode()
            node.id = npu  # 简单使用 rank 作为 node id
            node.name = f"compute_{npus_count}npus_{tensor_size_mb}MB"
            node.type = COMP_NODE
            node.attr.append(ChakraAttr(name="is_cpu_op", bool_val=False))
            node.attr.append(ChakraAttr(name="tensor_size", int64_val=tensor_size_bytes))
            node.attr.append(ChakraAttr(name="num_ops", int64_val=num_ops))

            encode_message(et, node)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npus-count", type=int, required=True)
    parser.add_argument("--tensor-size-mb", type=int, required=True)
    parser.add_argument("--num-ops", type=int, required=True,
                        help="Total ops per node (e.g., 1000000000)")
    args = parser.parse_args()

    generate_compute(
        npus_count=int(args.npus_count),
        tensor_size_mb=int(args.tensor_size_mb),
        num_ops=int(args.num_ops),
        path="./",
    )


if __name__ == "__main__":
    main()
