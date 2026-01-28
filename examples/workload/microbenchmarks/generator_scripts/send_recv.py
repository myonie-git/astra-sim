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
    COMM_RECV_NODE,
    COMM_SEND_NODE,
)
from extern.graph_frontend.chakra.schema.protobuf.et_def_pb2 import (
    AttributeProto as ChakraAttr,
)
from extern.graph_frontend.chakra.schema.protobuf.et_def_pb2 import Node as ChakraNode
from extern.graph_frontend.chakra.src.third_party.utils.protolib import (
    encodeMessage as encode_message,
)


def generate_send_recv(
    npus_count: int,
    msg_size: int,
    src: int = 0,
    dst: int = 1,
    tag: int = 0,
    path: str = "./",
) -> None:
    """
    Generate point-to-point Send/Recv ET files.

    Args:
        npus_count (int): Number of NPUs in the job (>= 2).
        msg_size (int): Message size in MB.
        src (int): Source NPU rank.
        dst (int): Destination NPU rank.
        tag (int): Communication tag to match Send/Recv.
        path (str): Output root path.
    """
    if npus_count < 2:
        raise ValueError("npus_count must be >= 2")
    if src == dst:
        raise ValueError("src and dst must be different")
    if not 0 <= src < npus_count or not 0 <= dst < npus_count:
        raise ValueError("src/dst must be within [0, npus_count)")
    if msg_size <= 0:
        raise ValueError("msg_size must be > 0 (in MB)")

    et_path = os.path.join(path, "send_recv", f"{npus_count}npus_{msg_size}MB")
    os.makedirs(et_path, exist_ok=True)

    size_bytes = msg_size * 1024 * 1024
    node_id = 0

    for npu in range(npus_count):
        et_filename = os.path.join(et_path, f"send_recv.{npu}.et")

        with open(et_filename, "wb") as et:
            # metadata
            encode_message(et, GlobalMetadata(version="0.0.4"))

            # only the participating ranks get nodes; others remain idle
            if npu not in (src, dst):
                continue

            node = ChakraNode()
            node.id = node_id
            node.name = f"send_recv_{npus_count}npus_{msg_size}MB"
            node.type = COMM_SEND_NODE if npu == src else COMM_RECV_NODE
            node.attr.append(ChakraAttr(name="is_cpu_op", bool_val=False))
            node.attr.append(ChakraAttr(name="comm_src", int32_val=src))
            node.attr.append(ChakraAttr(name="comm_dst", int32_val=dst))
            node.attr.append(ChakraAttr(name="comm_size", int64_val=size_bytes))
            node.attr.append(ChakraAttr(name="comm_tag", int32_val=tag))

            encode_message(et, node)

            # increment only when an actual node is emitted
            node_id += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npus-count", type=int, required=True)
    parser.add_argument("--msg-size", type=int, required=True,
                        help="Message size in MB")
    parser.add_argument("--src", type=int, default=0)
    parser.add_argument("--dst", type=int, default=1)
    parser.add_argument("--tag", type=int, default=0)
    args = parser.parse_args()

    generate_send_recv(
        npus_count=int(args.npus_count),
        msg_size=int(args.msg_size),
        src=int(args.src),
        dst=int(args.dst),
        tag=int(args.tag),
        path="./",
    )


if __name__ == "__main__":
    main()
