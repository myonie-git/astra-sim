cd /data/ros/astra-sim/extern/graph_frontend
conda activate chakra
export NPROC=2
export OUT=ddp_trace_out
export CUDA_VISIBLE_DEVICES=1,2

torchrun --standalone --nproc_per_node=${NPROC} \
  resnet18_ddp_chakra.py \
  --out_dir ${OUT} \
  --steps 20 \
  --trace_wait 1 --trace_warmup 1 --trace_active_steps 10 \
  --batch_size 8

OUTABS=$(realpath ${OUT})

for r in $(seq 0 $((NPROC-1))); do
  chakra_trace_link \
    --rank ${r} \
    --chakra-host-trace ${OUTABS}/pytorch_et_rank${r}.json \
    --chakra-device-trace ${OUTABS}/kineto_rank${r}.json \
    --output-file ${OUTABS}/linked_rank${r}.json
done
mkdir -p "$OUTABS/astra_workload"

for r in 0 1; do
  chakra_converter PyTorch \
    --input "$OUTABS/linked_rank${r}.json" \
    --output "$OUTABS/astra_workload/chakra_et.${r}.et"
done