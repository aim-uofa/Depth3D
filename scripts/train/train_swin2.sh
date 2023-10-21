export NNODES=1
export NODE_RANK=0
echo "Number nodes: ${NNODES}"
echo "NODE_RANK: ${NODE_RANK}"

TORCH_DISTRIBUTED_DEBUG=INFO python mono/tools/train.py \
        'mono/configs/DPTHead/swin2_384.py' \
        --load-from pretrained_weights/dpt_swin2_large_384.pt \
        --launcher slurm \
        --use-tensorboard 