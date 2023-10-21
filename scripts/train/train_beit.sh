export NNODES=1
export NODE_RANK=0
echo "Number nodes: ${NNODES}"
echo "NODE_RANK: ${NODE_RANK}"

TORCH_DISTRIBUTED_DEBUG=INFO python mono/tools/train.py \
        'mono/configs/DPTHead/beitl16_512.py' \
        --load-from pretrained_weights/dpt_beit_large_512.pt \
        --launcher slurm \
        --use-tensorboard 