export NNODES=1
export NODE_RANK=0
echo "Number nodes: ${NNODES}"
echo "NODE_RANK: ${NODE_RANK}"

# The pre-trained weight will be loaded according to the path in "Depth3D/data_info/pretrained_weight.py"
TORCH_DISTRIBUTED_DEBUG=INFO python mono/tools/train.py \
        'mono/configs/HourglassDecoder/convlarge_544x1216_0.1_150.py' \
        --launcher slurm \
        --use-tensorboard 