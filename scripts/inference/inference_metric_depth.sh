DATA_ROOT=None

TEST_ANNO_PATH=$1
if [ $# -ge 2 ]; then
  DATA_ROOT=$2
fi

# echo $TEST_ANNO_PATH
# echo $DATA_ROOT

python mono/tools/test_scale_cano.py \
    'mono/configs/test/beitl16_512_in_the_wild.py' \
    --load-from weights/metricdepth_beit_large_512x512.pth \
    --show-dir outputs_beit_metric_depth \
    --test_anno_path $TEST_ANNO_PATH \
    --data_root $DATA_ROOT \
    --launcher None
