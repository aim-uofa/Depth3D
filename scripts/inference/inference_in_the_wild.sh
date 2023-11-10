RGB_FOLDER=$1

python mono/tools/test_scale_cano.py \
    'mono/configs/test/beitl16_512_in_the_wild.py' \
    --load-from weights/metricdepth_beit_large_512x512.pth \
    --show-dir outputs_beit_in_the_wild \
    --in_the_wild_rgb_folder $RGB_FOLDER \
    --launcher None
