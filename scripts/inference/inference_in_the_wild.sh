python mono/tools/test_scale_cano.py \
    'mono/configs/test/beitl16_512_in_the_wild.py' \
    --load-from weights/metricdepth_beit_large_512x512.pth \
    --show-dir demo_data/outputs_beit_in_the_wild \
    --data_root demo_data \
    --in_the_wild_rgb_folder demo_data/rgb \
    --launcher None