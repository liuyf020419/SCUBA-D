# !/bin/bash

testlist=demo_json/demo_list_f

CUDA_VISIBLE_DEVICES=0 python3.8 inference_par.py \
    --test_list ${testlist} \
    --max_sample_num 10000 \
    --write_pdbfile \
    --batch_size 1 \
    --sample_from_raw_pdbfile \
    --diff_noising_scale 0.1 \