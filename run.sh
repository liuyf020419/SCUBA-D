# !/bin/bash
testlist=demo_json/demo_list

CUDA_VISIBLE_DEVICES=1 python3.8 inference_par.py \
    --test_list ${testlist} \
    --write_pdbfile \
    --batch_size 1 \
    --sample_from_raw_pdbfile \
    --diff_noising_scale 0.1 \