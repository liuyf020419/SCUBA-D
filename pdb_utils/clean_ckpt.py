import os, sys
import torch

def run(tb_cleaned_ckpt, cleaned_ckpt):
    state = torch.load(tb_cleaned_ckpt, map_location='cpu')
    cleaned_state = {}
    for k, v in state.items():
        if k in ['model', 'd_model']:
            cleaned_state[k] = v
            
    with open(cleaned_ckpt, 'wb') as f:
        torch.save(cleaned_state, f)


if __name__ == '__main__':
    tb_cleaned_ckpt = '/home/liuyf/proteins/SCUBA-diff-main-pub/savedir/sketch_mask90_domain_tune_noceaatype_minbatch_sketch_mask/checkpoint/checkpoint_last.pt'
    cleaned_ckpt = '/home/liuyf/proteins/SCUBA-diff-main-pub/savedir/sketch_mask90_domain_tune_noceaatype_minbatch_sketch_mask/checkpoint/checkpoint_clean.pt'
    run(tb_cleaned_ckpt, cleaned_ckpt)