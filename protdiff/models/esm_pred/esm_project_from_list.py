import os

import torch
import numpy as np

from esm_projection import predict_aatype, load_esm1b_projection_head, af2_index_to_aatype, calc_simi_ident_seqs


def run(npydir_files: str, fasta_fiilename: str='scuba_r_aatype_pred.fasta'):
    predictor = load_esm1b_projection_head()
    if isinstance(npydir_files, str):
        with open(npydir_files, 'r') as reader:
            npyfiles = [line.strip() for line in reader.readlines()]
    elif isinstance(npydir_files, list):
        npyfiles = npydir_files
    name_list, pred_seq_list, simi_list, ident_list = [], [], [], []

    for f_idx, npyfile in enumerate(npyfiles):
        name = os.path.basename(npyfile).split('.')[-2] + '_' + str(f_idx)
        name_list.append(name)
        print(name)

        pred_esm1b_dict = np.load(npyfile, allow_pickle=True)
        pred_esm1b = pred_esm1b_dict
        # pred_esm1b = pred_esm1b_dict['esm1b']
        # gt_aatype_af2idx = pred_esm1b_dict['gt_aatype'].reshape((-1, )).tolist()
        # gt_aatype = ''.join([af2_index_to_aatype[aa] for aa in gt_aatype_af2idx])

        pred_aatype = predict_aatype(predictor, torch.from_numpy(pred_esm1b))
        # sim, ident = calc_simi_ident_seqs([gt_aatype, pred_aatype])
        pred_seq_list.append(pred_aatype)
        # simi_list.append(sim[0])
        # ident_list.append(ident[0])

    with open(fasta_fiilename, 'w') as writer:
        for f_idx in range(len(name_list)):
            # fasta_comment = ','.join([ name_list[f_idx], 'similarity: '+str(round(simi_list[f_idx], 3)), 'identity: '+str(round(ident_list[f_idx], 3)) ])
            fasta_comment = name_list[f_idx]
            fasta_seq = pred_seq_list[f_idx]
            writer.write(f'>{fasta_comment}\n{fasta_seq}\n')

    

if __name__ == '__main__':
    npy_root = '/train14/superbrain/yfliu25/structure_refine/debug_fixprior_fixaffine_evoformer2/test_only_esm1b'
    npydir_files = [f'{npy_root}/{protein}/{protein}_esm1b_pred.npy' for protein in os.listdir(npy_root)]
    # fasta_fiilename = None
    run(npydir_files)