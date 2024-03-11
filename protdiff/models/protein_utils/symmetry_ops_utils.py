import numpy as np
import torch
import torch.nn.functional as F

from .rigid import quat_to_rot, rot_to_quat
from .backbone import backbone_frame_to_atom3_std, coord_to_frame_affine


def get_assemly_xt_from_au_xt_in_ori(au_xt_dict_in_ori, au_trans, au_rots):
    au_num = au_rots.shape[0]
    assembly_xt_dict = {}

    if au_xt_dict_in_ori.__contains__('esm'):
        assembly_esm = au_xt_dict_in_ori['esm'].repeat(1, au_num, 1)
        assembly_xt_dict['esm'] = assembly_esm

    if au_xt_dict_in_ori.__contains__('affine'):
        au_affine = au_xt_dict_in_ori['affine']
        batch_size, res_num = au_affine.shape[:2]
        assemblly_affine_list = []

        ori_quat = au_affine[..., :4]
        ori_trans = au_affine[..., 4:]
        ori_rot = quat_to_rot(ori_quat)

        ori_coord = backbone_frame_to_atom3_std(
            torch.reshape(ori_rot, (-1, 3, 3)),
            torch.reshape(ori_trans, (-1, 3)),
        ).reshape(batch_size, res_num, 3, 3)
        
        for au_idx in range(au_num):
            au_tran = au_trans[au_idx]
            au_rot = au_rots[au_idx]

            au_sym_coord = (ori_coord @ au_rot.float())# + au_tran
            au_sym_affine = coord_to_frame_affine(au_sym_coord)['affine']
            
            assemblly_affine_list.append(au_sym_affine)

        assemblly_affine = torch.cat(assemblly_affine_list, -2)
        assembly_xt_dict['affine'] = assemblly_affine.float()

    return assembly_xt_dict



def get_au_centriod_trans(affine7, au_length):
    batch_size, res_num = affine7.shape[:2]
    au_centroid_trans_list = []

    au_num = res_num // au_length
    for au_idx in range(au_num):
        au_start_idx = au_idx * au_length
        au_end_idx = (au_idx + 1) * au_length
        au_all_trans = affine7[..., au_start_idx: au_end_idx, 4:]
        au_reduce_trans = au_all_trans.mean(-2)
        au_centroid_trans_list.append(au_reduce_trans)

    return torch.stack(au_centroid_trans_list)



def get_assembly_batch_from_au_batch(au_batch, au_num, gap_size=100, resrange=(-32, 32), resmask_num=33, chainrange=(-4, 4), chainmask_num=5):
    au_single_res_rel = au_batch['single_res_rel']
    device = au_single_res_rel.device
    batch_size, res_num = au_single_res_rel.shape

    au_single_res_rel_end = au_single_res_rel[:, -1]
    assembly_single_res_rel_idx = torch.cat([ 
        au_idx * (au_single_res_rel_end + gap_size) + au_single_res_rel 
            for au_idx in range(au_num) 
        ], -1)[0]

    assembly_pair_res_rel_idx = assembly_single_res_rel_idx[:, None] - assembly_single_res_rel_idx

    assembly_unclip_single_chain_rel_idx = torch.arange(au_num).reshape(-1, 1).repeat(1, res_num).reshape((-1, )).to(device)
    assembly_pair_chain_rel_idx = assembly_unclip_single_chain_rel_idx[:, None] - assembly_unclip_single_chain_rel_idx

    assembly_pair_res_rel_idx = torch.where(torch.any(torch.stack([assembly_pair_res_rel_idx > resrange[1], 
                            assembly_pair_res_rel_idx < resrange[0]]), 0), resmask_num, assembly_pair_res_rel_idx)

    assembly_pair_chain_rel_idx = torch.where(torch.any(torch.stack([assembly_pair_chain_rel_idx > chainrange[1], 
                            assembly_pair_chain_rel_idx < chainrange[0]]), 0), chainmask_num, assembly_pair_chain_rel_idx)

    assembly_single_res_rel = assembly_single_res_rel_idx[None].repeat(batch_size, 1).to(device)

    assembly_pair_res_rel = (assembly_pair_res_rel_idx - resrange[0])[None].repeat(batch_size, 1, 1).to(device)
    assembly_pair_chain_rel = (assembly_pair_chain_rel_idx - chainrange[0])[None].repeat(batch_size, 1, 1).to(device)

    assembly_seq_mask = au_batch['seq_mask'].repeat(1, au_num).to(device)
    assembly_affine_mask = au_batch['affine_mask'].repeat(1, au_num) .to(device)
    assembly_pair_mask = (assembly_seq_mask[:, :, None] * assembly_seq_mask[:, None]).to(device)

    assembly_condition = au_batch['condition'].repeat(1, au_num).to(device)

    au_batch['au_seq_mask'] = au_batch['seq_mask']
    au_batch['au_pair_mask'] = au_batch['pair_mask']
    au_batch['au_affine_mask'] = au_batch['affine_mask']
    au_batch['au_single_res_rel'] = au_batch['single_res_rel']
    au_batch['au_pair_res_rel'] = au_batch['pair_res_rel']
    au_batch['au_pair_chain_rel'] = au_batch['pair_chain_rel']
    au_batch['au_condition'] = au_batch['condition']

    au_batch['seq_mask'] = assembly_seq_mask
    au_batch['affine_mask'] = assembly_affine_mask
    au_batch['pair_mask'] = assembly_pair_mask
    au_batch['single_res_rel'] = assembly_single_res_rel
    au_batch['pair_res_rel'] = assembly_pair_res_rel
    au_batch['pair_chain_rel'] = assembly_pair_chain_rel
    au_batch['condition'] = assembly_condition

    



