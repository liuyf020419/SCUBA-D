import logging
import sys, os
from tqdm import trange
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .prior_net import PriorModule
from .ddpm import DDPM
from .folding_af2 import residue_constants
from .folding_af2.all_atom import atom37_to_frames
from .folding_af2 import r3

from .train_utils import mask_loss, merge_all

from .protein_utils.write_pdb import write_multichain_from_atoms
from .protein_utils.add_o_atoms import add_atom_O
from .protein_utils import rigid
from .protein_geom_utils import add_c_beta_from_crd

sys.path.append("../../")
from pdb_utils.calc_dssp import get_feature_from_dssp, preprocess_dssp_df

logger = logging.getLogger(__name__)


def get_sstype_from_coords(pdbfile):
    df = get_feature_from_dssp(pdbfile, file_type='PDB', return_type='df', add_ca_coord=True)
    df = preprocess_dssp_df(df, add_ss_idx=True)
    encoded_sstype = torch.from_numpy(df.loc[:, 'SS3enc'].to_numpy())
    return encoded_sstype


class PriorDDPM(nn.Module):
    def __init__(self, config, global_config, data_config) -> None:
        super().__init__()
        self.config = config
        self.global_config = global_config
        self.data_config = data_config

        self.prior_module = PriorModule(self.config, self.global_config, self.data_config)
        self.diff_module = DDPM(self.config, self.global_config)


    def forward(self, batch):
        losses_dict = {}
        batchsize, L, N, _ = batch['gt_pos'].shape
        make_mask(batch['len'], batchsize, L, batch)

        mu_dict, prior_loss_dict = self.prior_module(batch)
        mu_dict = {k: v.detach() for k, v in mu_dict.items()}
        diff_loss_dict = self.diff_module(batch, mu_dict)
        # if self.global_config.is_Dnet:
        pred_dict, diff_loss_dict = diff_loss_dict

        prior_loss_dict = {'prior_'+k: v for k, v in prior_loss_dict.items()}
        losses_dict.update(prior_loss_dict)
        diff_loss_dict = {'diff_'+k: v for k, v in diff_loss_dict.items()}
        losses_dict.update(diff_loss_dict)

        losses_dict = mask_loss(batch['loss_mask'], losses_dict)
        loss = sum([losses_dict[k].mean() * 
                        self.global_config.loss_weight[k] \
                for k in self.global_config.loss_weight if k in losses_dict.keys()])
        losses_dict['loss'] = loss

        return pred_dict, losses_dict


    @torch.no_grad()
    def sampling(
        self, 
        batch, 
        pdb_prefix, 
        diff_step: int, 
        noising_mode_idx: int, 
        condition=None, 
        return_traj=False, 
        ddpm_fix=False, 
        rigid_fix=False, 
        epoch=1, 
        diff_noising_scale=1.0, 
        iterate_mode=4
        ):
        
        batchsize, L, N, _ = batch['traj_pos'].shape
        device = batch['traj_pos'].device
        make_mask(batch['len'], batchsize, L, batch)

        for epoch_idx in range(epoch):
            mu_dict = self.prior_module.sampling(batch, pdb_prefix, noising_mode_idx, condition, epoch_idx=epoch_idx, return_traj=return_traj)
            diffused_coord_0 = self.diff_module.sampling(
                batch, pdb_prefix, diff_step, mu_dict, return_traj, ddpm_fix=ddpm_fix, rigid_fix=rigid_fix, term_num=epoch_idx, diff_noising_scale=diff_noising_scale)

            last_iter_coords = diffused_coord_0
            last_iter_coords = last_iter_coords - last_iter_coords.mean([1, 2], keepdim=True)
            last_iter_affine = get_batch_quataffine(last_iter_coords[..., :3, :])

            if iterate_mode == 0:
                noising_mode_idx = 0
                batch['traj_pos'] = last_iter_coords
                batch["traj_backbone_frame"] = last_iter_affine
            elif iterate_mode == 4:
                noising_mode_idx = 4
                white_noise_scale = self.data_config.white_noise.white_noise_scale

                sstype = torch.stack([get_sstype_from_coords(b_pdbfile) for b_pdbfile in batch['last_pdbfiles']]).to(device)

                # # add ss translation noise
                noising_quat = rigid.rand_quat([batchsize, L]).to(device)
                noising_coord = torch.randn(batchsize, L, 3).to(device) * white_noise_scale
                noising_affine = torch.cat([noising_quat, noising_coord], -1)
                noising_pos = add_c_beta_from_crd(
                    rigid.affine_to_pos(noising_affine.reshape(-1, 7)).reshape(batchsize, L, -1, 3), add_O=True).to(device)
                
                traj_pos = torch.where(sstype[..., None, None] == 1, noising_pos[..., :4, :], last_iter_coords)
                traj_pos = torch.where(condition.to(device)[..., None, None] == 1, last_iter_coords, traj_pos)
                traj_frame = get_batch_quataffine(traj_pos)
                
                batch['traj_pos_ss'] = traj_pos
                batch['traj_backbone_frame_ss'] = traj_frame
                

def make_noise_from_sstype(sstype, noise_scale=2.0):
    ss3type = sstype[0]
    ss_start_indexs = (torch.where((ss3type[1:] - ss3type[:-1]) != 0)[0] + 1).long()
    ss_start_indexs = torch.cat([torch.LongTensor([0]).to(ss3type.device), ss_start_indexs])
    ss_lens = ss_start_indexs[1:] - ss_start_indexs[:-1]
    ss_lens = torch.cat([ss_lens, (len(ss3type) - ss_start_indexs[-1]).unsqueeze(0)])
    pos_noise = torch.cat([(torch.rand(1, 3).repeat(ss_len.item(), 1) + 1) * noise_scale for ss_len in ss_lens]).to(ss3type.device)
    return pos_noise


def make_mask(lengths, batchsize, max_len, batch_dict):
    seq_mask = torch.zeros(batchsize, max_len)
    pair_mask = torch.zeros(batchsize, max_len, max_len)
    # import pdb; pdb.set_trace()
    for idx in range(len(lengths)):
        length = lengths[idx]
        seq_mask[idx, :length] = torch.ones(length)
        pair_mask[idx, :length, :length] = torch.ones(length, length)

    batch_dict['affine_mask'] = seq_mask.to(batch_dict['aatype'].device)
    batch_dict['pair_mask'] = pair_mask.to(batch_dict['aatype'].device)
    batch_dict['seq_mask'] = seq_mask.to(batch_dict['aatype'].device)


def get_batch_quataffine(pos):
    batchsize, nres, natoms, _ = pos.shape
    assert natoms >= 3
    alanine_idx = residue_constants.restype_order_with_x["A"]
    aatype = torch.LongTensor([alanine_idx] * nres)[None].repeat(batchsize, 1).to(pos.device)
    all_atom_positions = F.pad(pos, (0, 0, 0, 37-natoms, 0, 0), "constant", 0)
    all_atom_mask = torch.ones(batchsize, nres, 37).to(pos.device)
    frame_dict = atom37_to_frames(aatype, all_atom_positions, all_atom_mask)

    return frame_dict['rigidgroups_gt_frames']




