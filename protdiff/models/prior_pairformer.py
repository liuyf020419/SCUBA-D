import logging
import os
import time

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .protein_geom_utils import generate_pair_from_pos, add_c_beta_from_crd, preprocess_pair_feature, preprocess_pair_feature_advance, get_descrete_feature, get_descrete_dist

from .prior_encoder_module import PairEncoder

from .folding_af2.ipa_rigid_net import StructureModule
from .folding_af2 import r3, residue_constants, quat_affine
from .folding_af2.ipa_2d_net import EvoformerPairBlock_
from .folding_af2.all_atom import atom37_to_frames

from .protein_utils import rigid
from .protein_utils.backbone import backbone_fape_loss, structural_violation_loss, backbone_frame_to_atom3_std
from .protein_utils.add_o_atoms import add_atom_O

from .dist_pred import DistogramClassifier
from .esm.gvp_pred_module import GVPSinglePred
from .train_utils import mask_loss, merge_all

logger = logging.getLogger(__name__)


STD_RIGID_COORD = torch.FloatTensor(
    [[-0.525,  1.363,  0.000],
     [ 0.000,  0.000,  0.000],
     [ 1.526, -0.000, -0.000],
     [ 0.627,  1.062,  0.000]]
)
restype_order_num_to_restye = {v: k for k, v in residue_constants.restype_order.items()}


class PriorPairNet(nn.Module):
    def __init__(self, model_config, global_config, data_config):
        super().__init__()
        
        self.config = model_config
        self.global_config = global_config
        self.data_config = data_config

        self.noising_mode_dict = {
            'SCUBA_noising': 0, 
            'CG_map_noising': 1, 
            'FG_map_noising': 2,
            'white_noising': 3,
            'white_ss_noising': 4,
            "seq_denoising": 5,
            "condition_white_noising": 6,
            "cdr_reconstruction": 7}
                                            
        self.pair_encoder = PairEncoder(
            self.config.refine_net.pair_encoder, 
            self.global_config,
            self.config.refine_net.pair_encoder.encode_FG)

        self.denoising_2d_block = EvoformerPairBlock_(
            self.config.refine_net.denoising_2d_block, global_config, 
            self.config.refine_net.pair_encoder.pair_channel, 
            self.config.refine_net.denoising_2d_block.pair_channel)

        self.structure_module_2d = StructureModule(
            self.config.refine_net.structure_module_2d, self.global_config)

        ## auxillary loss
        if self.global_config.loss_weight.prior_ditogram_classify_loss > 0.0:
            self.distogram_pred_config = self.config.refine_net.distogram_pred
            self.distogram_predictor = DistogramClassifier(
                self.distogram_pred_config, 
                self.config.refine_net.denoising_2d_block.pair_channel
                )

        if self.global_config.loss_weight.prior_esm_single_pred_loss > 0.0:
            self.esm_single_pred_config = self.config.refine_net.esm_single_pred
            self.esm_single_predictor = GVPSinglePred(
                self.esm_single_pred_config.gvp_esm_single_pred_config, 
                self.global_config.esm_num
                )


    def forward(self, batch):
        batchsize, L, N, _ = batch['gt_pos'].shape
        device = batch['gt_pos'].device

        noising_mode = self.data_config.common.noising_mode
        noising_mode_idx = 0
        if "+" in noising_mode:
            tmp_noising_mode_idx = [self.noising_mode_dict[mode] for mode in noising_mode.split("+")]
            noising_mode_idx = np.random.choice(tmp_noising_mode_idx)
        else:
            assert noising_mode in list(self.noising_mode_dict.keys())
            noising_mode_idx = self.noising_mode_dict[noising_mode]

        batch['noising_mode_idx'] = noising_mode_idx
        logger.info(f'noising mode: {noising_mode_idx}')

        condition = None
        if noising_mode_idx != 0:
            if noising_mode_idx in [6, 2]:
                inpainting_mask_modes = np.random.randint(0, 3, 1)
                condition = gen_batch_inpainting_condition(
                    self.data_config.condition_fix, batch['gt_pos'][..., :4, :], L, inpainting_mask_modes)
                
            self.merge_pos_frame_data(
                batch, noising_mode=noising_mode_idx, condition=condition)

        single, pair, frame = merge_all(batch)

        if batch['noising_mode_idx'] in [0, 2, 3, 4, 5, 6]:
            if batch['noising_mode_idx'] == 5:
                pair_source = batch['gt_pos']
            else:
                pair_source = batch['traj_pos']
            geom_pair = generate_pair_from_pos(pair_source[..., :4, :])
        else:
            raise ValueError(f'mode: {noising_mode_idx} is not availabll')

        pair_config = self.config.refine_net.pair_encoder
        geom_pair = preprocess_pair_feature_advance(
            geom_pair, rbf_encode=pair_config.rbf_encode, 
            num_rbf=pair_config.num_rbf, 
            tri_encode=pair_config.tri_encode, 
            tri_num=pair_config.tri_num)

        pair_act = self.pair_encoder(
            pair, batch['pair_mask'], geom_pair, condition=condition)
        single_act = torch.ones(batchsize, L, self.config.refine_net.structure_module_2d.single_channel).to(device)
        rep_frame = frame['traj_backbone_frame']

        representations = {
            "single": single_act,
            "pair": pair_act,
            "frame": rep_frame,
            "pair_mask": batch["pair_mask"],
            "seq_mask": batch["seq_mask"]}

        act_pair = self.denoising_2d_block(representations['pair'], representations['pair_mask'])[0]
        representations['pair'] = act_pair
        pred_dict = self.structure_module_2d(representations=representations)

        if self.global_config.loss_weight.prior_ditogram_classify_loss > 0.0:
            pred_map_descrete = self.distogram_predictor(
                representations['pair'] + torch.swapaxes(representations['pair'], -2, -3)
                )
            pred_dict['pred_map_descrete'] = pred_map_descrete
        
        if condition is None:
            condition = torch.zeros((batchsize, L)).to(device)
        else:
            condition = condition
        return_dict, loss_dict = self.all_loss(
            batch, pred_dict, condition=condition)

        loss_dict = mask_loss(batch['loss_mask'], loss_dict)
        batch['condition'] = condition

        return return_dict, loss_dict


    def process(self, batch, single, pair, frame, noising_mode_idx, condition=None):
        batchsize, L, N, _ = batch['traj_pos'].shape
        aatype = batch['aatype']

        if noising_mode_idx in [0, 2, 3, 4, 5, 6]:
            geom_pair = generate_pair_from_pos(batch['traj_pos'][..., :4, :])
        else:
            raise ValueError(f'mode: {noising_mode_idx} is not available')
            
        pair_config = self.config.refine_net.pair_encoder
        geom_pair = preprocess_pair_feature_advance(
            geom_pair, rbf_encode=pair_config.rbf_encode, 
            num_rbf=pair_config.num_rbf, 
            tri_encode=pair_config.tri_encode, 
            tri_num=pair_config.tri_num)
        
        pair_act = self.pair_encoder(
            pair, batch['pair_mask'], geom_pair, condition=condition)
        single_act = torch.ones(
            batchsize, L, self.config.refine_net.structure_module_2d.single_channel).to(aatype.device)

        representations = {
            "single": single_act,
            "pair": pair_act,
            "frame": frame['traj_backbone_frame'],
            "pair_mask": batch["pair_mask"],
            "seq_mask": batch["seq_mask"]}

        act_pair = self.denoising_2d_block(representations['pair'], representations['pair_mask'])[0]
        representations['pair'] = act_pair
        pred_dict = self.structure_module_2d(representations=representations)

        pred_affine = pred_dict['traj']
        pred_dict = traj_dict = get_coord_from_pred_affine(pred_affine, -1)
        
        traj_dict = {
            'coord': torch.stack([traj_dict['coord']], 0),
            'rot': torch.stack([traj_dict['rot']], 0),
            'trans': torch.stack([traj_dict['trans']], 0)
        }
        pred_esm_single = self.esm_single_predictor(batch, traj_dict)

        pred_dict.update({'esm': pred_esm_single})

        traj_affine = r3.rigids_to_quataffine_m(r3.rigids_from_tensor_flat12(batch['traj_backbone_frame'])).to_tensor()
        batch['traj_affine'] = traj_affine
        batch['condition'] = condition

        return pred_dict


    def all_loss(self, batch, pred_dict, condition=None):
        return_dict = {}
        pred_affine = pred_dict['traj']
        
        traj_num, batchsize, nres, _ = pred_affine.shape
        gt_pos = batch['gt_pos']
        gt_affine = r3.rigids_to_quataffine_m(r3.rigids_from_tensor_flat12(batch['gt_backbone_frame'])).to_tensor()
        batch['gt_affine'] = gt_affine

        loss_dict = {}
        loss_unclamp, loss_clamp = [], []
        traj_pre_pos = []
        trans_list, rot_list = [], []
        for traj_idx in range(traj_num):
            quat = pred_affine[traj_idx, ..., :4]
            trans = pred_affine[traj_idx, ..., 4:]
            rot = rigid.quat_to_rot(quat)

            pred_pos = backbone_frame_to_atom3_std(
                            torch.reshape(rot, (-1, 3, 3)), 
                            torch.reshape(trans, (-1, 3))
            )
            pred_pos = torch.reshape(pred_pos, (batchsize, nres, 3, 3))
            pred_pair = generate_pair_from_pos(add_c_beta_from_crd(pred_pos))
            pred_pair = preprocess_pair_feature(pred_pair)

            fape_mask = batch['affine_mask']
            rot_list.append(rot)
            trans_list.append(trans)

            affine_p = torch.where(condition[..., None] == 1, gt_affine[:, :, 0], pred_affine[traj_idx])
            coord_p = torch.where(condition[..., None, None] == 1, gt_pos[:, :, :3], pred_pos)
            traj_pre_pos.append(pred_pos)
            fape, fape_clamp = self.fape_loss(
                affine_p, coord_p, gt_affine[:, :, 0], gt_pos[:, :, :3], fape_mask)

            loss_unclamp.append(fape)
            loss_clamp.append(fape_clamp)

        loss_unclamp = torch.stack(loss_unclamp)
        loss_clamp = torch.stack(loss_clamp)

        clamp_weight = self.global_config.fape.clamp_weight
        loss = loss_unclamp * (1.0 - clamp_weight) + loss_clamp * clamp_weight
        
        last_loss = loss[-1]
        traj_loss = loss.mean()
        traj_weight = self.global_config.fape.traj_weight
        loss = last_loss + traj_weight * traj_loss

        loss_dict['fape_loss'] = loss
        loss_dict['clamp_fape_loss'] = loss_clamp[-1]
        loss_dict['unclamp_fape_loss'] = loss_unclamp[-1]
        loss_dict['last_loss'] = last_loss
        loss_dict['traj_loss'] = traj_loss

        
        if self.global_config.loss_weight.prior_ditogram_classify_loss > 0.0:
            pred_maps_descrete = pred_dict['pred_map_descrete']
            distogram_list = []
            if self.distogram_pred_config.pred_all_dist:
                if self.distogram_pred_config.atom3_dist:
                    dist_type_name = ['ca-ca', 'n-n', 'c-c', 'ca-n', 'ca-c', 'n-c']
                else:
                    if self.distogram_pred_config.ca_dist:
                        dist_type_name = ['ca-ca']
                    else:
                        dist_type_name = ['cb-cb']

                for dist_type_idx, dist_type in enumerate(dist_type_name):
                    gt_map_descrete = get_descrete_dist(batch['gt_pos'], dist_type, self.distogram_pred_config.distogram_args)
                    dim_start = (dist_type_idx) * self.distogram_pred_config.distogram_args[-1]
                    dim_end = (dist_type_idx + 1) * self.distogram_pred_config.distogram_args[-1]
                    pred_map = pred_maps_descrete[..., dim_start: dim_end]
                    distogram_loss = F.cross_entropy(
                        pred_map.reshape(-1, self.distogram_pred_config.distogram_args[-1]), 
                        gt_map_descrete.reshape(-1), reduction='none'
                        ).reshape(batchsize, nres, nres)
                    distogram_list.append(distogram_loss)
            
            else:
                descrete_pair, all_angle_masks = get_descrete_feature(
                    batch['gt_pos'][..., :4, :], return_angle_mask=True, mask_base_ca=False)
                gt_descrete_pair = descrete_pair[..., 1:].long() # remove ca dist map

                for pair_idx in range(4):
                    if pair_idx in [0, 1, 2]:
                        bin_num = self.distogram_pred_config.distogram_args[-1]
                        dim_start = (pair_idx) * self.distogram_pred_config.distogram_args[-1]
                        dim_end = (pair_idx + 1) * self.distogram_pred_config.distogram_args[-1]
                        pred_map = pred_maps_descrete[..., dim_start: dim_end]
                    else:
                        bin_num = self.distogram_pred_config.distogram_args[-1]//2
                        pred_map = pred_maps_descrete[..., -bin_num:]
                    distogram_loss = F.cross_entropy(
                        pred_map.reshape(-1, bin_num), 
                        gt_descrete_pair[..., pair_idx].reshape(-1), reduction='none'
                        ).reshape(batchsize, nres, nres)

                    if pair_idx in [1, 2, 3]:
                        distogram_loss = distogram_loss * all_angle_masks
                    distogram_list.append(distogram_loss)

            distogram_loss = torch.stack(distogram_list).mean(0)
            distogram_loss = distogram_loss * batch['pair_mask']
            loss_dict['ditogram_classify_loss'] = torch.sum(distogram_loss) / (torch.sum(batch['pair_mask']) + 1e-6)

        if self.global_config.loss_weight.prior_esm_single_pred_loss > 0.0:
            true_norm_esm_single = batch['norm_esm_single'] # B, L, D
            coord_dict = {
                'coord': torch.stack(traj_pre_pos, 0),
                'rot': torch.stack(rot_list, 0),
                'trans': torch.stack(trans_list, 0)
            }
            
            pred_esm_single = self.esm_single_predictor(batch, coord_dict)
            
            return_dict['esm'] = pred_esm_single
            return_dict['coord'] = coord_dict['coord'][-1]
            return_dict['trans'] = coord_dict['trans'][-1]
            return_dict['rot'] = coord_dict['rot'][-1]
            return_dict['affine'] = pred_affine[-1]

            esm_single_error = F.mse_loss(pred_esm_single, true_norm_esm_single, reduction='none') # B, L, D
            esm_single_mask = (batch['seq_mask'] * batch['esm_single_mask'])[..., None].repeat(1,1,self.global_config.esm_num)
            esm_single_error = esm_single_error * esm_single_mask
            loss_dict['esm_single_pred_loss'] = torch.sum(esm_single_error) / (torch.sum(esm_single_mask) + 1e-6)

        return return_dict, loss_dict


    def sampling(self, batch, noising_mode_idx, condition=None):
        batch['noising_mode_idx'] = [noising_mode_idx]
        batchsize, L, N, _ = batch['traj_pos'].shape
        device = batch['traj_pos'].device
        # logger.info(f'noising mode: {noising_mode_idx}')

        if condition is None:
            condition = torch.zeros((batchsize, L)).long().to(device)
        else:
            condition = condition[None].long().to(device)

        if noising_mode_idx == 5:
            batch["traj_pos"] = batch["gt_pos"]
            batch["traj_backbone_frame"] = batch["gt_backbone_frame"]
            noising_mode_idx = 0

        ## main code 
        if noising_mode_idx != 0:
            self.merge_pos_frame_data(
                batch, noising_mode=noising_mode_idx, condition=condition)

        single, pair, frame = merge_all(batch)

        pred_dict = self.process(
            batch, single, pair, frame, noising_mode_idx, condition)

        return pred_dict


    def fape_loss(self, affine_p, coord_p, affine_0, coord_0, mask, mask_2d=None, return_nosum=False):
        quat_0 = affine_0[..., :4]
        trans_0 = affine_0[..., 4:]
        rot_0 = rigid.quat_to_rot(quat_0)

        quat_p = affine_p[..., :4]
        trans_p = affine_p[..., 4:]
        rot_p = rigid.quat_to_rot(quat_p)
        fape_tuple = backbone_fape_loss(
                coord_p, rot_p, trans_p,
                coord_0, rot_0, trans_0, mask,
                clamp_dist=self.global_config.fape.clamp_distance,
                length_scale=self.global_config.fape.loss_unit_distance,
                mask_2d=mask_2d, return_nosum=return_nosum
            )

        return fape_tuple
            

    def merge_pos_frame_data(self, data_dict: dict, noising_mode=2, loop_recon=False, condition=None):
        # import pdb; pdb.set_trace()
        if noising_mode == 1: # CG_map_noising
            raise ValueError(f'CG_map_noising is not avialable')
        
        
        elif noising_mode == 2: # FG_map_noising
            if data_dict.__contains__('gt_pos'):
                gt_pos = data_dict['gt_pos']
            else:
                gt_pos = data_dict['traj_pos']
            device= gt_pos.device
            batchsize, L = gt_pos.shape[:2]
            white_noise_scale = self.data_config.white_noise.white_noise_scale

            noising_quat = rigid.rand_quat([batchsize, L]).to(device)
            noising_coord = torch.randn(batchsize, L, 3).to(device) * white_noise_scale
            noising_affine = torch.cat([noising_quat, noising_coord], -1)
            # import pdb; pdb.set_trace()
            noising_pos = add_c_beta_from_crd(
                rigid.affine_to_pos(noising_affine.reshape(-1, 7)).reshape(batchsize, L, -1, 3), add_O=True)
            traj_pos = torch.where(condition[..., None, None] == 1, gt_pos, noising_pos)
            traj_frame = get_batch_quataffine(traj_pos)


        elif noising_mode == 3: # white res noise
            gt_pos = data_dict['gt_pos']
            batchsize, L = gt_pos.shape[:2]
            ca_noise_scale = self.data_config.white_noise.ca_noise_scale
            if isinstance(ca_noise_scale, list):
                ca_noise_scale = np.random.uniform(ca_noise_scale[0], ca_noise_scale[1], 1)[0]

            gt_affine = r3.rigids_to_quataffine_m(
                r3.rigids_from_tensor_flat12(data_dict["gt_backbone_frame"])).to_tensor()[..., 0, :]
            traj_quat = rigid.noising_quat(gt_affine[..., :4])
            traj_coord = rigid.noising_coord(gt_affine[..., 4:], ca_noise_scale)
            traj_affine = torch.cat([traj_quat, traj_coord], -1)

            traj_pos = add_c_beta_from_crd(
                rigid.affine_to_pos(traj_affine.reshape(-1, 7)).reshape(batchsize, L, -1, 3), add_O=True)
            traj_frame = get_batch_quataffine(traj_pos)
        
        
        elif noising_mode == 4: # white ss noise
            # traj_pos = data_dict['traj_pos_ss']
            # traj_frame = data_dict['traj_backbone_frame_ss']
            traj_pos = data_dict['traj_pos']
            traj_frame = data_dict['traj_backbone_frame']

        elif noising_mode == 5: # white ss noise
            traj_pos = data_dict['gt_pos']
            traj_frame = get_batch_quataffine(traj_pos)

        elif noising_mode == 6: # white ss noise
            gt_pos = data_dict['gt_pos']
            batchsize, L = gt_pos.shape[:2]

            ca_noise_scale = self.data_config.white_noise.ca_noise_scale
            if isinstance(ca_noise_scale, list):
                ca_noise_scale = np.random.uniform(ca_noise_scale[0], ca_noise_scale[1], 1)[0]

            gt_affine = r3.rigids_to_quataffine_m(
                r3.rigids_from_tensor_flat12(data_dict["gt_backbone_frame"])).to_tensor()[..., 0, :]
            traj_quat = rigid.noising_quat(gt_affine[..., :4])
            traj_coord = rigid.noising_coord(gt_affine[..., 4:], ca_noise_scale)
            assert condition is not None
            traj_quat = torch.where(condition[..., None] == 1, gt_affine[..., :4], traj_quat)
            traj_coord = torch.where(condition[..., None] == 1, gt_affine[..., 4:], traj_coord)
            traj_affine = torch.cat([traj_quat, traj_coord], -1)

            traj_pos = add_c_beta_from_crd(
                rigid.affine_to_pos(traj_affine.reshape(-1, 7)).reshape(batchsize, L, -1, 3), add_O=True)
            traj_frame = get_batch_quataffine(traj_pos)

        data_dict["traj_pos"] = traj_pos
        data_dict["traj_backbone_frame"] = traj_frame



def get_coord_from_pred_affine(affine, traj_idx):
    traj_num, batchsize, nres, _ = affine.shape
    quat = affine[traj_idx, ..., :4]
    trans = affine[traj_idx, ..., 4:]
    rot = rigid.quat_to_rot(quat)

    pred_pos = backbone_frame_to_atom3_std(
                    torch.reshape(rot, (-1, 3, 3)), 
                    torch.reshape(trans, (-1, 3))
    )
    pred_pos = torch.reshape(pred_pos, (batchsize, nres, 3, 3))
    traj_dict = {
        'coord': pred_pos,
        'trans': trans,
        'rot': rot,
        'affine': affine[traj_idx]
    }
    return traj_dict


def gen_aatype_random_mask(config, batchsize, seq_len, mask_mode, ca_pos):
    p_rand = config.p_rand # 0.3
    p_lin = config.p_lin
    p_spatial = config.p_spatial

    min_lin_len = int(p_lin[0] * seq_len) # 0.70
    max_lin_len = int(p_lin[1] * seq_len) # 0.95
    lin_len = torch.randint(min_lin_len, max_lin_len, [1]).item()

    min_knn = p_spatial[0] # 0.05
    max_knn = p_spatial[1] # 0.30
    # knn = int((torch.rand([1]) * (max_knn-min_knn) + min_knn).item() * seq_len)
    knn = int(np.random.uniform(min_knn, max_knn, 1) * seq_len)

    if mask_mode == 0: # random
        mask = (torch.rand(batchsize, seq_len) > p_rand).long()

    elif mask_mode == 1: # linear
        mask = torch.zeros(batchsize, seq_len)
        start_index = torch.randint(0, seq_len-lin_len, [batchsize])
        mask_idx = start_index[:, None] + torch.arange(lin_len)
        mask.scatter_(1, mask_idx, torch.ones_like(mask_idx).float())

    elif mask_mode == 2: # spatial
        central_absidx = torch.randint(0, seq_len, [batchsize])
        ca_map = torch.sqrt(torch.sum(torch.square(ca_pos[:, None] - ca_pos[:, :, None]), -1) + 1e-10)
        batch_central_knnid = torch.stack([ca_map[bid, central_absidx[bid]] for bid in range(batchsize)])
        knn_idx = torch.argsort(batch_central_knnid)[:, :knn]
        mask = torch.ones(batchsize, seq_len).to(ca_map.device)
        mask.scatter_(1, knn_idx, torch.zeros_like(knn_idx).float())

    elif mask_mode == 3: # full
        mask = torch.zeros(batchsize, seq_len)

    return mask


def gen_batch_inpainting_condition(config, coords, max_len=None, mask_mode=2, sstype=None):
    batchsize, seq_len = coords.shape[:2]
    device = coords.device

    min_p_rand, max_p_rand = config.p_rand # 0.05, 0.3
    min_block_len, max_block_len = config.random_block_len # [3, 7]
    min_p_lin, max_p_lin = config.p_lin # 0.05, 0.3
    min_knn, max_knn = config.p_spatial # 0.05, 0.3
    unmask_min_knn = 1 - min_knn # 0.95
    unmask_max_knn = 1 - max_knn # 0.7

    if max_len is not None:
        if seq_len < max_len:
            max_len = seq_len
    else:
        max_len = seq_len

    # import pdb; pdb.set_trace()
    if mask_mode == 0: # random min blocks
        block_len = np.random.randint(min_block_len, max_block_len, 1)[0]
        min_rand_num = int(( min_p_rand )/block_len * seq_len) # 0.10/5
        max_rand_num = int(( max_p_rand )/block_len * seq_len) # 0.30/5
        block_num = np.random.randint(min_rand_num, max_rand_num, 1)[0]

        start_index = torch.randint(0, seq_len-block_len, [batchsize, block_num])
        mask_seq = torch.ones(batchsize, seq_len).to(device)
        mask_idx = (start_index[..., None] + torch.arange(block_len)).reshape(batchsize, -1).to(device)
        mask_seq.scatter_(1, mask_idx, torch.zeros_like(mask_idx).float().to(device))

    elif mask_mode == 1: # linear
        min_lin_len = int(( min_p_lin ) * seq_len) # 0.05
        max_lin_len = int(( max_p_lin ) * seq_len) # 0.30
        lin_len = np.random.randint(min_lin_len, max_lin_len, 1)[0]

        start_index = torch.randint(0, seq_len-lin_len, [batchsize])
        mask_seq = torch.ones(batchsize, seq_len).to(device)
        mask_idx = (start_index[:, None] + torch.arange(lin_len)).to(device)
        mask_seq.scatter_(1, mask_idx, torch.zeros_like(mask_idx).float().to(device))

    elif mask_mode == 2: # spatial mask
        # dffferent prob to inpainting task
        ca_coords = coords[..., 1, :]
        ca_dist_pair = torch.sqrt(torch.sum(torch.square(ca_coords[:, :, None] - ca_coords[:, None]), -1) + 1e-10)
        knn = int(np.random.uniform(unmask_max_knn, unmask_min_knn, 1) * seq_len)

        central_absidx = torch.randint(0, max_len, [batchsize])
        batch_central_knnid = torch.stack([ca_dist_pair[bid, central_absidx[bid]] for bid in range(batchsize)])
        knn_idx = torch.argsort(batch_central_knnid)[:, :knn]
        mask_seq = torch.zeros(batchsize, seq_len).to(device)
        mask_seq.scatter_(1, knn_idx, torch.ones_like(knn_idx).float())

    elif mask_mode == 3:
        mask_seq = torch.where(sstype == 1, 0, 1)

    elif mask_mode == 4:
        mask_seq = torch.ones(batchsize, seq_len).to(device)

    else:
        raise NotImplementedError(f'mask_mode: {mask_mode} not implemented')

    inpaiting_condition = mask_seq
    return inpaiting_condition


def get_batch_quataffine(pos):
    batchsize, nres, natoms, _ = pos.shape
    assert natoms == 5
    alanine_idx = residue_constants.restype_order_with_x["A"]
    aatype = torch.LongTensor([alanine_idx] * nres)[None].repeat(batchsize, 1).to(pos.device)
    all_atom_positions = F.pad(pos, (0, 0, 0, 37-5, 0, 0), "constant", 0)
    all_atom_mask = torch.ones(batchsize, nres, 37).to(pos.device)
    frame_dict = atom37_to_frames(aatype, all_atom_positions, all_atom_mask)

    return frame_dict['rigidgroups_gt_frames']
