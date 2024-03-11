import logging
from tqdm import trange

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from .protein_utils.rigid import rigid_from_3_points, rot_to_quat, quat_to_axis_angles, axis_angle_to_pos, quat_to_rot, rand_quat, pos_to_affine7, affine6_to_affine7, affine7_to_affine6, affine_to_pos
from .protein_utils.backbone import backbone_frame_to_atom3_std, backbone_fape_loss, coord_to_frame_affine
from .protein_geom_utils import get_descrete_dist, get_descrete_feature, add_c_beta_from_crd
from .protein_utils.covalent_loss import structural_violation_loss
from .protein_utils.add_o_atoms import add_atom_O
from .protein_geom_utils import get_internal_angles
from .folding_af2 import r3

from .protein_utils.write_pdb import write_multichain_from_atoms
from .diff_evoformeripa import EvoformerIPA
from .protein_utils.pyalign import KabschCycleAlign

from .esm_pred.esm_projection import predict_aatype, load_esm1b_projection_head, af2_index_to_aatype, calc_simi_ident_seqs

logger = logging.getLogger(__name__)

kabschalign = KabschCycleAlign()
predictor = load_esm1b_projection_head()


class DDPM(nn.Module):
    def __init__(self, config, global_config) -> None:
        super().__init__()
        self.config = config
        self.global_config = global_config

        beta_start, beta_end = global_config.diffusion.betas
        T = global_config.diffusion.T
        self.T = T
        betas = torch.linspace(beta_start, beta_end, T, dtype=torch.float32)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))
        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))

        # Calculations for posterior q(y_{t-1} | y_t, y_0)
        posterior_variance = betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)
        posterior_variance = torch.stack([posterior_variance, torch.FloatTensor([1e-20] * self.T)])
        posterior_log_variance_clipped = posterior_variance.max(dim=0).values.log()
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        posterior_mean_coef1 = betas * alphas_cumprod_prev.sqrt() / (1 - alphas_cumprod)
        posterior_mean_coef2 = (1 - alphas_cumprod_prev) * alphas.sqrt() / (1 - alphas_cumprod)
        posterior_mean_coef3 = (1 - (betas * alphas_cumprod_prev.sqrt() + alphas.sqrt() * (1 - alphas_cumprod_prev))/ (1 - alphas_cumprod)) # only for mu from prior
        self.register_buffer('posterior_log_variance_clipped', posterior_log_variance_clipped)
        self.register_buffer('posterior_mean_coef1', posterior_mean_coef1)
        self.register_buffer('posterior_mean_coef2', posterior_mean_coef2)
        self.register_buffer('posterior_mean_coef3', posterior_mean_coef3) # only for mu from prior

        self.x0_pred_net = EvoformerIPA(config.diff_model, global_config)

        self.position_scale = 0.0667
        self.affine_scale = torch.FloatTensor([1.0, 1.0, 1.0, 1.0] + [1.0 / self.position_scale] * 3)
        self.affine_tensor_scale = torch.FloatTensor([1.0, 1.0, 1.0] + [1.0 / self.position_scale] * 3)


    def q_sample(self, x0_dict: dict, t, mu_dict=None, condition=None, gt_affine7=None, noising_scale=1.0):
        # Calculations for posterior q(x_{t} | x_0, mu)
        xt_dict = {}
        if x0_dict.__contains__('esm'):
            xt_esm = self.degrad_esm(x0_dict['esm'], t, mu_dict if mu_dict is not None else None, noising_scale=noising_scale)
            xt_dict['esm'] = xt_esm

        if x0_dict.__contains__('affine'):
            xt_affine = self.degrad_affine(x0_dict['affine'], t, mu_dict if mu_dict is not None else None, condition, gt_affine7, noising_scale=noising_scale)
            xt_dict['affine'] = xt_affine

        return xt_dict


    def q_posterior(self, xt_dict, x0_dict, t, mu_dict=None):
        # Calculations for posterior q(x_{t-1} | x_t, x_0, mu)
        q_posterior_dict = {}
        if mu_dict is None:
            mu_dict = {k: 0. for k in xt_dict.keys()}
        for k in xt_dict.keys():
            if k != 'affine':
                posterior_mean = self.posterior_mean_coef1[t] * x0_dict[k] + self.posterior_mean_coef2[t] * xt_dict[k]
                model_log_variance = self.posterior_log_variance_clipped[t]
                posterior_mean = posterior_mean + mu_dict[k] * self.posterior_mean_coef3[t]

                eps = torch.randn_like(posterior_mean) if t > 0 else torch.zeros_like(posterior_mean)
                x_t_1 = posterior_mean + eps * (0.5 * model_log_variance).exp()
                q_posterior_dict[k] = x_t_1
            else:
                x0_affine6 = affine7_to_affine6(x0_dict[k])
                xt_affine6 = affine7_to_affine6(xt_dict[k])
                mu_affine6 = affine7_to_affine6(mu_dict[k])

                posterior_mean = self.posterior_mean_coef1[t] * x0_affine6 + self.posterior_mean_coef2[t] * xt_affine6
                model_log_variance = self.posterior_log_variance_clipped[t]
                posterior_mean = posterior_mean + mu_affine6 * self.posterior_mean_coef3[t]

                eps = torch.randn_like(posterior_mean) if t > 0 else torch.zeros_like(posterior_mean)
                x_t_1 = posterior_mean + eps * (0.5 * model_log_variance).exp() * self.affine_tensor_scale.to(x0_affine6.device)
                q_posterior_dict[k] = affine6_to_affine7(x_t_1)

        return q_posterior_dict

    
    def degrad_esm(self, esm_0, t, prior_dict=None, noising_scale=1.0):
        t1 = t[..., None, None]
        if prior_dict is None:
            prior_esm = 0
        else:
            prior_esm = prior_dict['esm']
        noise = torch.randn_like(esm_0)
        degrad_esm = (esm_0 - prior_esm) * self.sqrt_alphas_cumprod[t1] + noise * self.sqrt_one_minus_alphas_cumprod[t1] * noising_scale + prior_esm
        return degrad_esm


    def degrad_affine(self, affine_0, t, prior_dict=None, condition=None, gt_affine7=None, noising_scale=1.0):
        device = affine_0.device
        t1 = t[..., None, None]

        affine_tensor = affine7_to_affine6(affine_0)

        if prior_dict is not None:
            prior_affine_tensor = affine7_to_affine6(prior_dict['affine'])
        else:
            prior_affine_tensor = 0

        noise = torch.randn_like(affine_tensor) * self.sqrt_one_minus_alphas_cumprod[t1] * noising_scale * self.affine_tensor_scale.to(device)
        degraded_affine_tensor = (affine_tensor - prior_affine_tensor) * self.sqrt_alphas_cumprod[t1] + noise + prior_affine_tensor

        degraded_affine = affine6_to_affine7(degraded_affine_tensor)

        if condition is not None:
            assert gt_affine7 is not None
            degraded_affine = torch.where(condition[..., None]==1, gt_affine7[..., 0, :], degraded_affine)

        return degraded_affine
    

    def forward(self, batch: dict, mu_dict: dict=None):
        affine_0 = r3.rigids_to_quataffine_m(r3.rigids_from_tensor_flat12(batch['gt_backbone_frame'])).to_tensor()[..., 0, :]
        device = batch['gt_pos'].device
        batch_size = batch['gt_pos'].shape[0]

        t = torch.randint(0, self.T, (batch_size,), device=device).long()
        batch['t'] = t
        batch['sqrt_alphas_cumprod'] = self.sqrt_alphas_cumprod[t]
        x0_dict = {
            'affine': affine_0,
            'esm': batch['norm_esm_single']
        }
        batch['x0_dict'] = x0_dict
     
        if mu_dict is not None:
            xt_dict = self.q_sample(batch['x0_dict'], t, mu_dict, batch['condition'], batch['gt_affine'])
        else:
            xt_dict = self.q_sample(batch['x0_dict'], t)
        batch['xt_dict'] = xt_dict

        pred_dict = self.x0_pred_net(batch)
 
        affine_p = pred_dict['traj']
        losses, fape_dict = self.fape_loss(affine_p, affine_0, batch['gt_pos'][..., :3, :], batch['seq_mask'], batch['condition'])

        if pred_dict.__contains__('esm'):
            esm_loss = self.esm_loss(batch, pred_dict['esm'], batch['norm_esm_single'])
            losses.update(esm_loss)
        if pred_dict.__contains__('aatype'):
            esm_ce_loss = self.esm_ce_loss(batch, pred_dict['aatype'])
            losses.update(esm_ce_loss)
        if pred_dict.__contains__('distogram'):
            distogram_loss = self.distogram_loss(batch, pred_dict['distogram'])
            losses.update(distogram_loss)
        if pred_dict.__contains__('aux_distogram'):
            distogram_loss = self.distogram_aux_loss(batch, pred_dict['aux_distogram'])
            losses.update(distogram_loss)
        if self.global_config.loss_weight.diff_violation_loss:
            atom14_positions = fape_dict['coord'][-1]
            violation_loss = structural_violation_loss(batch, atom14_positions, self.global_config.violation_config)
            losses['violation_loss'] = violation_loss

        fape_dict['esm'] = pred_dict['esm']


        return fape_dict, losses


    @torch.no_grad()
    def sampling(
        self, 
        batch: dict, 
        pdb_prefix: str, 
        step_num: int, 
        mu_dict: dict=None, 
        init_noising_scale=1.0, 
        ddpm_fix=False, 
        rigid_fix=False, 
        term_num=0, 
        diff_noising_scale=1.0
        ):
        device = batch['aatype'].device
        batch_size, num_res = batch['aatype'].shape[:2]
        fape_condition = None
        rigid_fix_align_freq = 7

        if mu_dict is None:
            affine_tensor_nosie = torch.randn((1, num_res, 6), dtype=torch.float32).to(device) * init_noising_scale
            affine_tensor_t = affine_tensor_nosie * self.affine_tensor_scale.to(device)
            affine_t = affine6_to_affine7(affine_tensor_t)

            esm_t = torch.randn(
                (1, num_res, self.global_config.esm_num), dtype=torch.float32).to(device)
        else:
            affine_tensor_noise = torch.randn((1, num_res, 6), dtype=torch.float32).to(device) * init_noising_scale
            affine_tensor_t = affine7_to_affine6(mu_dict['affine']) + affine_tensor_noise * self.affine_tensor_scale.to(device)
            affine_t = affine6_to_affine7(affine_tensor_t)

            # import pdb; pdb.set_trace()
            if not ddpm_fix:
                affine_t = torch.where(batch['condition'][..., None]==1, batch['traj_affine'][..., 0, :], affine_t)

            esm_noise = torch.randn(
                (1, num_res, self.global_config.esm_num), dtype=torch.float32).to(device) * init_noising_scale
            esm_t = mu_dict['esm'] + esm_noise

        if ddpm_fix:
            fix_condition = batch['condition']
            batch['condition'] = torch.zeros_like(batch['condition'])

        xt_dict = {
            'affine': affine_t,
            'esm': esm_t
        }
        batch['xt_dict'] = xt_dict
        
        if not batch.__contains__('gt_backbone_frame'):
            batch['gt_pos'] = batch['traj_pos']
            affine_0 = r3.rigids_to_quataffine_m(
                r3.rigids_from_tensor_flat12(batch['traj_backbone_frame'])
                ).to_tensor()[..., 0, :]
        else:
            affine_0 = r3.rigids_to_quataffine_m(
                r3.rigids_from_tensor_flat12(batch['gt_backbone_frame'])
                ).to_tensor()[..., 0, :]
            
        reduced_chain_label = list(set(batch['merged_chain_label'][0].tolist()))

        t_scheme = list(range(self.T-1, -1, -step_num))
        return_pdbfiles = []
        if t_scheme[-1] != 0:
            t_scheme.append(0)
            
        # for t_idx, t in enumerate(t_scheme):
        for t_idx in range(len(t_scheme)):
            t = t_scheme[t_idx]
            t = torch.LongTensor([t] * batch_size).to(device)
            batch['t'] = t

            x0_dict = self.x0_pred_net(batch)
            x0_dict = {k: v[-1] if k == 'traj' else v for k, v in x0_dict.items()}
            # x0_dict['affine'] = x0_dict['traj']
            if not ddpm_fix:
                fape_condition = batch['condition']
            else:
                fape_condition = fix_condition

            # generate traj and logger
            affine_p = x0_dict['traj']
            losses, pred_x0_dict = self.fape_loss(
                affine_p[None], affine_0, 
                batch['gt_pos'][..., :3, :], batch['seq_mask'], 
                fape_condition)

            if t[0] == 3:
                for batch_idx in range(batch_size):
                    coords_list = []
                    for chain_label in reduced_chain_label:
                        coords_list.append(
                            add_atom_O(pred_x0_dict['coord'][0, batch_idx][batch['merged_chain_label'][0] == chain_label].detach().cpu().numpy()[..., :3, :]).reshape(-1, 3)
                        )

                    write_multichain_from_atoms(coords_list,
                        f'{pdb_prefix}_diff_term_{term_num}_scale_{diff_noising_scale}_batch_{batch_idx}.pdb', natom=4)
                    return_pdbfiles.append(f'{pdb_prefix}_diff_term_{term_num}_scale_{diff_noising_scale}_batch_{batch_idx}.pdb')
                    
            if ddpm_fix:
                if fix_condition is not None:
                    if rigid_fix:
                        x0_dict_affine = x0_dict['traj']
                        if t_idx % rigid_fix_align_freq == 0:
                            for batch_idx in range(batch_size):
                                # mobile part of coord_gt to part of coord_pred
                                rotransed_gt_pos = kabschalign.align(
                                    batch['gt_pos'][batch_idx][fix_condition[0] == 1][:, :3], 
                                    pred_x0_dict['coord'][0, batch_idx][fix_condition[0] == 1],
                                    cycles=1, verbose=False)
                                rotransed_affine_0 = coord_to_frame_affine(rotransed_gt_pos)['affine'][0]
                        # replace part of coord_pred with part of coord_gt
                        x0_dict_affine[batch_idx][fix_condition[0] == 1] = rotransed_affine_0
                        x0_dict['affine'] = x0_dict_affine
                    else:
                        x0_dict['affine'] = torch.where(fix_condition[..., None] == 1, affine_0, x0_dict['traj'])
                else:
                    x0_dict['affine'] = x0_dict['traj']
            else:
                x0_dict['affine'] = x0_dict['traj']

            if not ddpm_fix:
                x_t_1_dict = self.q_sample(x0_dict, t, mu_dict, batch['condition'], affine_0[..., None, :], noising_scale=diff_noising_scale)
            else:
                x_t_1_dict = self.q_sample(x0_dict, t, mu_dict, noising_scale=diff_noising_scale)
            batch['xt_dict'] = x_t_1_dict
            
        # logger.info(f'term: {term_num}; generated')
        # return x0_dict
        batch['last_pdbfiles'] = return_pdbfiles
        return add_c_beta_from_crd(pred_x0_dict['coord'][0])
    
    
    def fape_loss(self, affine_p, affine_0, coord_0, mask, cond=None):
        quat_0 = affine_0[..., :4]
        trans_0 = affine_0[..., 4:]
        rot_0 = quat_to_rot(quat_0)

        batch_size, num_res = affine_0.shape[:2]

        rot_list, trans_list, coord_list = [], [], []
        num_ouputs = affine_p.shape[0]
        loss_unclamp, loss_clamp = [], []
        for i in range(num_ouputs):
            quat = affine_p[i, ..., :4]
            trans = affine_p[i, ..., 4:]
            rot = quat_to_rot(quat)
            coord = backbone_frame_to_atom3_std(
                torch.reshape(rot, (-1, 3, 3)),
                torch.reshape(trans, (-1, 3)),
            )
            # import pdb; pdb.set_trace()
            coord = torch.reshape(coord, (batch_size, num_res, 3, 3))
            coord_list.append(coord)
            rot_list.append(rot),
            trans_list.append(trans)

            if cond is not None:
                mask_2d = mask[..., None] * mask[..., None, :]
                affine_p_ = affine_p[i]
                coord_p_ = coord
            else:
                mask_2d = 1 - (cond[..., None] * cond[..., None, :])
                mask_2d = mask_2d * (mask[..., None] * mask[..., None, :])   
                # import pdb; pdb.set_trace()
                affine_p_ = torch.where(cond[..., None] == 1, affine_0, affine_p[i])
                coord_p_ = torch.where(cond[..., None, None] == 1, coord_0[:, :, :3], coord)

            quat_p_ = affine_p_[..., :4]
            trans_p_ = affine_p_[..., 4:]
            rot_p_ = quat_to_rot(quat_p_)

            fape, fape_clamp = backbone_fape_loss(
                coord_p_, rot_p_, trans_p_,
                coord_0, rot_0, trans_0, mask,
                clamp_dist=self.global_config.fape.clamp_distance,
                length_scale=self.global_config.fape.loss_unit_distance,
                mask_2d=mask_2d
            )
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

        losses = {
            'fape_loss': loss,
            'clamp_fape_loss': loss_clamp[-1],
            'unclamp_fape_loss': loss_unclamp[-1],
            'last_loss': last_loss,
            'traj_loss': traj_loss,
        }
        coord_dict = {
            'coord': torch.stack(coord_list),
            'rot': torch.stack(rot_list),
            'trans': torch.stack(trans_list)
        }
        return losses, coord_dict


    def esm_loss(self, batch, pred_esm, true_esm):
        esm_loss_dict = {}
        esm_single_error = F.mse_loss(pred_esm, true_esm, reduction='none') # B, L, D
        esm_single_mask = (batch['seq_mask'] * batch['esm_single_mask'])[..., None].repeat(1,1,self.global_config.esm_num)
        esm_single_error = esm_single_error * esm_single_mask
        esm_loss_dict['esm_single_pred_loss'] = torch.sum(esm_single_error) / (torch.sum(esm_single_mask) + 1e-6)
        return esm_loss_dict


    def esm_ce_loss(self, batch, pred_aatype):
        esm_ce_loss_dict = {}
        batchsize, num_res = pred_aatype.shape[:2]
        esm_ce_loss = F.cross_entropy(pred_aatype.reshape(-1, 20), batch['aatype'].reshape(-1), reduction='none').reshape(batchsize, num_res)
        esm_ce_loss = esm_ce_loss * batch['seq_mask']
        esm_ce_loss_dict['esm_single_ce_loss'] = torch.sum(esm_ce_loss) / (torch.sum(batch['seq_mask']) + 1e-6)
        return esm_ce_loss_dict


    def distogram_loss(self, batch, pred_maps_descrete):
        batchsize, nres = pred_maps_descrete.shape[:2]
        distogram_list = []
        distogram_pred_config = self.config.diff_model.distogram_pred
        if distogram_pred_config.pred_all_dist:
            if distogram_pred_config.atom3_dist:
                dist_type_name = ['ca-ca', 'n-n', 'c-c', 'ca-n', 'ca-c', 'n-c']
            else:
                if distogram_pred_config.ca_dist:
                    dist_type_name = ['ca-ca']
                else:
                    dist_type_name = ['cb-cb']

            for dist_type_idx, dist_type in enumerate(dist_type_name):
                gt_map_descrete = get_descrete_dist(batch['gt_pos'], dist_type, distogram_pred_config.distogram_args)
                dim_start = (dist_type_idx) * distogram_pred_config.distogram_args[-1]
                dim_end = (dist_type_idx + 1) * distogram_pred_config.distogram_args[-1]
                pred_map = pred_maps_descrete[..., dim_start: dim_end]
                distogram_loss = F.cross_entropy(
                    pred_map.reshape(-1, distogram_pred_config.distogram_args[-1]), 
                    gt_map_descrete.reshape(-1), reduction='none'
                    ).reshape(batchsize, nres, nres)
                distogram_list.append(distogram_loss)
        
        else:
            descrete_pair, all_angle_masks = get_descrete_feature(
                batch['gt_pos'][..., :4, :], return_angle_mask=True, mask_base_ca=False)
            gt_descrete_pair = descrete_pair[..., 1:].long() # remove ca dist map

            for pair_idx in range(4):
                if pair_idx in [0, 1, 2]:
                    bin_num = distogram_pred_config.distogram_args[-1]
                    dim_start = (pair_idx) * distogram_pred_config.distogram_args[-1]
                    dim_end = (pair_idx + 1) * distogram_pred_config.distogram_args[-1]
                    pred_map = pred_maps_descrete[..., dim_start: dim_end]
                else:
                    bin_num = distogram_pred_config.distogram_args[-1]//2
                    pred_map = pred_maps_descrete[..., -bin_num:]
                distogram_loss = F.cross_entropy(
                    pred_map.reshape(-1, bin_num), 
                    gt_descrete_pair[..., pair_idx].reshape(-1), reduction='none'
                    ).reshape(batchsize, nres, nres)

                if pair_idx in [1, 2, 3]:
                    distogram_loss = distogram_loss * all_angle_masks
                distogram_list.append(distogram_loss)
        # import pdb; pdb.set_trace()
        distogram_loss = torch.stack(distogram_list).mean(0)
        distogram_loss = distogram_loss * batch['pair_mask']
        distogram_loss_reduce = torch.sum(distogram_loss) / (torch.sum(batch['pair_mask']) + 1e-6)
        return {"ditogram_classify_loss": distogram_loss_reduce}


    def distogram_aux_loss(self, batch, pred_maps_descrete):
        batchsize, nres = pred_maps_descrete.shape[:2]
        distogram_list = []
        distogram_pred_config = self.config.diff_model.distogram_pred
        aux_dist_type_name = ['n-n', 'c-c', 'ca-n', 'ca-c', 'n-c']

        for dist_type_idx, dist_type in enumerate(aux_dist_type_name):
            gt_map_descrete = get_descrete_dist(batch['gt_pos'], dist_type, distogram_pred_config.distogram_args)
            dim_start = (dist_type_idx) * distogram_pred_config.distogram_args[-1]
            dim_end = (dist_type_idx + 1) * distogram_pred_config.distogram_args[-1]
            pred_map = pred_maps_descrete[..., dim_start: dim_end]
            distogram_loss = F.cross_entropy(
                pred_map.reshape(-1, distogram_pred_config.distogram_args[-1]), 
                gt_map_descrete.reshape(-1), reduction='none'
                ).reshape(batchsize, nres, nres)
            distogram_list.append(distogram_loss)

        distogram_loss = torch.stack(distogram_list).mean(0)
        distogram_loss = distogram_loss * batch['pair_mask']
        distogram_loss_reduce = torch.sum(distogram_loss) / (torch.sum(batch['pair_mask']) + 1e-6)
        return {"ditogram_aux_classify_loss": distogram_loss_reduce}


    def affine_to_coord(self, affine):
        quat = affine[..., :4]
        trans = affine[..., 4:]
        rot = quat_to_rot(quat)
        coord = backbone_frame_to_atom3_std(
            torch.reshape(rot, (-1, 3, 3)),
            torch.reshape(trans, (-1, 3)),
        )
        return coord
    

def fasta_writer(fasta_f, fasta_dict):
    with open(fasta_f, 'w') as writer:
        for fasta_query, seq in fasta_dict.items():
            writer.write(f'>{fasta_query}\n{seq}\n')