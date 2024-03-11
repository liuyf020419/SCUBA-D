import os, sys
import json
import logging

import random
import math
import numpy as np
import torch
from torch.utils import data
import torch.nn.functional as F
from .dataset import BaseDataset

sys.path.append("protdiff/models")
from folding_af2.all_atom import atom37_to_frames
from folding_af2.common import residue_constants
from folding_af2.quat_affine import QuatAffine, quat_multiply, apply_rot_to_vec, quat_to_rot
from folding_af2.r3 import rigids_to_quataffine_m, rigids_from_tensor_flat12
from protein_utils import rigid
from protein_geom_utils import generate_pair_from_pos

sys.path.append("pdb_utils/data_parser")
from protein_coord_parser_new import PoteinCoordsParser

sys.path.append("pdb_utils")
from calc_dssp import get_feature_from_dssp, preprocess_dssp_df, ENCODESS32NUM


logger = logging.getLogger(__name__)


STD_RIGID_COORD = torch.FloatTensor(
    [[-0.525, 1.363, 0.000],
    [0.000, 0.000, 0.000],
    [1.526, -0.000, -0.000],
    [0.627, 1.062, 0.000]]
)


class ProtDiffParDataset(BaseDataset):
    def __init__(self, config, par_data_list, batch_size):
        super().__init__()
        self.data_list= par_data_list
        self.config = config.model
        self.config_data = config.data
        self.global_config = self.config.global_config

        self.protein_list = []
        self._epoch = 0
        self.max_len = self.global_config.max_len
        self.enlarge_gap = self.global_config.enlarge_gap
        self.batch_size = batch_size

        with open(par_data_list, 'r') as f:
            for line in f:
                par_file = line.strip()
                self.protein_list.append(par_file)

        logger.info(f'list size: {len(self.protein_list)}')

    def __len__(self):
        return len(self.protein_list)

    def data_sizes(self):
        return [l[1] for l in self.protein_list]
    
    def reset_data(self, epoch):
        self._epoch = epoch

    def __getitem__(self, item:int):
        par_file = self.protein_list[item]
        
        with open(par_file, 'r') as reader:
            par_dict = json.load(reader)
        
        global_config = par_dict['global_config']
        pdbfile = global_config['input_file']
        pdbname = os.path.basename(global_config['output_prefix'])

        # data_dict = self.make_from_pdb_file(pdbfile, chain, datatype=datatype)
        data_dict = self.make_from_par_dict_new(pdbfile, par_dict, mode=global_config['mode'])
        resrange = (-self.config.refine_net.pair_encoder.pair_res_rel, self.config.refine_net.pair_encoder.pair_res_rel)
        resmask_num = self.config.refine_net.pair_encoder.pair_res_rel + 1
        chainrange = (-self.config.refine_net.pair_encoder.pair_chain_rel, self.config.refine_net.pair_encoder.pair_chain_rel)
        chainmask_num = self.config.refine_net.pair_encoder.pair_chain_rel + 1
        self.get_position_embedding(data_dict, relpdb_residx=data_dict['single_res_rel'], 
                                    enlarge_gap=self.enlarge_gap, resrange=resrange,
                                    resmask_num=resmask_num, chainrange=chainrange, chainmask_num=chainmask_num)
        mode_config = par_dict[global_config['mode']]
        chain_names = ''.join(list(mode_config.keys()))
        data_dict['pdbname'] = pdbname + '_' + chain_names
        data_dict['output_prefix'] = global_config['output_prefix']
        data_dict['cfg_file'] = par_file

        return data_dict


    
    def make_from_par_dict_new(self, poteinfile, par_dict, mode, chain_gap=100):
        data_dict = {}
        assert mode in ['gen_from_noise', 'loopsampling', 'refine_prior']
        mode_config = par_dict[mode]
        if mode in ['gen_from_noise']:
            sstype = mode_config['target_sstype']
            if mode_config.__contains__('mode_config'):
                ss_noise_scale = mode_config['ss_noise_scale']
            else:
                ss_noise_scale = [5.0, 8.0]
            if sstype is not None:
                encoded_sstype = torch.tensor([ENCODESS32NUM[ss] for ss in sstype]).long()
            mode_config = mode_config['chain_config']
        elif mode in ['refine_prior']:
            sstype = get_sstype_from_coords(poteinfile)
            mode_config = mode_config['chain_config']

        if len(poteinfile) > 0:
            assert os.path.isfile(poteinfile)
            chain_names = list(mode_config.keys())
            PDBparser = PoteinCoordsParser(
                poteinfile, chain=chain_names,
                pseudo_gly=False
                )
            input_pdb_pos = torch.from_numpy(PDBparser.chain_main_crd_array).float()
            global_pos_center = torch.cat([input_pdb_pos[:, 1]]).mean(0)
            # input_pdb_pos = input_pdb_pos - pos_center
            pdb_raw_idx = PDBparser.pdbresID
            # pdb_rawidx_to_posidx_dict = PDBparser.pdbresID2absID
            input_merged_chain_label = np.concatenate([
                [chain_idx] * chain_len for chain_idx, chain_len in enumerate(PDBparser.multichain_length_dict.values())
            ])
        else:
            assert mode in ['gen_from_noise']
            chain_names = list(mode_config.keys())
            global_pos_center = torch.zeros(3,)
            
        
        batch_all_traj_pos, batch_all_traj_frame = [], []
        
        for b_idx in range(self.batch_size):
            all_traj_pos, all_traj_res_rel, fix_condition, merged_chain_label = [], [], [], []

            for chain_idx, chain_name in enumerate(chain_names):
                chain_traj_pos = []
                chain_fix_condition = []
                chain_par = mode_config[chain_name]
                if len(poteinfile) > 0:
                    chain_input_pos = torch.from_numpy(PDBparser.get_main_crd_array(chain_name)).float()
                    chain_res_num = chain_input_pos.shape[0]
                    chain_rawidx_to_posidx_dict = PDBparser.get_pdbresID2absID(chain_name)
                    chain_input_pos = add_pseudo_c_beta_from_gly(chain_input_pos)
                    chain_input_pos = chain_input_pos - global_pos_center # + relative_chain_pos_center
                    
                else:
                    assert (not '_' in chain_par)
                
                
                if mode == 'gen_from_noise':
                    if not ( ("UNFIX" in chain_par) or ("FIX" in chain_par) ):
                        chain_par_list = chain_par.split(';')
                        for motif_p in chain_par_list:
                            if '_' in motif_p:
                                start_rawidx, end_rawidx = motif_p.split('_')
                                if ']' in end_rawidx:
                                    end_rawidx = end_rawidx[:-1]
                                    add_end = True
                                else:
                                    end_rawidx = end_rawidx
                                    add_end = False
  
                                start_posidx = chain_rawidx_to_posidx_dict[int(start_rawidx)]
                                if add_end:
                                    end_posidx = chain_rawidx_to_posidx_dict[int(end_rawidx)] + 1
                                else:
                                    end_posidx = chain_rawidx_to_posidx_dict[int(end_rawidx)]
                                    
                                input_pdb_motif_pos = chain_input_pos[start_posidx: end_posidx]
                                
                                chain_traj_pos.append(add_pseudo_c_beta_from_gly(input_pdb_motif_pos))
                                chain_fix_condition.extend(np.ones(input_pdb_motif_pos.shape[0]))
                                merged_chain_label.extend(np.ones(input_pdb_motif_pos.shape[0]) * chain_idx)
                            else:
                                motif_len = int(motif_p)
                                noising_quat = rigid.rand_quat([1, motif_len])
                                noising_coord = torch.randn(1, motif_len, 3) * self.config_data.white_noise.white_noise_scale
                                noising_affine = torch.cat([noising_quat, noising_coord], -1)
                                noising_pos = rigid.affine_to_pos(noising_affine.reshape(-1, 7)).reshape(motif_len, -1, 3)
                                if sstype is not None:
                                    encoded_partial_sstype = encoded_sstype[len(chain_traj_pos): len(chain_traj_pos)+motif_len]

                                    traj_coords, traj_flat12s = build_rdsketch_from_sstype(
                                        encoded_partial_sstype[:, None], ss_noise_scale, 
                                        self.config_data.white_noise.white_noise_scale, 
                                        ss_mask_p_range=[0.0, 0.0])
                                    chain_traj_pos.append(traj_coords)
                                else:
                                    chain_traj_pos.append(add_pseudo_c_beta_from_gly(noising_pos))
                                chain_fix_condition.extend(np.zeros(motif_len))
                                merged_chain_label.extend(np.ones(motif_len) * chain_idx)
                        chain_traj_pos = torch.cat(chain_traj_pos)
                    else:
                        if "UNFIX" in chain_par:
                            raise KeyError(f'UNFIX par in gen_from_noise not available now')
                        elif "FIX" in chain_par:
                            chain_traj_pos = chain_input_pos
                            chain_fix_condition.extend(np.ones(chain_traj_pos.shape[0]))
                            merged_chain_label.extend(np.ones(chain_traj_pos.shape[0]) * chain_idx)
                        else:
                            raise KeyError(f'chain par: {chain_par} unknown')
                            
                    chain_traj_len = chain_traj_pos.shape[0]
                    chain_traj_res_rel = np.arange(chain_traj_len)
                    all_traj_pos.append(chain_traj_pos)
                    
                    if not ( ("UNFIX" in chain_par) or ("FIX" in chain_par) ):
                        if chain_idx != 0:
                            all_traj_res_rel.extend(chain_traj_res_rel + all_traj_res_rel[-1] + chain_gap)
                        else:
                            all_traj_res_rel.extend(chain_traj_res_rel)
                    else:
                        assert len(poteinfile) > 0
                        if chain_idx != 0:
                            all_traj_res_rel.extend(np.array(pdb_raw_idx[chain_name]) + all_traj_res_rel[-1] + chain_gap)
                        else:
                            all_traj_res_rel.extend(np.array(pdb_raw_idx[chain_name]))
                        
                    fix_condition.extend(chain_fix_condition)
                    
                elif mode == 'loopsampling':
                    assert len(poteinfile) > 0
                    last_end_pos = 0
                    last_start_pos = 0
                    if not ( ("UNFIX" in chain_par) or ("FIX" in chain_par) ):
                        chain_par_list = chain_par.split(';')
                        for motif_p in chain_par_list:
                            if ',' in motif_p:
                                loop_rawidx_range, new_loop_len = motif_p.split(',')
                                loop_start_rawidx, loop_end_rawidx = loop_rawidx_range.split('_')
                                if ']' in loop_end_rawidx:
                                    loop_end_rawidx = loop_end_rawidx[:-1]
                                    add_end = True
                                else:
                                    loop_end_rawidx = loop_end_rawidx
                                    add_end = False
                                new_loop_len = int(new_loop_len)
                            else:
                                loop_rawidx_range = motif_p
                                loop_start_rawidx, loop_end_rawidx = loop_rawidx_range.split('_')
                                
                                if ']' in loop_end_rawidx:
                                    loop_end_rawidx = loop_end_rawidx[:-1]
                                    add_end = True
                                else:
                                    loop_end_rawidx = loop_end_rawidx
                                    add_end = False
                                new_loop_len = int(loop_end_rawidx) - int(loop_start_rawidx)

                            loop_start_posidx = chain_rawidx_to_posidx_dict[int(loop_start_rawidx)]
                            if add_end:
                                loop_end_posidx = chain_rawidx_to_posidx_dict[int(loop_end_rawidx)] + 1
                            else:
                                loop_end_posidx = chain_rawidx_to_posidx_dict[int(loop_end_rawidx)]
                                
                            ## before loop motif
                            assert loop_start_posidx >= last_start_pos
                            assert last_end_pos <= loop_start_posidx
                            before_loop_motif_pos = chain_input_pos[last_end_pos: loop_start_posidx]

                            chain_traj_pos.append(add_pseudo_c_beta_from_gly(before_loop_motif_pos))
                            chain_fix_condition.extend(np.ones(before_loop_motif_pos.shape[0]))
                            merged_chain_label.extend(np.ones(before_loop_motif_pos.shape[0]) * chain_idx)
                            
                            ## new loop pos
                            noising_quat = rigid.rand_quat([1, new_loop_len])
                            noising_coord = torch.randn(1, new_loop_len, 3) * self.config_data.white_noise.white_noise_scale
                            noising_affine = torch.cat([noising_quat, noising_coord], -1)
                            noising_pos = rigid.affine_to_pos(noising_affine.reshape(-1, 7)).reshape(new_loop_len, -1, 3)
                            chain_traj_pos.append(add_pseudo_c_beta_from_gly(noising_pos))
                            chain_fix_condition.extend(np.zeros(new_loop_len))
                            merged_chain_label.extend(np.ones(new_loop_len) * chain_idx)
                            
                            last_end_pos = loop_end_posidx
                            last_start_pos = loop_start_posidx
                        
                        # last motif
                        if last_end_pos is not None:
                            last_motif_pos = chain_input_pos[last_end_pos:]
                            if len(last_motif_pos) > 0:
                                chain_traj_pos.append(add_pseudo_c_beta_from_gly(last_motif_pos))
                                chain_fix_condition.extend(np.ones(len(last_motif_pos)))
                                merged_chain_label.extend(np.ones(len(last_motif_pos)) * chain_idx)
                            chain_traj_pos = torch.cat(chain_traj_pos)
                        
                    else:
                        if "UNFIX" in chain_par:
                            raise KeyError(f'UNFIX par in loopsampling not available now')
                        elif "FIX" in chain_par:
                            chain_traj_pos = chain_input_pos
                            chain_fix_condition.extend(np.ones(chain_traj_pos.shape[0]))
                            merged_chain_label.extend(np.ones(chain_traj_pos.shape[0]) * chain_idx)
                        else:
                            raise KeyError(f'chain par: {chain_par} unknown')
                        
                    chain_traj_len = chain_traj_pos.shape[0]
                    chain_traj_res_rel = np.arange(chain_traj_len)
                    all_traj_pos.append(chain_traj_pos)
                    
                    if chain_idx != 0:
                        all_traj_res_rel.extend(chain_traj_res_rel + all_traj_res_rel[-1] + chain_gap)
                    else:
                        all_traj_res_rel.extend(chain_traj_res_rel)
                    fix_condition.extend(chain_fix_condition)
                        
                elif mode == 'refine_prior':
                    assert len(poteinfile) > 0
                    merged_chain_label = input_merged_chain_label
                    if chain_idx != 0:
                        all_traj_res_rel.extend(np.array(pdb_raw_idx[chain_name]) + all_traj_res_rel[-1] + chain_gap)
                    else:
                        all_traj_res_rel.extend(np.array(pdb_raw_idx[chain_name]))
                    
                    chain_fix_condition = np.zeros(chain_input_pos.shape[0])
                    if not ( ("UNFIX" in chain_par) or ("FIX" in chain_par) ):
                        chain_par_list = chain_par.split(';')
                        if len(chain_par_list) > 0:
                            for motif_p in chain_par_list:
                                if '_' in motif_p: 
                                    start_rawidx, end_rawidx = motif_p.split('_')
                                    if ']' in end_rawidx:
                                        end_rawidx = end_rawidx[:-1]
                                        add_end = True
                                    else:
                                        end_rawidx = end_rawidx
                                        add_end = False

                                    start_posidx = chain_rawidx_to_posidx_dict[int(start_rawidx)]
                                    if add_end:
                                        end_posidx = chain_rawidx_to_posidx_dict[int(end_rawidx)] + 1
                                    else:
                                        end_posidx = chain_rawidx_to_posidx_dict[int(end_rawidx)]
                                    chain_fix_condition[start_posidx: end_posidx] = 1
                                else:
                                    raise KeyError(f'chain par: {chain_par} unknown')
                    else:
                        if "UNFIX" in chain_par:
                            chain_fix_condition = np.zeros(chain_input_pos.shape[0])
                        elif "FIX" in chain_par:
                            chain_fix_condition = np.ones(chain_input_pos.shape[0])
                        else:
                            raise KeyError(f'chain par: {chain_par} unknown')
                        
                    raw_chain_input = add_pseudo_c_beta_from_gly(chain_input_pos)
                    white_noise_scale = self.config_data.white_noise.white_noise_scale
                    # # add ss translation noise
                    noising_quat = rigid.rand_quat([1, chain_res_num])
                    noising_coord = torch.randn(1, chain_res_num, 3) * white_noise_scale
                    noising_affine = torch.cat([noising_quat, noising_coord], -1)
                    noising_pos = add_pseudo_c_beta_from_gly(
                        rigid.affine_to_pos(noising_affine.reshape(-1, 7)).reshape(chain_res_num, -1, 3))

                    traj_pos = torch.where(sstype[..., None, None] == 1, noising_pos, chain_input_pos)
                    traj_pos = torch.where(torch.from_numpy(chain_fix_condition)[..., None, None] == 1, raw_chain_input, traj_pos)
                    
                    all_traj_pos.append(traj_pos)
                    fix_condition.extend(chain_fix_condition)
            
            all_traj_pos = torch.cat(all_traj_pos)
            pos_center = torch.cat([all_traj_pos[:, 1]]).mean(0)
            all_traj_pos = all_traj_pos - pos_center
            traj_frame = get_quataffine(all_traj_pos)
            
            batch_all_traj_pos.append(all_traj_pos)
            batch_all_traj_frame.append(traj_frame)
        
        batch_all_traj_pos = torch.stack(batch_all_traj_pos)
        batch_all_traj_frame = torch.stack(batch_all_traj_frame)

        logger.info(f'fixed residue idx in new pdb file {torch.where(torch.tensor(fix_condition))[0].numpy().tolist()}')
        
        aatype = torch.LongTensor([residue_constants.restype_order_with_x['A'] for _ in range(len(all_traj_res_rel))])
        data_dict['traj_pos'] = batch_all_traj_pos.float()
        data_dict['traj_backbone_frame'] = batch_all_traj_frame.float()
        data_dict['aatype'] = aatype
        data_dict['len'] = torch.LongTensor([len(aatype)])
        data_dict['single_res_rel'] = np.array(all_traj_res_rel)
        data_dict['fix_condition'] = fix_condition
        data_dict['noising_mode'] = 4
        data_dict['merged_chain_label'] = torch.LongTensor(merged_chain_label)

        return data_dict
        

    def get_position_embedding(self, data_dict, relpdb_residx, resrange=(-32, 32), resmask_num=33, 
                                    chainrange=(-4, 4), chainmask_num=5, enlarge_gap=True, gap_size=100):

        split_idx = np.arange(len(relpdb_residx))[np.append(np.diff(relpdb_residx) != 1, False)] + 1
        # last chain
        chain_num = len(split_idx) + 1
        chain_lens = np.diff(np.append(np.concatenate([[0], split_idx]), len(relpdb_residx) ))

        if enlarge_gap:
            res_rel_idx = []
            for idx, chain_len in enumerate(chain_lens):
                if idx != 0:
                    res_rel_idx.extend(np.arange(chain_len) + res_rel_idx[-1] + gap_size)
                else:
                    res_rel_idx.extend(np.arange(chain_len))

            data_dict["single_res_rel"] = torch.LongTensor(res_rel_idx)

        else:
            single_part_res_rel_idx = np.concatenate([np.arange(chain_len) for chain_len in chain_lens])
            single_all_chain_rel_idx = np.concatenate([np.ones(chain_len[1], dtype=np.int32) * chain_len[0] \
                                                        for chain_len in enumerate(chain_lens)])

            single_all_res_rel_idx = relpdb_residx - relpdb_residx[0]
            data_dict["single_all_res_rel"] = torch.from_numpy(single_all_res_rel_idx)
            data_dict["single_part_res_rel"] = torch.from_numpy(single_part_res_rel_idx)
            data_dict["single_all_chain_rel"] = torch.from_numpy(single_all_chain_rel_idx)


        pair_res_rel_idx = relpdb_residx[:, None] - relpdb_residx

        unclip_single_chain_rel_idx = np.repeat(np.arange(chain_num), chain_lens)
        pair_chain_rel_idx = unclip_single_chain_rel_idx[:, None] - unclip_single_chain_rel_idx
        
        pair_res_rel_idx = np.where(np.any(np.stack([pair_res_rel_idx > resrange[1], 
                                pair_res_rel_idx < resrange[0]]), 0), resmask_num, pair_res_rel_idx)

        pair_chain_rel_idx = np.where(np.any(np.stack([pair_chain_rel_idx > chainrange[1], 
                                pair_chain_rel_idx < chainrange[0]]), 0), chainmask_num, pair_chain_rel_idx)

        data_dict["pair_res_rel"] = torch.from_numpy(pair_res_rel_idx.astype(np.int64)) - resrange[0]
        data_dict["pair_chain_rel"] = torch.from_numpy(pair_chain_rel_idx.astype(np.int64)) - chainrange[0]


def get_quataffine(pos):
    assert len(pos.shape)
    nres, natoms, _ = pos.shape
    assert natoms == 5
    alanine_idx = residue_constants.restype_order_with_x["A"]
    aatype = torch.LongTensor([alanine_idx] * nres)
    all_atom_positions = F.pad(pos, (0,0,0,37-5), "constant", 0)
    all_atom_mask = torch.ones(nres, 37)
    frame_dict = atom37_to_frames(aatype, all_atom_positions, all_atom_mask)

    return frame_dict['rigidgroups_gt_frames']


def add_pseudo_c_beta_from_gly(pos):
    vec_ca = pos[:, 1]
    vec_n = pos[:, 0]
    vec_c = pos[:, 2]
    vec_o = pos[:, 3]
    b = vec_ca - vec_n
    c = vec_c - vec_ca
    a = torch.cross(b, c)
    vec_cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + vec_ca
    return torch.stack([vec_n, vec_ca, vec_c, vec_cb, vec_o]).permute(1,0,2)


def get_sstype_from_coords(pdbfile):
    file_type = pdbfile.split('.')[-1]
    df = get_feature_from_dssp(pdbfile, file_type=file_type, return_type='df', add_ca_coord=True)
    df = preprocess_dssp_df(df, add_ss_idx=True)
    encoded_sstype = torch.from_numpy(df.loc[:, 'SS3enc'].to_numpy())
    return encoded_sstype


def build_rdsketch_from_sstype(sstype, ss_noise_scale, loop_noise_scale, ss_mask_p_range):
    ss3type = sstype[:, 0]
    ss_start_indexs = (torch.where((ss3type[1:] - ss3type[:-1]) != 0)[0] + 1).long()
    ss_start_indexs = torch.cat([torch.LongTensor([0]), ss_start_indexs])
    ss_end_indexs = torch.cat([ss_start_indexs[1:]-1, torch.LongTensor([len(ss3type)])])
    ss_lens = ss_start_indexs[1:] - ss_start_indexs[:-1]
    ss_lens = torch.cat([ss_lens, (len(ss3type) - ss_start_indexs[-1]).unsqueeze(0)])
    start_sstypes = torch.index_select(ss3type, 0, ss_start_indexs)

    if isinstance(ss_noise_scale, list):
        ss_noise_scale = np.random.uniform(ss_noise_scale[0], ss_noise_scale[1], 1)[0]
    assert isinstance(ss_mask_p_range, list)
    ss_mask_p = np.random.uniform(ss_mask_p_range[0], ss_mask_p_range[1], 1)[0]

    traj_coords = []
    for ss_idx, ss in enumerate(start_sstypes):
        ss_len = ss_lens[ss_idx]

        if ((ss_len > 2) and (ss != 1)):
            if np.random.rand(1)[0] > ss_mask_p:
                qT = rigid.rand_quat(torch.rand(1).shape[:-1])

                new_traj_rot = rigid.quat_to_rot(qT)
                updated_traj_trans = torch.randn(3) * ss_noise_scale
                
                sketch_ss_pos = add_pseudo_c_beta_from_gly(
                    torch.from_numpy(
                        gen_peptides_zero_mass_center_peptides(ss_len, SS3_num_to_name[ss.item()])).float()
                    )
                traj_ss_pos = update_rigid_pos_new(sketch_ss_pos, updated_traj_trans, new_traj_rot)
                traj_coords.append(traj_ss_pos)
            else:
                noising_quat = rigid.rand_quat([1, ss_len])
                noising_coord = torch.randn(1, ss_len, 3) * loop_noise_scale
                noising_affine = torch.cat([noising_quat, noising_coord], -1)
                noising_pos = rigid.affine_to_pos(noising_affine.reshape(-1, 7)).reshape(ss_len, -1, 3)
                traj_coords.append(add_pseudo_c_beta_from_gly(noising_pos))

        else:
            noising_quat = rigid.rand_quat([1, ss_len])
            noising_coord = torch.randn(1, ss_len, 3) * loop_noise_scale
            noising_affine = torch.cat([noising_quat, noising_coord], -1)
            noising_pos = rigid.affine_to_pos(noising_affine.reshape(-1, 7)).reshape(ss_len, -1, 3)
            traj_coords.append(add_pseudo_c_beta_from_gly(noising_pos))

    traj_coords = torch.cat(traj_coords)
    traj_flat12s = get_quataffine(traj_coords)

    return traj_coords, traj_flat12s


def update_rigid_pos_new(pos, translation, rotation):
    assert len(pos.shape) == 3
    L, N, _ = pos.shape
    ca_mass_pos = pos[:, 1].mean(0)
    new_ca_mass_pos = ca_mass_pos + translation
    roted_pos = torch.matmul(pos.reshape(-1, 3) - ca_mass_pos, rotation)
    updated_pos = roted_pos.reshape(L, N, -1)
    updated_pos = updated_pos + new_ca_mass_pos[None, None]
    
    return updated_pos


def pad_dim(data, dim, max_len):
    """ dim int or [int, int]
    """
    if (isinstance(dim, int) or (isinstance(dim, list) and len(dim) == 0)):
        dim = dim
        if isinstance(dim, int):
            dims = [dim]
        else:
            dims = dim
    else:
        dims = dim
        dim = dim[0]
        
    def convert_pad_shape(pad_shape):
        l = pad_shape[::-1]
        pad_shape = [item for sublist in l for item in sublist]
        return pad_shape

    shape = [d for d in data.shape]
    assert(shape[dim] <= max_len)

    if shape[dim] == max_len:
        return data

    pad_len = max_len - shape[dim]

    pad_shape = []
    for d in dims:
        tmp_pad_shape = [[0, 0]] * d + [[0, pad_len]] + [[0, 0]] * (len(shape) - d -1)
        pad_shape.append(convert_pad_shape(tmp_pad_shape))

    data_pad = F.pad(data, np.sum(pad_shape, 0).tolist(), mode='constant', value=0)
    return data_pad


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return {}
    cat_data = {}
    max_len = max([b['aatype'].shape[0] for b in batch])


    for name in batch[0].keys():
        if name in ['loss_mask', 'len']:
            cat_data[name] = torch.cat([b[name] for b in batch], dim=0)
        elif name in ['pair_res_rel', 'pair_chain_rel']:
            data = torch.cat([pad_dim(b[name], [0, 1], max_len)[None] for b in batch], dim=0)
            cat_data[name] = data
        elif name in ['pdbname']:
            data = [b[name] for b in batch]
            cat_data[name] = data
        else:
            data = torch.cat([pad_dim(b[name], 0, max_len)[None] for b in batch], dim=0)
            cat_data[name] = data

    return cat_data


def data_is_nan(data):
    for k, v in data.items():
        if torch.isnan(v.abs().sum()):
            return True
    return False


def to_tensor(arr):
    if isinstance(arr, np.ndarray):
        if arr.dtype in [np.int64, np.int32]:
            return torch.LongTensor(arr)
        elif arr.dtype in [np.float64, np.float32]:
            return torch.FloatTensor(arr)
        elif arr.dtype == np.bool:
            return torch.BoolTensor(arr)
        else:
            return arr
    else:
        return arr


def moveaxis(data, source, destination):
  n_dims = len(data.shape)
  dims = [i for i in range(n_dims)]
  if source < 0:
    source += n_dims
  if destination < 0:
    destination += n_dims

  if source < destination:
    dims.pop(source)
    dims.insert(destination, source)
  else:
    dims.pop(source)
    dims.insert(destination, source)

  return data.permute(*dims)

