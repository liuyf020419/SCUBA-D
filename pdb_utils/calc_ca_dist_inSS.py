import os

import numpy as np
import torch

from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP
import pandas as pd

dssp_bin = '/train14/superbrain/lhchen/protein/tools/dssp-3.1.4/mkdssp'

import sys
sys.path.append('protdiff/models')
from protein_geom_utils import add_c_beta_from_crd, generate_pair_from_pos


ENCODESS32NUM = {
    "H": 0,
    "L": 1,
    "E": 2
}

ATOMS_TYPE=['N', 'CA', 'C']

def get_feature_from_dssp(pdbfile, return_type = "list", add_coord=False, return_coords=False):
    assert os.path.basename(pdbfile).split('.')[1] == 'pdb'
    assert return_type in ['list', 'dict', 'df']

    p = PDBParser()
    structure = p.get_structure(os.path.basename(pdbfile).split('.')[0], pdbfile)
    model = structure[0]
    dssp = DSSP(model, pdbfile, dssp=dssp_bin, acc_array="Sander", file_type="pdb")

    if add_coord:
        all_coords = []
        for atom_type in ATOMS_TYPE:
            ca_coords = np.stack([atom_inf.get_coord() for atom_inf in list(model.get_atoms()) if atom_inf.name == atom_type])
            all_coords.append(ca_coords)
        all_coords = np.stack(all_coords, -2)

    if return_type == "list":
        return dssp.property_list
    elif return_type == "dict":
        return dssp.property_dict
    elif return_type == "df":
        df = pd.DataFrame(
            np.array(dssp.property_list), 
            columns=('dssp index', 'amino acid', 'secondary structure', 'relative ASA', 'phi', 'psi',
                    'NH_O_1_relidx', 'NH_O_1_energy', 'O_NH_1_relidx', 'O_NH_1_energy',
                    'NH_O_2_relidx', 'NH_O_2_energy', 'O_NH_2_relidx', 'O_NH_2_energy'))
        if add_coord:
            for atom_idx, atom_type in enumerate(ATOMS_TYPE):
                df[f'X_{atom_type}'] = all_coords[:, atom_idx, 0]
                df[f'Y_{atom_type}'] = all_coords[:, atom_idx, 1]
                df[f'Z_{atom_type}'] = all_coords[:, atom_idx, 2]
            if return_coords:
                return df, all_coords

        return df


# def get_feature_from_dssp(pdbfile, return_type = "list", add_ca_coord=False):
#     assert os.path.basename(pdbfile).split('.')[1] == 'pdb'
#     assert return_type in ['list', 'dict', 'df']

#     p = PDBParser()
#     structure = p.get_structure(os.path.basename(pdbfile).split('.')[0], pdbfile)
#     model = structure[0]
#     dssp = DSSP(model, pdbfile, dssp=dssp_bin, acc_array="Sander", file_type="pdb")

#     if add_ca_coord:
#         ca_coords = np.stack([atom_inf.get_coord() for atom_inf in list(model.get_atoms()) if atom_inf.name == 'CA'])

#     if return_type == "list":
#         return dssp.property_list
#     elif return_type == "dict":
#         return dssp.property_dict
#     elif return_type == "df":
#         df = pd.DataFrame(
#             np.array(dssp.property_list), 
#             columns=('dssp index', 'amino acid', 'secondary structure', 'relative ASA', 'phi', 'psi',
#                     'NH_O_1_relidx', 'NH_O_1_energy', 'O_NH_1_relidx', 'O_NH_1_energy',
#                     'NH_O_2_relidx', 'NH_O_2_energy', 'O_NH_2_relidx', 'O_NH_2_energy'))
#         df['X_CA'] = ca_coords[:, 0]
#         df['Y_CA'] = ca_coords[:, 1]
#         df['Z_CA'] = ca_coords[:, 2]

#         return df


def preprocess_dssp_df(df: pd.DataFrame, add_ss_idx=False):
    rsa_series = df['relative ASA']
    ss8_series = df['secondary structure']
    ss3_series = ss8_series.copy()

    ss3_series.loc[(ss8_series == 'T')|(ss8_series == 'S')|(ss8_series == '-')] = "L"
    ss3_series.loc[(ss8_series == 'H') | (ss8_series == 'G') | (ss8_series == 'I')] = "H"
    ss3_series.loc[(ss8_series == 'B') | (ss8_series == 'E')] = "E"
    
    ss3encode_series = ss3_series.copy()
    ss3encode_series.replace(ENCODESS32NUM, inplace=True)

    new_dict ={
        "resid": df["dssp index"], 
        "aatype": df["amino acid"], 
        "SS3": ss3_series, 
        "SS3enc": ss3encode_series,
        "SS8": ss8_series, 
        "RSA": rsa_series
        }

    if add_ss_idx:
        ss3_str = ''.join(ss3_series.to_list())
        simp_ss3 = calc_simp_SS(ss3_series.to_list())
        ss_len = np.diff(simp_ss3[:, 0].astype(np.int16))
        ss_len = np.append(ss_len, len(ss3_str)-int(simp_ss3[-1, 0]))

        ss_idx = np.concatenate([np.array([ss_absidx] * ss_len[ss_absidx]) for ss_absidx in range(len(ss_len))])
        new_dict['SS_idx'] = ss_idx

    for atom_type in ATOMS_TYPE:
        if f'X_{atom_type}' in df.columns:
            new_dict[f'X_{atom_type}'] = df[f'X_{atom_type}']
            new_dict[f'Y_{atom_type}'] = df[f'Y_{atom_type}']
            new_dict[f'Z_{atom_type}'] = df[f'Z_{atom_type}']

    newdf = pd.DataFrame(
        new_dict
        ).set_index("resid")

    return newdf


def calc_simp_SS(SS) -> np.ndarray:
    simp_SS = []
    last_ss = None
    for id, sstate in enumerate(SS):
        if id == 0:
            last_ss = sstate
            simp_SS.append([id, sstate])
        else:
            if sstate == last_ss:
                continue
            else:
                last_ss = sstate
                simp_SS.append([id, sstate])
    return np.asarray(simp_SS)


def find_nearest_CAdist_inSS(ca_pos, sstype, sstype_idx, clamp_dist=20):
    ca_dist_map = torch.sqrt(torch.sum(torch.square(ca_pos[:, None] - ca_pos[None]), -1) + 1e-10)

    HH_dist_list = []
    HE_dist_list = []

    for ss_idx, ss in enumerate(sstype):
        if (ss == 0):
            cur_sstype_idx = sstype_idx[ss_idx]
            cur_ca_dist = ca_dist_map[ss_idx]
            neighbor_sortedidx = torch.argsort(ca_dist_map[ss_idx])
            sameSS_res_mask = sstype_idx != cur_sstype_idx

            ## HH stat
            nearest_res_mask_HH = torch.all(torch.stack([sameSS_res_mask, (sstype == 0)]), 0)
            nearest_res_idx_HH = neighbor_sortedidx[nearest_res_mask_HH[neighbor_sortedidx]][0]
            nearset_ca_dist_HH = cur_ca_dist[nearest_res_idx_HH]
            if nearset_ca_dist_HH <= clamp_dist:
                HH_dist_list.append(nearset_ca_dist_HH)

            ## HE
            if torch.any(sstype == 2):
                nearest_res_mask_HE = torch.all(torch.stack([sameSS_res_mask, (sstype == 2)]), 0)
                nearest_res_idx_HE = neighbor_sortedidx[nearest_res_mask_HE[neighbor_sortedidx]][0]
                nearset_ca_dist_HE = cur_ca_dist[nearest_res_idx_HE]
                if nearset_ca_dist_HE <= clamp_dist:
                    HE_dist_list.append(nearset_ca_dist_HE)

    return HH_dist_list, HE_dist_list
        

def find_nearest_CAfeature_inSS(coords, sstype, sstype_idx, clamp_dist=20):
    ca_pos = coords[:, 1]
    ca_dist_map = torch.sqrt(torch.sum(torch.square(ca_pos[:, None] - ca_pos[None]), -1) + 1e-10)
    coords_with_beta = add_c_beta_from_crd(coords[None]) # 1, L, 4, 3
    pair_feature = generate_pair_from_pos(coords_with_beta)[0] # L, L, 4

    HH_ca_dist_list = []
    HE_ca_dist_list = []
    HH_cb_dist_list = []
    HE_cb_dist_list = []
    HH_omega_list = []
    HE_omega_list = []
    HH_phi_list = []
    HE_phi_list = []
    HH_theta_list = []
    HE_theta_list = []

    for ss_idx, ss in enumerate(sstype):
        if (ss == 0):
            cur_sstype_idx = sstype_idx[ss_idx]

            cur_ca_dist = ca_dist_map[ss_idx]
            cur_cb_dist = pair_feature[..., 0][ss_idx]
            cur_omega = pair_feature[..., 1][ss_idx]
            cur_phi =   pair_feature[..., 2][ss_idx]
            cur_theta = pair_feature[..., 3][ss_idx]

            neighbor_sortedidx = torch.argsort(ca_dist_map[ss_idx])
            sameSS_res_mask = sstype_idx != cur_sstype_idx

            ## HH stat
            nearest_res_mask_HH = torch.all(torch.stack([sameSS_res_mask, (sstype == 0)]), 0)
            nearest_res_idx_HH = neighbor_sortedidx[nearest_res_mask_HH[neighbor_sortedidx]][0]

            nearset_ca_dist_HH = cur_ca_dist[nearest_res_idx_HH]
            nearset_cb_dist_HH = cur_cb_dist[nearest_res_idx_HH]
            nearset_omega_HH = cur_omega[nearest_res_idx_HH]
            nearset_phi_HH = cur_phi[nearest_res_idx_HH]
            nearset_theta_HH = cur_theta[nearest_res_idx_HH]

            if nearset_ca_dist_HH <= clamp_dist:
                HH_ca_dist_list.append(nearset_ca_dist_HH)
                HH_cb_dist_list.append(nearset_cb_dist_HH)
                HH_omega_list.append(nearset_omega_HH)
                HH_phi_list.append(nearset_phi_HH)
                HH_theta_list.append(nearset_theta_HH)

            ## HE
            if torch.any(sstype == 2):
                nearest_res_mask_HE = torch.all(torch.stack([sameSS_res_mask, (sstype == 2)]), 0)
                nearest_res_idx_HE = neighbor_sortedidx[nearest_res_mask_HE[neighbor_sortedidx]][0]

                nearset_ca_dist_HE = cur_ca_dist[nearest_res_idx_HE]
                nearset_cb_dist_HE = cur_cb_dist[nearest_res_idx_HE]
                nearset_omega_HE = cur_omega[nearest_res_idx_HE]
                nearset_phi_HE = cur_phi[nearest_res_idx_HE]
                nearset_theta_HE = cur_theta[nearest_res_idx_HE]

                if nearset_ca_dist_HE <= clamp_dist:
                    HE_ca_dist_list.append(nearset_ca_dist_HE)
                    HE_cb_dist_list.append(nearset_cb_dist_HE)
                    HE_omega_list.append(nearset_omega_HE)
                    HE_phi_list.append(nearset_phi_HE)
                    HE_theta_list.append(nearset_theta_HE)

    stat_dict = {
        'HH_ca_dist': HH_ca_dist_list,
        'HE_ca_dist': HE_ca_dist_list,
        'HH_cb_dist': HH_cb_dist_list,
        'HE_cb_dist': HE_cb_dist_list,
        'HH_omega': HH_omega_list,
        'HE_omega': HE_omega_list,
        'HH_phi': HH_phi_list,
        'HE_phi': HE_phi_list,
        'HH_theta': HH_theta_list,
        'HE_theta': HE_theta_list,
    }

    return stat_dict



def get_SScentermass_coords(ca_pos, ss3type):
    ss_start_indexs = (torch.where((ss3type[1:] - ss3type[:-1]) != 0)[0] + 1).long()
    ss_start_indexs = torch.cat([torch.LongTensor([0]), ss_start_indexs])
    ss_end_indexs = torch.cat([ss_start_indexs[1:]-1, torch.LongTensor([len(ss3type)])])
    ss_lens = ss_start_indexs[1:] - ss_start_indexs[:-1]
    ss_lens = torch.cat([ss_lens, (len(ss3type) - ss_start_indexs[-1]).unsqueeze(0)])
    start_sstypes = torch.index_select(ss3type, 0, ss_start_indexs)

    SScentermass_coords = []
    for ss_idx, ss in enumerate(start_sstypes):
        ss_len = ss_lens[ss_idx]
        ss_start_index = ss_start_indexs[ss_idx]
        ss_end_index = ss_end_indexs[ss_idx]  

        if ((ss_len > 2) and (ss != 1)):
            gt_ss_pos = ca_pos[ss_start_index: ss_end_index+1]
            SScentermass_coords.append(torch.mean(gt_ss_pos, 0))

    return SScentermass_coords



if __name__ == "__main__":
    from Bio.PDB import PDBParser
    from Bio.PDB.DSSP import DSSP

    pdbfile = '/train14/superbrain/yfliu25/structure_refine/ProtDiff_new2d_inpainting_denoising_mask_partial_aa/savedir/S_Wr_Ws_I_0.70_0.95_V_pre_train_noAA/gen/step_153999/iter_66/1-0/1-0_epoch0_iter_65_traj_5.pdb'

    # df = get_feature_from_dssp(pdbfile, return_type='df', add_ca_coord=True)
    df, coords = get_feature_from_dssp(pdbfile, return_type='df', add_coord=True, return_coords=True)
    df = preprocess_dssp_df(df, add_ss_idx=True)

    # ca_pos = torch.from_numpy(df.to_numpy()[:, -3:].astype(np.float32))
    # sstype = torch.from_numpy(df['SS3enc'].to_numpy().astype(np.int32))
    # sstype_idx = torch.from_numpy(df['SS_idx'].to_numpy().astype(np.int32))
    # # SScentermass_coords = get_SScentermass_coords(ca_pos, sstype)
    # import pdb; pdb.set_trace()
    # HH_dist_list, HE_dist_list = find_nearest_CAdist_inSS(ca_pos, sstype, sstype_idx)
    # import pdb; pdb.set_trace()

    coords = torch.from_numpy(coords.astype(np.float32))
    sstype = torch.from_numpy(df['SS3enc'].to_numpy().astype(np.int32))
    sstype_idx = torch.from_numpy(df['SS_idx'].to_numpy().astype(np.int32))
    # SScentermass_coords = get_SScentermass_coords(ca_pos, sstype)
    import pdb; pdb.set_trace()
    # HH_dist_list, HE_dist_list = find_nearest_CAdist_inSS(coords, sstype, sstype_idx)
    stat_dict = find_nearest_CAfeature_inSS(coords, sstype, sstype_idx)
    import pdb; pdb.set_trace()
