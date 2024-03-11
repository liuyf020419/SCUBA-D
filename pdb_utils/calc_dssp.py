import os
import sys

import numpy as np
import pandas as pd
import torch

from Bio.PDB import PDBParser, MMCIFParser
from Bio.PDB.DSSP import DSSP


ENCODESS32NUM = {
    "H": 0,
    "C": 1,
    "L": 1,
    "E": 2
}

def get_feature_from_dssp(pdbfile, return_type = "list", file_type: str=None, add_ca_coord=False):
    assert return_type in ['list', 'dict', 'df']

    if file_type.upper() == 'PDB':
        p = PDBParser()
        structure = p.get_structure(os.path.basename(pdbfile).split('.')[0], pdbfile)
        model = structure[0]
    elif file_type.upper() == 'CIF':
        p = MMCIFParser()
        structure = p.get_structure(os.path.basename(pdbfile).split('.')[0], pdbfile)
        model = structure[0]
    else:
        raise ValueError()

    dssp = DSSP(model, pdbfile, dssp='mkdssp', acc_array="Sander", file_type=file_type)

    if add_ca_coord:
        ca_coords = np.stack([atom_inf.get_coord() for atom_inf in list(model.get_atoms()) if atom_inf.name == 'CA'])

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
        df['X_CA'] = ca_coords[:, 0]
        df['Y_CA'] = ca_coords[:, 1]
        df['Z_CA'] = ca_coords[:, 2]

        return df


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

    if 'X_CA' in df.columns:
        new_dict['X_CA'] = df['X_CA']
        new_dict['Y_CA'] = df['Y_CA']
        new_dict['Z_CA'] = df['Z_CA']

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


if __name__ == "__main__":
    from Bio.PDB import PDBParser
    from Bio.PDB.DSSP import DSSP

    # pdbfile = '/train14/superbrain/yfliu25/structure_refine/ProtDiff_new2d_inpainting_denoising_mask_partial_aa/savedir/S_Wr_Ws_I_0.70_0.95_V_pre_train_noAA/gen/step_153999/iter_66/1-0/1-0_epoch0_iter_65_traj_5.pdb'
    pdbfile = '4ogs.pdb'

    df = get_feature_from_dssp(pdbfile, return_type='df', file_type='pdb', add_ca_coord=True)
    df = preprocess_dssp_df(df, add_ss_idx=True)
    print(''.join(df['SS3'].tolist()))