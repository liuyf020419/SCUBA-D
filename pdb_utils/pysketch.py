import os
from copy import deepcopy

import torch
import torch.nn.functional as F

import numpy as np

from pysketch_stat import load_gmm
from gen_cart_from_ic import build_cart_from_tors
from pyalign import kabsch_rotation, rotrans_coords
from write_pdb import write_multichain_from_atoms

import sys
sys.path.append("protdiff/models")
from protein_utils import rigid
from protein_utils.add_o_atoms import add_atom_O


ic_params_dict = np.load(f'pdb_utils/sketch_dat/gmm_ic3_params_12000.npy', allow_pickle=True).item()

helix_ic_gmm = load_gmm(ic_params_dict['helix'])
beta_ic_gmm = load_gmm(ic_params_dict['beta'])
coil_ic_gmm = load_gmm(ic_params_dict['coil'])

standard_ic3 = {
    'helix': np.deg2rad([-57, -47, 180]),
    'beta': np.deg2rad([-119, 113, 180]),
}

SS3_name_dict = {
    'H': 'helix',
    'E': 'beta',
    'C': 'coil',
}

SS3_num_to_name = {
    0: 'helix',
    1: 'coil',
    2: 'beta'
}
SS3_name_to_num = {v:k for k, v in SS3_num_to_name.items()}


def rotaxis_to_rotmatrix(angle, axis):
    c = np.cos(angle)
    s = np.sin(angle)
    t = 1 - c
    x, y, z = axis
    rot = np.zeros((3, 3))
    # 1st row
    rot[0, 0] = t * x * x + c
    rot[0, 1] = t * x * y + s * z
    rot[0, 2] = t * x * z - s * y
    # 2nd row
    rot[1, 0] = t * x * y - s * z
    rot[1, 1] = t * y * y + c
    rot[1, 2] = t * y * z + s * x
    # 3rd row
    rot[2, 0] = t * x * z + s * y
    rot[2, 1] = t * y * z - s * x
    rot[2, 2] = t * z * z + c
    return rot.astype(np.float16)



def gen_loops(length: int, mass_center=None, ca_noise_scale=15):
    random_quat = rigid.rand_quat([1, length])
    random_coord = rigid.noising_coord(torch.zeros((1, length, 3)), ca_noise_scale)

    random_affine = torch.cat([random_quat, random_coord], -1)

    loops_crds3 = rigid.affine_to_pos(random_affine.reshape(-1, 7)).reshape(1, length, -1, 3)
    loops_crds4 = add_atom_O(loops_crds3[0].detach().cpu().numpy()[..., :3, :])
    return loops_crds4 + mass_center



def gen_continuous_peptides(length: int, sstype:str, standard=False):
    assert sstype in ['helix', 'beta', 'coil']

    if standard:
        if sstype in ['helix', 'beta']:
            gen_torsions = np.stack([standard_ic3[sstype] for _ in range(length)])
        elif sstype == 'coil':
            torsion_generator = coil_ic_gmm
            gen_torsions = torsion_generator.sample(length)[0]
        gen_torsions = torch.from_numpy(gen_torsions).float()

    else:
        if sstype == 'helix':
            torsion_generator = helix_ic_gmm
        elif sstype == 'beta':
            torsion_generator = beta_ic_gmm
        elif sstype == 'coil':
            torsion_generator = coil_ic_gmm
        gen_torsions = torch.from_numpy(torsion_generator.sample(length)[0]).float()

    coords4 = build_cart_from_tors(gen_torsions).numpy()

    return coords4



def rotrans_peptides(peptides, nterms_pos, orientation):
    assert len(peptides.shape) == 3
    peptides_mass_center = np.mean(peptides[:, 1], 0)

    target_orientation = np.array(orientation)
    norm_target_orientation = target_orientation/np.linalg.norm(target_orientation, keepdims=True)

    peptides_orientation = peptides[1:, 1].mean(0) - peptides[:-1, 1].mean(0)
    norm_peptides_orientation = peptides_orientation/np.linalg.norm(peptides_orientation, keepdims=True)

    rot_angles = np.arccos(np.dot(norm_target_orientation, norm_peptides_orientation).sum())
    rot_axis = np.cross(norm_peptides_orientation, norm_target_orientation)
    norm_rot_axis = rot_axis/np.linalg.norm(rot_axis, keepdims=True)

    # rotmaxtrix = R.from_rotvec(rot_angles * rot_axis).as_matrix()
    rotmaxtrix = rotaxis_to_rotmatrix(rot_angles, norm_rot_axis)
    rotransed_peptide = rotrans_coords(peptides - peptides_mass_center, (rotmaxtrix, np.array(nterms_pos)))

    rotransed_peptides_mass_center = np.mean(rotransed_peptide[:, 1], 0)
    rotransed_peptides_nterm_center = np.mean(rotransed_peptide[0, :], 0)
    vec_from_rotransed_nterm_to_mass_center = rotransed_peptides_mass_center - rotransed_peptides_nterm_center

    return rotransed_peptide + vec_from_rotransed_nterm_to_mass_center



def gen_peptides_ref_native_peptides(nat_peptides, sstype: str, standard=False):
    assert len(nat_peptides.shape) == 3
    length = nat_peptides.shape[0]
    nat_peptides_mass_center = np.mean(nat_peptides[:, 1], 0)

    new_peptides = gen_continuous_peptides(length, sstype, standard=standard)
    new_peptides_mass_center = np.mean(new_peptides[:, 1], 0)
    # print(nat_peptides.shape, 'nat')
    # print(new_peptides.shape, 'new')
    peptide_rot = kabsch_rotation(
            (new_peptides - new_peptides_mass_center).reshape(-1, 3), 
            (nat_peptides - nat_peptides_mass_center).reshape(-1, 3), 'np')
        
    rotransed_new_peptide = rotrans_coords(
        new_peptides - new_peptides_mass_center, 
        (peptide_rot, nat_peptides_mass_center))

    return rotransed_new_peptide


def gen_peptides_zero_mass_center_peptides(ss_length, sstype: str, standard=False):
    # assert len(nat_peptides.shape) == 3
    # length = nat_peptides.shape[0]
    # nat_peptides_mass_center = np.mean(nat_peptides[:, 1], 0)

    new_peptides = gen_continuous_peptides(ss_length, sstype, standard=standard)
    new_peptides_mass_center = np.mean(new_peptides[:, 1], 0)
    rotransed_new_peptide = new_peptides - new_peptides_mass_center
    # print(nat_peptides.shape, 'nat')
    # print(new_peptides.shape, 'new')
    # peptide_rot = kabsch_rotation(
    #         (new_peptides - new_peptides_mass_center).reshape(-1, 3), 
    #         (nat_peptides - nat_peptides_mass_center).reshape(-1, 3), 'np')
        
    # rotransed_new_peptide = rotrans_coords(
    #     new_peptides - new_peptides_mass_center, 
    #     (peptide_rot, nat_peptides_mass_center))

    return rotransed_new_peptide




def build_sketch_from_par(sketch_par_f, standard=False):
    par_list = []
    nterm_pos = []
    with open(sketch_par_f, 'r') as reader:
        for line in reader.readlines():
            par = line.strip().split()
            par_list.append(par)
            if (len(par) == 8):
                nterm_pos.append(list(map(lambda x: float(x), par[2:5] )))
    ss_pos_center = np.stack(nterm_pos).mean(0)
    ss_peptides_coords = []
    for par in par_list:
        if len(par) == 1:
            if int(par[0]) > 0:
                centered_loops = gen_loops(int(par[0]), ss_pos_center)
                ss_peptides_coords.append(centered_loops)
        else:
            assert len(par) == 8
            init_peptides = gen_continuous_peptides(int(par[1]), SS3_name_dict[par[0]], standard=standard)
            rotransed_peptides = rotrans_peptides(
                init_peptides, 
                list(map(lambda x: float(x), par[2:5] )),
                list(map(lambda x: float(x), par[5: ] ))
                )
            ss_peptides_coords.append(rotransed_peptides)

    overall_coords = np.concatenate(ss_peptides_coords)

    return overall_coords


def debug_kabschalign(length, sstype, standard=True):
    init_coords4 = gen_continuous_peptides(length, SS3_name_dict[sstype], standard)
    gen_coords4 = gen_peptides_ref_native_peptides(init_coords4, SS3_name_dict[sstype], standard)
    write_multichain_from_atoms([init_coords4.reshape(-1, 3)], f'{pdb_root}/debug_init_align_ss_{sstype}_length_{length}_standard_{standard}.pdb', natom=4)
    write_multichain_from_atoms([gen_coords4.reshape(-1, 3)], f'{pdb_root}/debug_rotransed_align_ss_{sstype}_length_{length}_standard_{standard}.pdb', natom=4)
    
    
def parse_sstypefile(sstypefile, sstype_line_idx=0, return_encoded=True):
    with open(sstypefile, 'r') as reader:
        all_lines = reader.readlines()
    sstype = all_lines[sstype_line_idx].strip()
    
    if return_encoded:
        return np.array([SS3_name_to_num[SS3_name_dict[ss]] for ss in sstype ])
    else:
        return sstype
        


if __name__ == '__main__':
    sketch_par_f = '/train14/superbrain/yfliu25/structure_refine/monomer_joint_PriorDDPM_ESM1b_unfixCEhead_Dnet_LE_MPNN_LC_trans_newmask/pdb_utils/sketch_dat/tim9_sketch_noloop.txt'
    pdb_root = '/train14/superbrain/yfliu25/structure_refine/monomer_joint_PriorDDPM_ESM1b_unfixCEhead_Dnet_LE_MPNN_LC_trans_newmask/pdb_utils/sketch_dat'
    overall_coords = build_sketch_from_par(sketch_par_f, standard=False)
    write_multichain_from_atoms([overall_coords.reshape(-1, 3)], f'{pdb_root}/debug_tim9_0loops.pdb', natom=4)
