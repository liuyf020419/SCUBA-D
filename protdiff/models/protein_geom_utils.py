import os
from itertools import combinations,permutations

import numpy as np
import torch
import torch.nn.functional as F


NOCBOMAINCHAINATOMS = ["N", "CA", "C"]
NOOMAINCHAINATOMS = ["N", "CA", "C", "CB"]
ANGLE_BINS = [-np.pi, np.pi, 64]
NONCYCLE_ANGLE_BINS = [0, np.pi, 32]
DIST_BINS = [0, 22, 64]


def add_c_beta_from_crd(frame_crd, dtype="Tensor", add_O=False):
    # frame_crd.shape [batchsize, nres, natoms, 3]
    assert len(frame_crd.shape) == 4
    vec_n = frame_crd[:, :, 0]
    vec_ca = frame_crd[:, :, 1]
    vec_c = frame_crd[:, :, 2]
    if add_O:
        vec_o = frame_crd[:, :, 3]
    b = vec_ca - vec_n
    c = vec_c - vec_ca
    if dtype == "Tensor":
        a = torch.cross(b, c)
    else:
        a = np.cross(b, c)
    vec_cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + vec_ca

    if dtype == "Tensor":
        if add_O:
            return torch.cat([
                vec_n[:, :, None], vec_ca[:, :, None], 
                vec_c[:, :, None], vec_cb[:, :, None], 
                vec_o[:, :, None]], 2)
        else:
            return torch.cat([
                vec_n[:, :, None], vec_ca[:, :, None], 
                vec_c[:, :, None], vec_cb[:, :, None]], 2)
    else:
        if add_O:
            return np.concatenate([
                vec_n[:, :, None], vec_ca[:, :, None], 
                vec_c[:, :, None], vec_cb[:, :, None], 
                vec_o[:, :, None]], 2)
        else:
            return np.concatenate([
                vec_n[:, :, None], vec_ca[:, :, None], 
                vec_c[:, :, None], vec_cb[:, :, None]], 2)


def dist_ch(x1, x2, axis=2):
    d = x2-x1 + 1e-10
    d2 = torch.sum(d*d, dim=axis) # √(x2 + y2 + z2)
    return torch.sqrt(d2 + 1e-10)


def angle_ch(x1, x2, x3, degrees=True, axis=2):
    ba = x1 - x2 + 1e-10
    ba = ba / (torch.norm(ba, dim=axis, keepdim=True) + 1e-10)
    bc = x3 - x2 + 1e-10
    bc = bc / (torch.norm(bc, dim=axis, keepdim=True) + 1e-10)
    cosine_angle = torch.clip(torch.sum(ba*bc, dim=axis), min=-1+1e-6, max=1-1e-6)
    if degrees:
        return np.float32(180.0 / np.pi) * torch.acos(cosine_angle)
    else:
        # if torch.any(torch.isnan(torch.acos(cosine_angle))):
        #     import pdb; pdb.set_trace()
        return torch.acos(cosine_angle)


def torsion_ch(x1, x2, x3, x4, degrees=True, axis=2):
    """Praxeolitic formula
    1 sqrt, 1 cross product"""
    b0 = -1.0*(x2 - x1) + 1e-10
    b1 = x3 - x2 + 1e-10
    b2 = x4 - x3 + 1e-10
    # normalize b1 so that it does not influence magnitude of vector
    # rejections that come next
    b1 = b1 / (torch.norm(b1, dim=axis, keepdim=True) + 1e-10)

    # vector rejections
    # v = projection of b0 onto plane perpendicular to b1
    #   = b0 minus component that aligns with b1
    # w = projection of b2 onto plane perpendicular to b1
    #   = b2 minus component that aligns with b1
    v = b0 - torch.sum(b0*b1, dim=axis, keepdim=True) * b1
    w = b2 - torch.sum(b2*b1, dim=axis, keepdim=True) * b1

    # angle between v and w in a plane is the torsion angle
    # v and w may not be normalized but that's fine since tan is y/x
    x = torch.sum(v*w, dim=axis)
    b1xv = torch.cross(b1, v)
    y = torch.sum(b1xv*w, dim=axis)
    # import pdb; pdb.set_trace()
    if degrees:
        return np.float32(180.0 / np.pi) * torch.atan2(y, x)
    else:
        return torch.atan2(y, x)


def preprocess_pair_feature(pair_feature):
    CB_dist = pair_feature[..., 0]
    omega_tor = pair_feature[..., 1]
    phi_ang = pair_feature[..., 2]
    theta_tor = pair_feature[..., 3]
    # import pdb; pdb.set_trace()
    CB_dist = torch.clamp(CB_dist, min=0, max=20) / 20
    omega_tor = torch.clamp(omega_tor, min=-np.pi, max=np.pi) / np.pi
    phi_ang = torch.clamp(phi_ang, min=0, max=np.pi) / np.pi
    theta_tor = torch.clamp(theta_tor, min=-np.pi, max=np.pi) / np.pi

    pair_feature = torch.cat([CB_dist[..., None], 
                              omega_tor[..., None], 
                              phi_ang[..., None], 
                              theta_tor[..., None]], -1)
    # import pdb; pdb.set_trace()
    return pair_feature


def rbf(D, num_rbf=16):
    device = D.device
    D_min, D_max, D_count = 2., 22., num_rbf
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1,1,1,-1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)
    RBF = torch.exp(-((D_expand - D_mu) / D_sigma)**2)
    return RBF


def triangle_encode_angles(angles: torch.Tensor, tri_num=5):
    """
    angles: translate to triangle form
    k: cosine(k*(torsion))/sine(k*(torsion))
    """
    flatten_angles = angles.reshape(-1, 1)
    flatten_encoded_angles = torch.cat([torch.cat([torch.cos((k+1) * flatten_angles), 
                                                   torch.sin((k+1) * flatten_angles)], -1) 
                                                   for k in range(tri_num)], -1)
    new_shape = list(angles.shape) + [tri_num * 2]
    return flatten_encoded_angles.reshape(new_shape)


def preprocess_pair_feature_advance(pair_feature, rbf_encode=True, num_rbf=16, tri_encode=True, tri_num=5):
    cb_dist = pair_feature[..., 0]
    omega_tor = pair_feature[..., 1]
    phi_ang = pair_feature[..., 2]
    theta_tor = pair_feature[..., 3]

    if rbf_encode:
        encoded_cb_dist = rbf(cb_dist, num_rbf)
    else:
        encoded_cb_dist = (torch.clamp(cb_dist, min=0, max=20) / 20)[..., None]
    if tri_encode:
        encoded_omega_tor = triangle_encode_angles(omega_tor, tri_num)
        encoded_phi_ang = triangle_encode_angles(phi_ang, tri_num)
        encoded_theta_tor = triangle_encode_angles(theta_tor, tri_num)
    else:
        encoded_omega_tor = (torch.clamp(omega_tor, min=-np.pi, max=np.pi) / np.pi)[..., None]
        encoded_phi_ang = (torch.clamp(phi_ang, min=0, max=np.pi) / np.pi)[..., None]
        encoded_theta_tor = (torch.clamp(theta_tor, min=-np.pi, max=np.pi) / np.pi)[..., None]

    pair_feature = torch.cat([encoded_cb_dist, 
                              encoded_omega_tor, 
                              encoded_phi_ang, 
                              encoded_theta_tor], -1)
    return pair_feature


def calc_beta_dist(beta_coords):
    dist = torch.sqrt(torch.sum(
        torch.square(beta_coords[:, :, None] - beta_coords[:, None]), -1) + 1e-10)
    return dist


def generate_pair_from_pos(batch_pos, degrees=False):
    assert len(batch_pos.shape) == 4
    assert batch_pos.shape[2] == 4 # at least 4 or 5 considering O atoms
    # import pdb; pdb.set_trace()
    batchsize, res_len = batch_pos.shape[:2]

    N_atoms = batch_pos[:, :, 0]
    CA_atoms = batch_pos[:, :, 1]
    C_atoms = batch_pos[:, :, 2]
    CB_atoms = batch_pos[:, :, 3]

    # cacluate CB dist map
    CB_dist_map = calc_beta_dist(CB_atoms)

    # calculate omega torsion map
    ca1 = CA_atoms[:, :, None].repeat(1, 1, res_len, 1)
    cb1 = CB_atoms[:, :, None].repeat(1, 1, res_len, 1)
    cb2 = CB_atoms[:, None].repeat(1, res_len, 1, 1)
    ca2 = CA_atoms[:, None].repeat(1, res_len, 1, 1)
    omega_torsion_map = torsion_ch(ca1, cb1, cb2, ca2, degrees=degrees, axis=3)

    # calculate phi angle map
    # same x1, x2 and x3
    phi_angle_map = angle_ch(ca1, cb1, cb2, degrees=degrees, axis=3)

    # calculate theta torsion map
    n1 = N_atoms[:, :, None].repeat(1, 1, res_len, 1)
    theta_torsion_map = torsion_ch(n1, ca1, cb1, cb2, degrees=degrees, axis=3)
    # import pdb; pdb.set_trace()
    pair_feature = torch.cat([CB_dist_map[..., None], 
                              omega_torsion_map[..., None], 
                              phi_angle_map[..., None], 
                              theta_torsion_map[..., None]], 
                            axis=-1)
    # torch.any(torch.isnan(batch_pos))
    if torch.any(torch.isnan(pair_feature)):
        import pdb; pdb.set_trace()

    return pair_feature


def descrete_angle(angle, edges=None):
    if edges is None:
        edges = ANGLE_BINS
    min_, max_, nbin_ = edges
    angle = (angle - min_) * nbin_ / (max_ - min_)
    angle = angle.int()
    angle = torch.clip(angle, 0, nbin_-1)
    return angle


def descrete_dist(dist, edges=None):
    if edges is None:
        edges = DIST_BINS
    min_, max_, nbin_ = edges
    dist = (dist - min_) * nbin_ / (max_ - min_)
    dist = dist.int()
    dist = torch.clip(dist, 0, nbin_-1)
    return dist


def descrete_2d_maps(feature2d, global_config):
    CB, omega, phi, theta = \
        feature2d[..., 0], feature2d[..., 1], feature2d[..., 2], feature2d[..., 3] 
    # import pdb; pdb.set_trace()
    CB = descrete_dist(CB, global_config.cb_bin)
    omega = descrete_angle(omega, global_config.omega_bin)
    phi = descrete_angle(phi, global_config.phi_bin)
    theta = descrete_angle(theta, global_config.theta_bin)

    feature2d = torch.cat([ CB[..., None], 
                            omega[..., None], 
                            phi[..., None], 
                            theta[..., None] ],
                        axis=-1)
    return feature2d




def get_internal_angles3(batch_pos, degrees=False):
    assert len(batch_pos.shape) == 4
    batchsize, L, atom_num, _ = batch_pos.shape
    
    N_atoms = batch_pos[:, :, 0]
    CA_atoms = batch_pos[:, :, 1]
    C_atoms = batch_pos[:, :, 2]

    # calculate phi torsion
    phi_c1 = C_atoms[:, :-1, :]
    phi_n = N_atoms[:, 1:, :]
    phi_ca = CA_atoms[:, 1:, :]
    phi_c = C_atoms[:, 1:, :]
    # calculate psi torsion
    psi_n1 = N_atoms[:, :-1, :]
    psi_ca = CA_atoms[:, :-1:, :]
    psi_c = C_atoms[:, :-1, :]
    psi_n2 = N_atoms[:, 1:, :]
    # calculate omega torsion
    omega_ca_1 = CA_atoms[:, :-1, :]
    omega_c = C_atoms[:, :-1, :]
    omega_n = N_atoms[:, 1:, :]
    omega_ca_2 = CA_atoms[:, 1:, :]

    phi_torsion = torsion_ch(phi_c1, phi_n, phi_ca, phi_c, degrees=degrees, axis=2)
    phi_torsion = F.pad(phi_torsion, (1, 0), "constant", 0)

    psi_torsion = torsion_ch(psi_n1, psi_ca, psi_c, psi_n2, degrees=degrees, axis=2)
    psi_torsion = F.pad(psi_torsion, (0, 1), "constant", 0)

    omega_torsion = torsion_ch(omega_ca_1, omega_c, omega_n, omega_ca_2, degrees=degrees, axis=2)
    omega_torsion = F.pad(omega_torsion, (0, 1), "constant", 0)

    internal_torsion = torch.stack([phi_torsion, psi_torsion, omega_torsion], -1)

    return internal_torsion





def get_internal_angles(batch_pos, degrees=False):
    assert len(batch_pos.shape) == 4
    batchsize, L, atom_num, _ = batch_pos.shape
    
    N_atoms = batch_pos[:, :, 0]
    CA_atoms = batch_pos[:, :, 1]
    C_atoms = batch_pos[:, :, 2]

    # calculate phi torsion
    phi_c1 = C_atoms[:, :-1, :]
    phi_n = N_atoms[:, 1:, :]
    phi_ca = CA_atoms[:, 1:, :]
    phi_c = C_atoms[:, 1:, :]
    # calculate psi torsion
    psi_n1 = N_atoms[:, :-1, :]
    psi_ca = CA_atoms[:, :-1:, :]
    psi_c = C_atoms[:, :-1, :]
    psi_n2 = N_atoms[:, 1:, :]

    phi_torsion = torsion_ch(phi_c1, phi_n, phi_ca, phi_c, degrees=degrees, axis=2)
    phi_torsion = F.pad(phi_torsion, (1, 0), "constant", 0)

    psi_torsion = torsion_ch(psi_n1, psi_ca, psi_c, psi_n2, degrees=degrees, axis=2)
    psi_torsion = F.pad(psi_torsion, (0, 1), "constant", 0)

    internal_torsion = torch.stack([phi_torsion, psi_torsion], -1)

    return internal_torsion


def get_descrete_dist(all_atoms, dist_type, distogram_args, return_dist_map=False):
    assert dist_type in ['ca-ca', 'n-n', 'c-c', 'ca-n', 'ca-c', 'n-c']
    atom1, atom2 = dist_type.split('-')

    n_atoms_coord  = all_atoms[..., 0, :]
    ca_atoms_coord = all_atoms[..., 1, :]
    c_atoms_coord  = all_atoms[..., 2, :]

    atom1_coord = eval(f'{atom1}_atoms_coord')
    atom2_coord = eval(f'{atom2}_atoms_coord')

    dist_map = torch.sqrt(torch.sum(torch.square(atom1_coord[:, None] - atom2_coord[:, :, None]), -1) + 1e-10)
    dist_map_descrete = descrete_dist(dist_map, distogram_args).long()

    if return_dist_map:
        return dist_map_descrete, dist_map
    else:
        return dist_map_descrete


def get_descrete_feature(all_atoms, return_angle_mask=True, mask_base_ca=True, mask_cutoff=10):
    ca_coord = all_atoms[..., 1, :]
    ca_dist_map = torch.sqrt(torch.sum(torch.square(ca_coord[:, None] - ca_coord[:, :, None]), -1) + 1e-10)

    pair_features = generate_pair_from_pos(all_atoms)
    cb_dist_map = pair_features[..., 0]
    omega_map   = pair_features[..., 1]
    phi_map     = pair_features[..., 2]
    theta_map   = pair_features[..., 3]

    if mask_base_ca:
        all_angle_masks = ca_dist_map < mask_cutoff # B, L, L
    else:
        all_angle_masks = cb_dist_map < mask_cutoff # B, L, L

    descrete_ca = descrete_dist(ca_dist_map, DIST_BINS)
    descrete_cb = descrete_dist(cb_dist_map, DIST_BINS)
    descrete_omega = descrete_dist(omega_map, ANGLE_BINS)
    descrete_phi = descrete_dist(phi_map, NONCYCLE_ANGLE_BINS)
    descrete_theta = descrete_dist(theta_map, ANGLE_BINS)

    descrete_pair = torch.stack([
        descrete_ca, descrete_cb,
        descrete_omega, descrete_theta,
        descrete_phi], -1)

    if return_angle_mask:
        return descrete_pair, all_angle_masks
    else:
        return descrete_pair


if __name__ == "__main__":
    batchsize = 10
    seq_len = 128
    # dist_pair = torch.rand(batchsize, seq_len, seq_len)
    # import pdb; pdb.set_trace()
    # rbf(dist_pair)

    coords = torch.rand(batchsize, seq_len, 3, 3)
    internal_torsion = get_internal_angles(coords)
    import pdb; pdb.set_trace()