import torch
import torch.nn.functional as F
import sys

from ..folding_af2 import all_atom
from ..folding_af2 import residue_constants
from ..folding_af2 import utils
from ..folding_af2 import r3


# calculate on average of valid atoms in 3GCB_A
# STD_RIGID_COORD = torch.FloatTensor(
#     [[-1.4589e+00, -2.0552e-07,  2.1694e-07],
#      [ 0.0000e+00,  0.0000e+00,  0.0000e+00],
#      [ 5.4261e-01,  1.4237e+00,  1.0470e-07],
#      [ 5.2744e-01, -7.8261e-01, -1.2036e+00],]
# )

STD_RIGID_COORD = torch.FloatTensor(
    [[-0.525,  1.363,  0.000],
     [ 0.000,  0.000,  0.000],
     [ 1.526, -0.000, -0.000],
     [ 0.627,  1.062,  0.000]]
)


AATYPE_BASE_BACKB = torch.zeros((20, 3, 3)).float()
for aa in residue_constants.restypes:
    aa_order = residue_constants.restype_order[aa]
    aa3 = residue_constants.restype_1to3[aa]
    aa_rigid = residue_constants.rigid_group_atom_positions[aa3]

    aa_atom3 = torch.FloatTensor(
        [aa_rigid[0][2], aa_rigid[1][2], aa_rigid[2][2]]
    )
    AATYPE_BASE_BACKB[aa_order] = aa_atom3


def coord_to_frame_affine(coord):
    if len(coord.shape) == 3:
        coord = coord[None]
    batch_size, res_num, atom_num = coord.shape[:3]
    device = coord.device

    alanine_idx = residue_constants.restype_order_with_x["A"]
    aatype = torch.LongTensor([alanine_idx] * res_num)[None].repeat(batch_size, 1).to(device)
    all_atom_positions = F.pad(coord, (0,0,0,37-atom_num), "constant", 0)
    all_atom_mask = torch.ones(batch_size, res_num, 37).to(device)
    all_atom_frame = all_atom.atom37_to_frames(aatype, all_atom_positions, all_atom_mask)['rigidgroups_gt_frames']

    all_atom_affine = r3.rigids_to_quataffine_m(r3.rigids_from_tensor_flat12(all_atom_frame)).to_tensor()[..., 0, :]

    return {
        'frame': all_atom_frame,
        'affine': all_atom_affine
    }


def atom3_to_backbone_frame(aatype, atom3, atom_mask=None):
    shapes = [s for s in atom3.shape]
    if atom_mask is None:
        atom_mask = torch.ones(atom3.shape[:-1]).to(atom_mask)

    frames = all_atom.atom37_to_backbone_frames(aatype, atom3, atom_mask)
    backb_frames = frames['rigidgroups_backbone_frames']
    rot = torch.reshape(backb_frames[..., :9], shapes[:-2] + [3, 3]) 
    trans = torch.reshape(backb_frames[..., 9:], shapes[:-2] + [3])
    return rot, trans


def backbone_frame_to_atom3(aatype, rot, trans):
    base_atom3 = AATYPE_BASE_BACKB.to(aatype.device)
    seq_base = base_atom3[aatype]
    atom3 = torch.bmm(seq_base, torch.transpose(rot, -1,-2)) + trans.unsqueeze(-2)
    return atom3


def backbone_frame_to_atom3_std(rot, trans, atomnum=3):
    assert (atomnum == 3) or (atomnum == 4)
    std_coord = STD_RIGID_COORD[:atomnum].to(trans.device)
    atom3 = torch.bmm(std_coord[None].repeat(trans.shape[0], 1, 1), torch.transpose(rot, -1,-2)) + trans.unsqueeze(-2)
    return atom3



def coord_to_frame_affine(coord):
    if len(coord.shape) == 3:
        coord = coord[None]
    batch_size, res_num, atom_num = coord.shape[:3]
    device = coord.device

    alanine_idx = residue_constants.restype_order_with_x["A"]
    aatype = torch.LongTensor([alanine_idx] * res_num)[None].repeat(batch_size, 1).to(device)
    all_atom_positions = F.pad(coord, (0,0,0,37-atom_num), "constant", 0)
    all_atom_mask = torch.ones(batch_size, res_num, 37).to(device)
    all_atom_frame = all_atom.atom37_to_frames(aatype, all_atom_positions, all_atom_mask)['rigidgroups_gt_frames']

    all_atom_affine = r3.rigids_to_quataffine_m(r3.rigids_from_tensor_flat12(all_atom_frame)).to_tensor()[..., 0, :]

    return {
        'frame': all_atom_frame,
        'affine': all_atom_affine
    }


def convert_to_local(
        coord,  # [B, L, N, 3]
        rot,    # [B, L, 3，3]
        trans   # [B, L, 3]
    ):
    batch_size, num_res, num_atoms = coord.shape[:3]
    coord_expand = torch.tile(coord[:, None], (1, num_res, 1, 1, 1))
    trans_expand = torch.tile(trans[:, :, None], (1, 1, num_res, 1))
    coord_expand = coord_expand - trans_expand.unsqueeze(-2)

    inv_rot = torch.transpose(rot, -1, -2)
    rot_expand = torch.tile(inv_rot[:, :, None], (1, 1, num_res, 1, 1))
    
    coord_flat = torch.reshape(coord_expand, (-1, num_atoms, 3))
    rot_flat = torch.reshape(rot_expand, (-1, 3, 3))

    local_coord = torch.bmm(coord_flat, rot_flat.transpose(-1, -2))
    local_coord = torch.reshape(local_coord, (batch_size, num_res, num_res, num_atoms, 3))
    return local_coord


# def backbone_fape_loss(
#         pred_coord,     # [B, L, N, 3]
#         pred_rot,       # [B, L, 3, 3]
#         pred_trans,     # [B, L, 3]
#         ref_coord,      # [B, L, N, 3]
#         ref_rot,        # [B, L, 3, 3]
#         ref_trans,      # [B, L, 3]
#         mask,           # [B, L]
#         clamp_dist = 10.0,
#         length_scale = 1.0
#     ):
#     pred_coord_local = convert_to_local(pred_coord, pred_rot, pred_trans)   # [B, L, L, N, 3]
#     ref_coord_local = convert_to_local(ref_coord, ref_rot, ref_trans)       # [B, L, L, N, 3]

#     mask2d = mask[..., None] * mask[..., None, :]
#     dist_map = torch.sqrt(torch.sum((pred_coord_local - ref_coord_local) ** 2, -1) + 1e-6)
#     dist_map = torch.mean(dist_map, -1)
#     dist_map_clamp = dist_map.clamp(max = clamp_dist)
    
#     dist_map = dist_map / length_scale
#     dist_map_clamp = dist_map_clamp / length_scale

#     loss = torch.sum(dist_map * mask2d) / (torch.sum(mask2d) + 1e-6)
#     loss_clamp = torch.sum(dist_map_clamp * mask2d) / (torch.sum(mask2d) + 1e-6)
#     return loss, loss_clamp



def backbone_fape_loss(
        pred_coord,     # [B, L, N, 3]
        pred_rot,       # [B, L, 3, 3]
        pred_trans,     # [B, L, 3]
        ref_coord,      # [B, L, N, 3]
        ref_rot,        # [B, L, 3, 3]
        ref_trans,      # [B, L, 3]
        mask,           # [B, L]
        clamp_dist = 10.0,
        length_scale = 1.0,
        mask_2d = None, 
        return_nosum = False
    ):
    # import pdb; pdb.set_trace()
    pred_coord_local = convert_to_local(pred_coord, pred_rot, pred_trans)   # [B, L, L, N, 3]
    ref_coord_local = convert_to_local(ref_coord, ref_rot, ref_trans)       # [B, L, L, N, 3]
    # import pdb; pdb.set_trace()
    if mask_2d is None:
        mask2d = mask[..., None] * mask[..., None, :]
    else:
        mask2d = mask_2d
    dist_map = torch.sqrt(torch.sum((pred_coord_local - ref_coord_local) ** 2, -1) + 1e-6)
    dist_map = torch.mean(dist_map, -1)
    dist_map_clamp = dist_map.clamp(max = clamp_dist)
    
    dist_map = dist_map / length_scale
    dist_map_clamp = dist_map_clamp / length_scale

    loss = torch.sum(dist_map * mask2d) / (torch.sum(mask2d) + 1e-6)
    loss_clamp = torch.sum(dist_map_clamp * mask2d) / (torch.sum(mask2d) + 1e-6)
    if return_nosum:
        return loss, loss_clamp, dist_map * mask2d
    else:
        return loss, loss_clamp
    



def find_structural_violations_batch(
    batch,
    atom14_pred_positions,  # (B, N, 14, 3)
    config
    ):
    """Computes several checks for structural violations."""
    atom14_pred_positions= atom14_pred_positions.float()
    batchsize, L, atoms_num = atom14_pred_positions.shape[:3]
    # import pdb; pdb.set_trace()
    # seq_mask: (B, N)
    # pred_atom_mask: (B, N, natoms)
    pred_atom_mask = batch['seq_mask'][..., None].repeat(1, 1, atoms_num)
    # Compute between residue backbone violations of bonds and angles.
    pseudo_aatype = torch.zeros((batchsize, L), dtype=torch.long, device=atom14_pred_positions.device)
    # import pdb; pdb.set_trace()
    connection_violations = all_atom.between_residue_bond_loss_batch(
        pred_atom_positions=atom14_pred_positions,
        pred_atom_mask=pred_atom_mask.float(), #.astype(jnp.float32),
        residue_index=batch['single_res_rel'].float(), #.astype(jnp.float32),
        aatype=pseudo_aatype,
        tolerance_factor_soft=config.violation_tolerance_factor,
        tolerance_factor_hard=config.violation_tolerance_factor)

    # Compute the Van der Waals radius for every atom
    # (the first letter of the atom name is the element type).
    # Shape: (N, 14).
    atomtype_radius = torch.FloatTensor([
        residue_constants.van_der_waals_radius[name[0]]
        for name in residue_constants.atom_types
    ]).to(atom14_pred_positions.device)
    num_res = L
    # (B, N, natoms)
    atom14_atom_radius = pred_atom_mask * atomtype_radius[None, None].repeat(batchsize, num_res, 1)[..., :atoms_num]
    # atom14_atom_radius = batch['seq_mask'] * torch.gather(
    #     atomtype_radius.unsqueeze(0).repeat(num_res, 1), 1, batch['residx_atom14_to_atom37'])

    # Compute the between residue clash loss.
    between_residue_clashes = all_atom.between_residue_clash_loss_batch(
        atom14_pred_positions=atom14_pred_positions,
        atom14_atom_exists=pred_atom_mask,
        atom14_atom_radius=atom14_atom_radius,
        residue_index=batch['single_res_rel'],
        overlap_tolerance_soft=config.clash_overlap_tolerance,
        overlap_tolerance_hard=config.clash_overlap_tolerance,
        natoms=atoms_num)

    # Compute all within-residue violations (clashes,
    # bond length and angle violations).
    restype_atom14_bounds = residue_constants.make_atom14_dists_bounds(
        overlap_tolerance=config.clash_overlap_tolerance,
        bond_length_tolerance_factor=config.violation_tolerance_factor)
    restype_atom14_bounds = {k: v[:, :atoms_num, :atoms_num] for k, v in restype_atom14_bounds.items()}
    # import pdb; pdb.set_trace()
    atom14_dists_lower_bound = torch.gather(
        restype_atom14_bounds['lower_bound'][None].repeat(batchsize, 1, 1, 1).to(pseudo_aatype.device), 1, pseudo_aatype[..., None, None].repeat(1, 1, atoms_num, atoms_num))
    atom14_dists_upper_bound = torch.gather(
        restype_atom14_bounds['upper_bound'][None].repeat(batchsize, 1, 1, 1).to(pseudo_aatype.device), 1, pseudo_aatype[..., None, None].repeat(1, 1, atoms_num, atoms_num))
    within_residue_violations = all_atom.within_residue_violations_batch(
        atom14_pred_positions=atom14_pred_positions,
        atom14_atom_exists=pred_atom_mask,
        atom14_dists_lower_bound=atom14_dists_lower_bound,
        atom14_dists_upper_bound=atom14_dists_upper_bound,
        tighten_bounds_for_loss=0.0,
        natoms=atoms_num)
    # Combine them to a single per-residue violation mask (used later for LDDT).
    per_residue_violations_mask = torch.max(torch.stack([
        connection_violations['per_residue_violation_mask'],
        torch.max(between_residue_clashes['per_atom_clash_mask'], dim=-1)[0],
        torch.max(within_residue_violations['per_atom_violations'],
                dim=-1)[0]], 1), dim=1)[0]

    return {
        'between_residues': {
            'bonds_c_n_loss_mean':
                connection_violations['c_n_loss_mean'],  # ()
            'angles_ca_c_n_loss_mean':
                connection_violations['ca_c_n_loss_mean'],  # ()
            'angles_c_n_ca_loss_mean':
                connection_violations['c_n_ca_loss_mean'],  # ()
            'connections_per_residue_loss_sum':
                connection_violations['per_residue_loss_sum'],  # (N)
            'connections_per_residue_violation_mask':
                connection_violations['per_residue_violation_mask'],  # (N)
            'clashes_mean_loss':
                between_residue_clashes['mean_loss'],  # ()
            'clashes_per_atom_loss_sum':
                between_residue_clashes['per_atom_loss_sum'],  # (N, 14)
            'clashes_per_atom_clash_mask':
                between_residue_clashes['per_atom_clash_mask'],  # (N, 14)
        },
        'within_residues': {
            'per_atom_loss_sum':
                within_residue_violations['per_atom_loss_sum'],  # (N, 14)
            'per_atom_violations':
                within_residue_violations['per_atom_violations'],  # (N, 14),
        },
        'total_per_residue_violations_mask':
            per_residue_violations_mask,  # (N)
    }


def find_structural_violations(
    batch,
    atom14_pred_positions,  # (N, 14, 3)
    config
    ):
    """Computes several checks for structural violations."""
    atom14_pred_positions= atom14_pred_positions.float()
    L, atoms_num = atom14_pred_positions.shape[:2]
    # import pdb; pdb.set_trace()
    pred_atom_mask = batch['seq_mask'][:, None].repeat(1, atoms_num)
    # Compute between residue backbone violations of bonds and angles.
    pseudo_aatype = torch.zeros((L,), dtype=torch.long, device=atom14_pred_positions.device)
    # import pdb; pdb.set_trace()
    connection_violations = all_atom.between_residue_bond_loss(
        pred_atom_positions=atom14_pred_positions,
        pred_atom_mask=pred_atom_mask.float(), #.astype(jnp.float32),
        residue_index=batch['single_res_rel'].float(), #.astype(jnp.float32),
        aatype=pseudo_aatype,
        tolerance_factor_soft=config.violation_tolerance_factor,
        tolerance_factor_hard=config.violation_tolerance_factor)

    # Compute the Van der Waals radius for every atom
    # (the first letter of the atom name is the element type).
    # Shape: (N, 14).
    atomtype_radius = torch.FloatTensor([
        residue_constants.van_der_waals_radius[name[0]]
        for name in residue_constants.atom_types
    ]).to(atom14_pred_positions.device)
    num_res = L
    atom14_atom_radius = pred_atom_mask * atomtype_radius.unsqueeze(0).repeat(num_res, 1)[:, :atoms_num]
    # atom14_atom_radius = batch['seq_mask'] * torch.gather(
    #     atomtype_radius.unsqueeze(0).repeat(num_res, 1), 1, batch['residx_atom14_to_atom37'])

    # Compute the between residue clash loss.
    between_residue_clashes = all_atom.between_residue_clash_loss(
        atom14_pred_positions=atom14_pred_positions,
        atom14_atom_exists=pred_atom_mask,
        atom14_atom_radius=atom14_atom_radius,
        residue_index=batch['single_res_rel'],
        overlap_tolerance_soft=config.clash_overlap_tolerance,
        overlap_tolerance_hard=config.clash_overlap_tolerance,
        natoms=atoms_num)

    # Compute all within-residue violations (clashes,
    # bond length and angle violations).
    restype_atom14_bounds = residue_constants.make_atom14_dists_bounds(
        overlap_tolerance=config.clash_overlap_tolerance,
        bond_length_tolerance_factor=config.violation_tolerance_factor)
    restype_atom14_bounds = {k: v[:, :atoms_num, :atoms_num] for k, v in restype_atom14_bounds.items()}
    
    atom14_dists_lower_bound = torch.gather(
        restype_atom14_bounds['lower_bound'].to(pseudo_aatype.device), 0, pseudo_aatype[..., None, None].repeat(1, atoms_num, atoms_num))
    atom14_dists_upper_bound = torch.gather(
        restype_atom14_bounds['upper_bound'].to(pseudo_aatype.device), 0, pseudo_aatype[..., None, None].repeat(1, atoms_num, atoms_num))
    within_residue_violations = all_atom.within_residue_violations(
        atom14_pred_positions=atom14_pred_positions,
        atom14_atom_exists=pred_atom_mask,
        atom14_dists_lower_bound=atom14_dists_lower_bound,
        atom14_dists_upper_bound=atom14_dists_upper_bound,
        tighten_bounds_for_loss=0.0,
        natoms=atoms_num)
    # Combine them to a single per-residue violation mask (used later for LDDT).
    per_residue_violations_mask = torch.max(torch.stack([
        connection_violations['per_residue_violation_mask'],
        torch.max(between_residue_clashes['per_atom_clash_mask'], dim=-1)[0],
        torch.max(within_residue_violations['per_atom_violations'],
                dim=-1)[0]]), dim=0)[0]

    return {
        'between_residues': {
            'bonds_c_n_loss_mean':
                connection_violations['c_n_loss_mean'],  # ()
            'angles_ca_c_n_loss_mean':
                connection_violations['ca_c_n_loss_mean'],  # ()
            'angles_c_n_ca_loss_mean':
                connection_violations['c_n_ca_loss_mean'],  # ()
            'connections_per_residue_loss_sum':
                connection_violations['per_residue_loss_sum'],  # (N)
            'connections_per_residue_violation_mask':
                connection_violations['per_residue_violation_mask'],  # (N)
            'clashes_mean_loss':
                between_residue_clashes['mean_loss'],  # ()
            'clashes_per_atom_loss_sum':
                between_residue_clashes['per_atom_loss_sum'],  # (N, 14)
            'clashes_per_atom_clash_mask':
                between_residue_clashes['per_atom_clash_mask'],  # (N, 14)
        },
        'within_residues': {
            'per_atom_loss_sum':
                within_residue_violations['per_atom_loss_sum'],  # (N, 14)
            'per_atom_violations':
                within_residue_violations['per_atom_violations'],  # (N, 14),
        },
        'total_per_residue_violations_mask':
            per_residue_violations_mask,  # (N)
    }



def find_structural_violations_group(
    batch,
    atom14_positions,
    config
):
    assert len(atom14_positions.shape) ==4, f"atom14_pred_positions shape {atom14_positions.shape}"
    traj_batch_size = atom14_positions.shape[0]
    batch_size = batch['seq_mask'].shape[0]
    traj_num = traj_batch_size//batch_size
    # import pdb; pdb.set_trace()
    if batch.__contains__('masked_FG_seq'):
        seq_mask = 1 - batch['masked_FG_seq'].repeat(traj_num, 1)
        seq_mask = seq_mask * batch['seq_mask'].repeat(traj_num, 1)
        tmp_batch = {'seq_mask': seq_mask,
                     'single_res_rel': batch['single_res_rel'].repeat(traj_num, 1)}
    else:
        tmp_batch = {'seq_mask': batch['seq_mask'].repeat(traj_num, 1),
                    'single_res_rel': batch['single_res_rel'].repeat(traj_num, 1)}
    # import pdb; pdb.set_trace()
    # vio_list= []
    # for i in range(traj_batch_size):
    #     a14_pos_curr= atom14_positions[i]
    #     curr_batch = {'seq_mask': tmp_batch['seq_mask'][i],
    #                 'single_res_rel': tmp_batch['single_res_rel'][i]}
    #     vio_curr= find_structural_violations(curr_batch, a14_pos_curr, config)
    #     vio_list.append(vio_curr)
    # violations={}
        
    # for k1 in ('between_residues','within_residues'):
    #     violations[k1]={}
    #     for k2 in vio_list[0][k1].keys():
    #         if len(vio_list[0][k1][k2].shape) ==0:
    #             violations[k1][k2]= sum(v[k1][k2] for v in vio_list)/ len(vio_list)
    #         else:
    #             violations[k1][k2] = torch.stack([v[k1][k2] for v in vio_list],dim=0)
    # violations['total_per_residue_violations_mask']= torch.stack([v['total_per_residue_violations_mask'] for v in vio_list], dim=0)
    violations = find_structural_violations_batch(tmp_batch, atom14_positions, config)
    return violations


def structural_violation_loss(batch, atom14_positions, config):
  """Computes loss for structural violations."""
  violations = find_structural_violations_group(batch, atom14_positions, config)
#   import pdb; pdb.set_trace()
  num_atoms = np.prod(list(atom14_positions.shape[:-1])).astype(np.float32)
  violation_loss = (
      violations['between_residues']['bonds_c_n_loss_mean'] +
      violations['between_residues']['angles_ca_c_n_loss_mean'] +
      violations['between_residues']['angles_c_n_ca_loss_mean'] +
      torch.sum(violations['between_residues']['clashes_per_atom_loss_sum'] +
                violations['within_residues']['per_atom_loss_sum']) /
                (1e-6 + num_atoms))

  return violation_loss