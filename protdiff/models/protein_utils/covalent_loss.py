import torch
import numpy as np

from ..folding_af2 import all_atom
from ..folding_af2 import residue_constants
from ..folding_af2 import utils



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