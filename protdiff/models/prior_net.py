import logging

import torch
import torch.nn as nn

from .protein_utils.backbone import backbone_frame_to_atom3_std, backbone_fape_loss, structural_violation_loss
from .protein_utils.add_o_atoms import add_atom_O

from .prior_pairformer import PriorPairNet

from .protein_utils.write_pdb import write_multichain_from_atoms

logger = logging.getLogger(__name__)


class PriorModule(nn.Module):
    def __init__(self, config, global_config, data_config) -> None:
        super().__init__()
        self.config = config
        self.global_config = global_config
        self.data_config = data_config
        self.prior_module = PriorPairNet(self.config, self.global_config, self.data_config)

    def forward(self, batch):
        prior_dict, losses = self.prior_module(batch)

        return prior_dict, losses

    @torch.no_grad()
    def sampling(self, batch, pdb_prefix, noising_mode_idx, condition=None, epoch_idx=0, return_traj=True):
        batch_size = batch['traj_pos'].shape[0]
        if (batch.__contains__('gt_pos') and (epoch_idx == 0)):
            gt_coord4 = add_atom_O(batch['gt_pos'].detach().cpu().numpy()[0, :, :3, :])
            write_multichain_from_atoms([gt_coord4.reshape(-1, 3)], f'{pdb_prefix}_gt.pdb')
            
        if return_traj:
            for batch_idx in range(batch_size):
                gt_coord4 = add_atom_O(batch['traj_pos'].detach().cpu().numpy()[batch_idx, :, :3, :])
                write_multichain_from_atoms([gt_coord4.reshape(-1, 3)], f'{pdb_prefix}_input_term_{epoch_idx}_batch_{batch_idx}.pdb')

        pred_dict = self.prior_module.sampling(batch, noising_mode_idx, condition)

        if return_traj:
            for batch_idx in range(batch_size):
                pred_coord4 = add_atom_O(pred_dict['coord'].detach().cpu().numpy()[batch_idx])
                write_multichain_from_atoms([pred_coord4.reshape(-1, 3)], f'{pdb_prefix}_prior_traj_term_{epoch_idx}_batch_{batch_idx}.pdb')
                if batch.__contains__('traj_pos'):
                    gt_coord4 = add_atom_O(batch['traj_pos'].detach().cpu().numpy()[batch_idx, :, :3, :])
                    write_multichain_from_atoms([gt_coord4.reshape(-1, 3)], f'{pdb_prefix}_init_traj_term_{epoch_idx}_batch_{batch_idx}.pdb')
                
        return pred_dict


