import torch
import torch.nn as nn
import torch.nn.functional as F

from .esm_projection import load_esm1b_projection_head, esm1b_aatype_index_range, af2_index_to_esm1b_from0_index


class EsmProjectionHead(nn.Module):
    def __init__(self, data_config) -> None:
        super().__init__()
        esm1b_LMhead_ckpt = data_config.esm1b_LMhead_ckpt
        esm1b_stat_meanstd_file = data_config.esm1b_stat_meanstd_file
        self.esm_projection_module = load_esm1b_projection_head(esm1b_LMhead_ckpt, esm1b_stat_meanstd_file)

    def forward(self, batch, esm_single):
        gt_aatype_af2_idx = batch['aatype']
        device = gt_aatype_af2_idx.device
        batch_size, res_num = gt_aatype_af2_idx.shape

        gt_aatype_esm1b_idx = torch.stack([
            torch.LongTensor([af2_index_to_esm1b_from0_index[af2_idx.item()] 
                for af2_idx in b_af2_idx]) 
            for b_af2_idx in gt_aatype_af2_idx]).long().to(device)

        pred_logits = self.esm_projection_module(esm_single)[..., esm1b_aatype_index_range[0]:esm1b_aatype_index_range[1]]

        loss_esm_ce = F.cross_entropy(
            pred_logits.reshape(-1, esm1b_aatype_index_range[1] - esm1b_aatype_index_range[0]), 
            gt_aatype_esm1b_idx.reshape(-1), reduction='none').reshape(batch_size, res_num)
        
        loss_esm_ce = loss_esm_ce * batch['seq_mask']
        loss_esm_ce_reduce = torch.sum(loss_esm_ce) / (torch.sum(batch['seq_mask']) + 1e-6)

        return loss_esm_ce_reduce
