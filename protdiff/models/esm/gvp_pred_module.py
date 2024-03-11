import torch
import torch.nn as nn
import torch.nn.functional as F

from .gvp_encoder import GVPTransformerEncoder


class GVPSinglePred(nn.Module):
    def __init__(self, config, out_num) -> None:
        super().__init__()
        self.config = config

        self.gvp_encoder = GVPTransformerEncoder(self.config)
        self.out_projection = nn.Linear(self.gvp_encoder.embed_dim, out_num)


    def forward(self, batch, coord_dict):
        ## pred data
        pred_coord_dict = {
            'coord': coord_dict['coord'][-1],
            'rot': coord_dict['rot'][-1]
        }

        pred_feature = self.process(pred_coord_dict, batch['seq_mask'], batch['single_res_rel'])

        return pred_feature


    def process(self, coord_dict: dict, seq_mask_traj, res_idx):
        batchsize, L = coord_dict['coord'].shape[:2]
        coord = coord_dict['coord'][..., :3, :]
        if not coord_dict.__contains__('rot'):
            new_shape = list(coord_dict['backbone_frame'].shape[:-2]) + [3, 3]
            rot = coord_dict['backbone_frame'][..., 0, :9].reshape(new_shape)
            rot = rot.reshape(batchsize, L, 3, 3)
        else:
            rot = coord_dict['rot']
        pseudo_aatype = torch.zeros(batchsize, L).long().to(coord.device)
        data_dict = {'coord': coord, 'encoder_padding_mask': seq_mask_traj.bool(), 
                    'confidence': torch.ones(batchsize, L).to(coord.device), 'rot': rot, 
                    'res_idx': res_idx,
                    'aatype': pseudo_aatype}

        encoder_out = self.gvp_encoder(data=data_dict, return_all_hiddens=True)
        encoder_feature = encoder_out['encoder_out'][0].permute(1, 0, 2) # batchsize, L, C
        single_out = self.out_projection(encoder_feature)
        
        return single_out
