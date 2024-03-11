import torch
import torch.nn as nn
import torch.nn.functional as F

from .esm.gvp_encoder import GVPTransformerEncoder
from .attention.modules import TransformerLayer
from .attention.dense_block import TransformerPositionEncoding

from .protein_geom_utils import get_internal_angles
from .local_env_utils.struct2seq import StructureTransformer, MPNNEncoder


class GVPDiscriminator(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

        self.gvp_encoder = GVPTransformerEncoder(self.config)
        self.out_projection = nn.Linear(self.gvp_encoder.embed_dim, 1)


    def forward(self, batch, pred_dict, detach_all=False):
        ## true data
        true_coord_dict = {
            'coord': batch['gt_pos'],
            'backbone_frame': batch['gt_backbone_frame']
        }
        true_feature, true_logits = self.process(true_coord_dict, batch['seq_mask'], batch['single_res_rel'])
        ## pred data
        pred_coord_dict = {
            'coord': pred_dict['coord'][-1],
            'rot': pred_dict['rot'][-1]
        }
        if detach_all:
            pred_coord_dict = {k:v.detach() for k, v in pred_coord_dict.items()}
        pred_feature, pred_logits = self.process(pred_coord_dict, batch['seq_mask'], batch['single_res_rel'])

        return true_logits, pred_logits, true_feature, pred_feature


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

        encoder_out_dict = self.gvp_encoder(data=data_dict, return_all_hiddens=True)

        encoder_feature = torch.stack(encoder_out_dict['encoder_states']).permute(2, 1, 0, 3) # batchsize, L, N, C
        encoder_out = encoder_out_dict['encoder_out'][0].permute(1, 0, 2) # batchsize, L, C
        single_flatten = self.out_projection(encoder_out)[..., 0] # batchsize, L
        logits = F.adaptive_avg_pool1d(single_flatten, 1)[..., 0] # batchsize,
        
        return encoder_feature, logits


class TransformerDiscriminator(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

        self.input_projection = nn.Linear(4, config.single_channel)
        self.pos_embdder = TransformerPositionEncoding(config.max_seq_len, config.single_channel)
        self.attention = nn.ModuleList(
            [
                TransformerLayer(
                    config.single_channel,
                    config.ffn_embed_dim,
                    config.attention_heads,
                    dropout = getattr(config, 'dropout', 0.0),
                    add_bias_kv=True,
                    use_esm1b_layer_norm=False,
                )
                for _ in range(config.layers)
            ]
        )
        self.out_projection = nn.Linear(config.single_channel, 1)


    def forward(self, batch, pred_dict, detach_all=False):
        gt_IC = get_internal_angles(batch['gt_pos'])
        pred_IC = get_internal_angles(pred_dict['coord'][-1])

        true_feature, true_logits = self.process(gt_IC, batch['seq_mask'], batch['single_res_rel'])

        if detach_all:
            pred_IC = pred_IC.detach()
        pred_feature, pred_logits = self.process(pred_IC, batch['seq_mask'], batch['single_res_rel'])

        return true_logits, pred_logits, true_feature, pred_feature
        

    def process(self, internal_angles, single_mask, res_pos):
        attention_feature = []
        padding_mask = 1.0 -single_mask

        if not padding_mask.any():
            padding_mask = None

        triangle_encode_angles = torch.cat([
            torch.sin(internal_angles), torch.cos(internal_angles)], -1)
        x = self.input_projection(triangle_encode_angles)

        pos_emb = self.pos_embdder(res_pos, index_select=True)
        x = x + pos_emb

        for layer in self.attention:
            x = x.transpose(0, 1)
            x, attn = layer(x, self_attn_padding_mask=padding_mask)
            x = x.transpose(0, 1)
            attention_feature.append(x)
        x = x * single_mask[..., None]

        single_flatten = self.out_projection(x)[..., 0] # batchsize, L
        logits = F.adaptive_avg_pool1d(single_flatten, 1)[..., 0] # batchsize,
        attention_feature = torch.stack(attention_feature).permute(1, 0, 2, 3)
        # import pdb; pdb.set_trace()

        return attention_feature, logits


class LocalEnvironmentTransformer(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

        self.structuretransformer = MPNNEncoder(
            node_features=config.node_features, 
            edge_features=config.edge_features, 
            hidden_dim=config.hidden_dim, 
            num_encoder_layers=config.num_encoder_layers, 
            k_neighbors=config.k_neighbors)

        self.LE_out_projection = nn.Linear(config.hidden_dim * 2, 1)
        self.LE_edge_aggregation = nn.Linear(20, 1)


    def forward(self, batch, pred_dict, detach_all=False):
        # import pdb; pdb.set_trace()
        ## true data
        true_feature, true_logits = self.process(batch['gt_pos'], batch['seq_mask'], batch['single_res_rel'])
        ## pred data
        if detach_all:
            pred_coord = pred_dict['coord'][-1].detach()
        else:
            pred_coord = pred_dict['coord'][-1]
        pred_feature, pred_logits = self.process(pred_coord, batch['seq_mask'], batch['single_res_rel'])
        # import pdb; pdb.set_trace()
        return true_logits, pred_logits, true_feature, pred_feature


    def process(self, coords, seq_mask, single_res_rel):
        batchsize , L = seq_mask.shape
        if len(coords.shape) != 4:
            coords = torch.reshape(coords, (batchsize, L, 3, 3))
        # pseudo_aatype = torch.zeros((batchsize, L)).long().to(seq_mask.device)
        # feature_dict = self.structuretransformer(coords, pseudo_aatype, L, seq_mask, single_res_rel)
        feature_dict = self.structuretransformer(coords, L, seq_mask, single_res_rel)
        # import pdb; pdb.set_trace()
        single_flatten = self.LE_out_projection(feature_dict['out_feature'])[..., 0] # batchsize, L, 20
        single_aggregate = self.LE_edge_aggregation(single_flatten)[..., 0] # batchsize, L
        logits = F.adaptive_avg_pool1d(single_aggregate, 1)[..., 0] # batchsize, 
        attention_feature = feature_dict['stacked_hidden']

        return attention_feature, logits