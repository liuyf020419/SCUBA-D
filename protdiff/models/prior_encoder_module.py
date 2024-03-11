import torch
import torch.nn as nn
import torch.nn.functional as F

from .folding_af2 import layers_batch
from .attention.modules import TransformerLayer
from .attention.dense_block import TransformerPositionEncoding

import numpy as np


class SingleEncoder(nn.Module):
    def __init__(self, config, encode_aa=False, encode_ss=False):
        super().__init__()

        self.config = config
        self.encode_aa = encode_aa
        self.encode_ss = encode_ss

        if self.encode_aa:
            self.aatype_embedding = nn.Embedding(22, config.aa_seq_channel)
        if self.encode_ss:
            self.sstype_embedding = nn.Embedding(5, config.ss_seq_channel)

        self.enlarge_gap = self.config.enlarge_gap

        if self.enlarge_gap:
            input_emb_channel = config.enlarge_seq_channel
            self.single_res_rel = TransformerPositionEncoding(config.max_enlarge_seq_len, d_model=config.enlarge_seq_channel)
            if self.encode_aa:
                input_emb_channel = input_emb_channel + config.aa_seq_channel
            if self.encode_ss:
                input_emb_channel = input_emb_channel + config.ss_seq_channel

            self.input_emb = layers_batch.Linear(input_emb_channel, config.single_channel)
       
        else:
            self.all_seq_encode = TransformerPositionEncoding(config.max_all_seq_len, d_model=config.all_seq_channel)
            self.all_seq_emb = layers_batch.Linear(config.all_seq_channel, config.all_seq_channel)

            self.part_seq_encode = TransformerPositionEncoding(config.max_part_seq_len, d_model=config.part_seq_channel)
            self.part_seq_emb = layers_batch.Linear(config.part_seq_channel, config.part_seq_channel)

            self.chain_abs_encode = TransformerPositionEncoding(config.max_chain_seq_len, d_model=config.chain_seq_channel)
            self.chain_abs_emb = layers_batch.Linear(config.chain_seq_channel, config.chain_seq_channel)

            self.input_emb = layers_batch.Linear(config.all_seq_channel + config.part_seq_channel + config.chain_seq_channel if not self.config.ss_emb \
                else config.all_seq_channel + config.part_seq_channel + config.chain_seq_channel + config.ss_seq_channel, config.single_channel)

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
        
    def forward(self, single, mask, aatype=None, ca_pos=None, mask_FG_seq=None, random_mask=False):
        self.random_mask = random_mask
        if self.enlarge_gap:
            single_emb = self.single_res_rel(single['single_res_rel'], index_select=True)
        else:
            # B, L, H
            all_seq_encode = self.all_seq_encode(single['single_all_res_rel'], index_select=True)
            all_seq_emb = self.all_seq_emb(all_seq_encode)

            part_seq_encode = self.part_seq_encode(single['single_part_res_rel'], index_select=True)
            part_seq_emb = self.part_seq_emb(part_seq_encode)

            chain_abs_encode = self.chain_abs_encode(single['single_all_chain_rel'], index_select=True)
            chain_abs_emb = self.chain_abs_emb(chain_abs_encode)

            # part_seq_emb = F.one_hot(single['single_part_res_rel'], num_classes=2 * self.config.max_single_part_res_rel + 1)
            # single_emb = self.part_seq_emb(part_seq_emb)

            single_emb = torch.cat([all_seq_emb, part_seq_emb, chain_abs_emb], -1)


        if self.encode_aa:
            assert aatype is not None
            if self.random_mask:
                # import pdb; pdb.set_trace()
                batchsize, L = single['single_res_rel'].shape
                if self.training:
                    mask_modes = np.random.randint(0, 4, 1)
                else:
                    mask_modes = [2]
                aa_mask = gen_random_mask(self.config, batchsize, L, mask_modes[0], ca_pos).to(aatype)
                aatype_masked = torch.where(aa_mask == 0, 21, aatype) # random mask
                aatype_masked = torch.where(mask == 0, 21, aatype_masked) # padding mask
            else:
                assert mask_FG_seq is not None
                aa_mask = mask_FG_seq
                aatype_masked = torch.where(mask_FG_seq == 0, 21, aatype) # random mask
                aatype_masked = torch.where(mask == 0, 21, aatype_masked) # padding mask

            aa_emb = self.aatype_embedding(aatype_masked)
            single_emb = torch.cat([single_emb, aa_emb], -1)

        if self.encode_ss:
            sstype = single['single_ssedges']
            sstype_masked = torch.where(mask == 0, 4, sstype)
            ss_emb = self.sstype_embedding(sstype_masked)
            single_emb = torch.cat([single_emb, ss_emb], -1)

        # import pdb;pdb.set_trace()
        x = self.input_emb(single_emb)

        padding_mask = 1.0 - mask

        if not padding_mask.any():
            padding_mask = None

        for layer in self.attention:
            x = x.transpose(0, 1)
            # import pdb; pdb.set_trace()
            x, attn = layer(x, self_attn_padding_mask=padding_mask)
            x = x.transpose(0, 1)
        x = x * mask[..., None]
        
        if self.encode_aa:
            return x, aa_mask, aatype_masked
        else:
            return x


class PairEncoder(nn.Module):
    def __init__(self, config, global_config, encode_FG=False):
        super().__init__()
        self.config = config
        self.global_config = global_config
        self.encode_FG = encode_FG

        self.pair_res_rel = nn.Embedding(config.pair_res_rel * 2 + 1 + 1, config.pair_res_channel)
        self.pair_chain_rel = nn.Embedding(config.pair_chain_rel * 2 + 1 + 1, config.pair_chain_channel)

        if self.encode_FG:
            input_channel = 4
            if self.config.rbf_encode:
                input_channel += self.config.num_rbf - 1
            if self.config.tri_encode:
                input_channel += self.config.tri_num * 2 * 3 - 3
            self.pair_FG_encoder = nn.Linear(input_channel, config.pair_FG_channel)

        pair_input_channel = config.pair_res_channel + config.pair_chain_channel

        if self.encode_FG:
            pair_input_channel = pair_input_channel + config.pair_FG_channel

        if self.config.condition_embedding:
            self.condition_emb = nn.Embedding(2, pair_input_channel)

        self.pair_input = layers_batch.Linear(pair_input_channel, config.pair_channel)

        # if self.config.pre_act_pair_layer:
        #     self.pre_activate_pair_mask = EvoformerPairBlock_(config, global_config, config.pair_channel, config.pair_channel)
        #     self.pre_activate_pair_denoising = EvoformerPairBlock_(config, global_config, config.pair_channel, config.pair_channel)


    def forward(self, pair, pair_mask, geom_pair=None, condition=None):
        # import pdb; pdb.set_trace()
        pair_chain_rel = self.pair_chain_rel(pair['pair_chain_rel'])
        pair_res_rel = self.pair_res_rel(pair['pair_res_rel'])
        pair_act = torch.cat([pair_res_rel, pair_chain_rel], -1)

        if self.encode_FG:
            if geom_pair is None:
                pair_FG_act = self.pair_FG_encoder(pair['masked_pair_map'])
            else:
                pair_FG_act = self.pair_FG_encoder(geom_pair)
            pair_act = torch.cat([pair_act, pair_FG_act], -1)
        
        x = self.pair_input(pair_act)

        if self.config.condition_embedding:
            if condition is None:
                B, L = pair_act.shape[:2]
                condition = torch.zeros(B, L).long().to(pair_act.device)
            else:
                condition = condition.long().to(pair_act.device)
            pair_condition = condition[:, :, None] * condition[:, None]
            condition_emb = self.condition_emb(pair_condition)
            x = x + condition_emb
            
        return x



def gen_random_mask(config, batchsize, seq_len, mask_mode, ca_pos):
    p_rand = config.p_rand
    p_lin = config.p_lin
    p_spatial = config.p_spatial

    min_lin_len = int(p_lin[0] * seq_len) # 0.25
    max_lin_len = int(p_lin[1] * seq_len) # 0.75
    lin_len = torch.randint(min_lin_len, max_lin_len, [1]).item()

    min_knn = p_spatial[0] # 0.1
    max_knn = p_spatial[1] # 0.5
    knn = int((torch.rand([1]) * (max_knn-min_knn) + min_knn).item() * seq_len)

    if mask_mode == 0: # random
        mask = (torch.rand(batchsize, seq_len) > p_rand).long()

    elif mask_mode == 1: # linear
        start_index = torch.randint(0, seq_len-lin_len, [batchsize])
        mask = torch.ones(batchsize, seq_len)
        mask_idx = start_index[:, None] + torch.arange(lin_len)
        mask.scatter_(1, mask_idx, torch.zeros_like(mask_idx).float())

    elif mask_mode == 2: # full
        mask = torch.zeros(batchsize, seq_len)

    elif mask_mode == 3: # spatial
        central_absidx = torch.randint(0, seq_len, [batchsize])
        ca_map = torch.sqrt(torch.sum(torch.square(ca_pos[:, None] - ca_pos[:, :, None]), -1) + 1e-10)
        batch_central_knnid = torch.stack([ca_map[bid, central_absidx[bid]] for bid in range(batchsize)])
        knn_idx = torch.argsort(batch_central_knnid)[:, :knn]
        mask = torch.ones(batchsize, seq_len).to(ca_map.device)
        mask.scatter_(1, knn_idx, torch.zeros_like(knn_idx).float())

    return mask
