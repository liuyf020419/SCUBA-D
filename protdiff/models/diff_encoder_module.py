import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .folding_af2 import layers_batch
from .folding_af2.ipa_2d_net import EvoformerPairBlock_
from .attention.modules import TransformerLayer
from .attention.dense_block import TransformerPositionEncoding


NOISE_SCALE = 5000
class ContinousNoiseSchedual(nn.Module):
    """
    noise.shape (batch_size, )
    """
    def __init__(self, d_model):
        super(ContinousNoiseSchedual, self).__init__()

        half_dim = d_model // 2
        emb = math.log(10000) / float(half_dim - 1)
        emb = torch.exp(torch.arange(half_dim) * -emb)
        # emb.shape (half_dim, )
        self.register_buffer("emb", emb, persistent=True)

    def forward(self, noise):
        """
        noise [B, 1]
        return [:seqlen, d_model]
        """
        if len(noise.shape) > 1:
            noise = noise.squeeze(-1)
        assert len(noise.shape) == 1

        exponents = NOISE_SCALE * noise[:, None] * self.emb[None, :]
        return torch.cat([exponents.sin(), exponents.cos()], dim=-1)


class DiffSingleEncoder(nn.Module):
    def __init__(self, config, global_config) -> None:
        super().__init__()
        self.config = config
        self.global_config = global_config
        self.single_channel = config.single_channel
        self.condition_embedding = config.condition_embedding

        self.single_res_pos = TransformerPositionEncoding(config.max_enlarge_seq_len, config.enlarge_seq_channel)
        self.position_act = layers_batch.Linear(config.enlarge_seq_channel, config.single_channel)

        self.esm_act = nn.Sequential(
            layers_batch.Linear(global_config.esm_num, config.single_channel, initializer='relu'),
            nn.ReLU(),
            layers_batch.Linear(config.single_channel, config.single_channel))

        self.single_act = layers_batch.Linear(config.single_channel, config.single_channel)

        self.t_emb = ContinousNoiseSchedual(config.single_channel)

        if self.condition_embedding:
            self.condition_emb = nn.Embedding(2, config.single_channel)

        if self.config.extra_attention:
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

    def forward(self, batch, esm):
        mask = batch['seq_mask']
        pos_emb = self.single_res_pos(batch['single_res_rel'], index_select=True)
        position_act = self.position_act(pos_emb)

        esm_act = self.esm_act(esm)

        t_emb = self.t_emb(batch['t'])

        x = self.single_act(position_act + esm_act) + t_emb[:, None, :]

        if self.condition_embedding:
            condition = batch['condition']
            # import pdb; pdb.set_trace()
            x = x + self.condition_emb(condition.long())

        if self.config.extra_attention:
            padding_mask = 1.0 - mask

            if not padding_mask.any():
                padding_mask = None

            for layer in self.attention:
                x = x.transpose(0, 1)
                x, attn = layer(x, self_attn_padding_mask=padding_mask)
                x = x.transpose(0, 1)
            x = x * mask[..., None]

        return x



class DiffPairEncoder(nn.Module):
    def __init__(self, config, global_config) -> None:
        super().__init__()
        self.config = config
        self.global_config = global_config
        self.pair_channel = config.pair_channel
        self.condition_embedding = config.condition_embedding

        self.pair_res_rel = nn.Embedding(config.pair_res_rel * 2 + 1 + 1, config.pair_res_channel)
        self.pair_chain_rel = nn.Embedding(config.pair_chain_rel * 2 + 1 + 1, config.pair_chain_channel)

        input_channel = 4
        if self.config.rbf_encode:
            input_channel += self.config.num_rbf - 1
        if self.config.tri_encode:
            input_channel += self.config.tri_num * 2 * 3 - 3
        # self.pair_FG_encoder = nn.Linear(input_channel, config.pair_FG_channel)
        self.pair_FG_encoder = nn.Sequential(
            layers_batch.Linear(input_channel, config.pair_FG_channel, initializer='relu'),
            nn.ReLU(),
            layers_batch.Linear(config.pair_FG_channel, config.pair_FG_channel))

        pair_input_channel = config.pair_res_channel + config.pair_chain_channel
        pair_input_channel = pair_input_channel + config.pair_FG_channel

        self.t_emb = ContinousNoiseSchedual(config.pair_channel)

        if self.condition_embedding:
            self.condition_emb = nn.Embedding(2, config.pair_channel)

        self.pair_act = layers_batch.Linear(pair_input_channel, config.pair_channel)


    def forward(self, batch, geom_pair):
        pair_chain_pos = self.pair_chain_rel(batch['pair_chain_rel'])
        pair_res_pos = self.pair_res_rel(batch['pair_res_rel'])
        pair_pos = torch.cat([pair_res_pos, pair_chain_pos], -1)

        pair_geom_act = self.pair_FG_encoder(geom_pair)
        pair = torch.cat([pair_pos, pair_geom_act], -1)

        t_emb = self.t_emb(batch['t'])

        pair_act = self.pair_act(pair) + t_emb[:, None, None, :]

        if self.condition_embedding:
            condition = batch['condition']
            pair_condition = condition[:, :, None] * condition[:, None]
            pair_act = pair_act + self.condition_emb(pair_condition.long())

        return pair_act
