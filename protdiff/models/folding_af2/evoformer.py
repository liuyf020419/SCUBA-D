import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .layers import *
from .template import *
from . import all_atom


class EvoformerNoMSABlock(nn.Module):
    def __init__(self, config, global_config, single_channel, pair_channel):
        super().__init__()
        self.config = config
        self.global_config = global_config
        # self.is_extra_msa = is_extra_msa

        self.dropout_factor = 0.0 if global_config.deterministic else 1.0

        self.outer_product = OuterProduct(
            config.outer_product_mean, global_config, single_channel, pair_channel
        )
        self.msa_column_attention = MSAColumnAttention(
            config.msa_column_attention, global_config, single_channel,
        )
        self.msa_transition = Transition(
            config.msa_transition, global_config, single_channel
        )
        self.triangle_multiplication_outgoing = TriangleMultiplication(
            config.triangle_multiplication_outgoing, global_config, pair_channel
        )
        self.triangle_multiplication_incoming = TriangleMultiplication(
            config.triangle_multiplication_incoming, global_config, pair_channel
        )
        self.triangle_attention_starting_node = TriangleAttention(
            config.triangle_attention_starting_node, global_config, pair_channel
        )
        self.triangle_attention_ending_node = TriangleAttention(
            config.triangle_attention_ending_node, global_config, pair_channel
        )
        self.pair_transition = Transition(
            config.pair_transition, global_config, pair_channel
        )

    def msa_block(self, msa_act, pair_act):
        c = self.config
        gc = self.global_config

        # msa_act, pair_act = activations['msa'], activations['pair']
        # msa_mask, pair_mask = masks['msa'], masks['pair']
        msa_mask = torch.ones_like(msa_act[0])
        pair_mask = torch.ones_like(pair_act)

        # if c.outer_product_mean.first:
        #     residual = self.outer_product(msa_act, msa_mask)
        #     residual = F.dropout(
        #         residual, training = self.training,
        #         p = self.dropout_factor * self.outer_product_mean.config.dropout_rate,
        #     )
        #     pair_act = pair_act + residual

        # residual = self.msa_row_attention_with_pair_bias(msa_act, msa_mask, pair_act)
        # residual = F.dropout(
        #     residual, training = self.training,
        #     p = self.dropout_factor * self.msa_row_attention_with_pair_bias.config.dropout_rate,
        # )
        # msa_act = msa_act + residual
        # if not self.is_extra_msa:
        #     residual = self.msa_column_attention(msa_act, msa_mask)
        #     residual = F.dropout(
        #         residual, training = self.training,
        #         p = self.dropout_factor * self.msa_column_attention.config.dropout_rate,
        #     )
        # else:
        #     residual = self.msa_column_global_attention(msa_act, msa_mask)
        #     residual = F.dropout(
        #         residual, training = self.training,
        #         p = self.dropout_factor * self.msa_column_global_attention.config.dropout_rate,
        #     )
        residual = self.msa_column_attention(msa_act, msa_mask)
        residual = F.dropout(
            residual, training = self.training,
            p = self.dropout_factor * self.msa_column_attention.config.dropout_rate,
        )
        msa_act = msa_act + residual 

        residual = self.msa_transition(msa_act, msa_mask)
        residual = F.dropout(
            residual, training = self.training,
            p = self.dropout_factor * self.msa_transition.config.dropout_rate,
        )
        msa_act = msa_act + residual 

        if not c.outer_product_mean.first:
            residual = self.outer_product(msa_act, msa_mask)
            residual = F.dropout(
                residual, training = self.training,
                p = self.dropout_factor * self.outer_product_mean.config.dropout_rate,
            )
            pair_act = pair_act + residual
        return msa_act, pair_act


    def pair_block(self, pair_act, pair_mask):
        residual = self.triangle_multiplication_outgoing(pair_act, pair_mask)
        residual = F.dropout(
            residual, training = self.training,
            p = self.dropout_factor * self.triangle_multiplication_outgoing.config.dropout_rate,
        )
        pair_act = pair_act + residual 

        residual = self.triangle_multiplication_incoming(pair_act, pair_mask)
        residual = F.dropout(
            residual, training = self.training,
            p = self.dropout_factor * self.triangle_multiplication_incoming.config.dropout_rate,
        )
        pair_act = pair_act + residual 

        residual = self.triangle_attention_starting_node(pair_act, pair_mask)
        residual = F.dropout(
            residual, training = self.training,
            p = self.dropout_factor * self.triangle_attention_starting_node.config.dropout_rate,
        )
        pair_act = pair_act + residual 

        residual = self.triangle_attention_ending_node(pair_act, pair_mask)
        residual = F.dropout(
            residual, training = self.training,
            p = self.dropout_factor * self.triangle_attention_ending_node.config.dropout_rate,
        )
        pair_act = pair_act + residual 

        residual = self.pair_transition(pair_act, pair_mask)
        residual = F.dropout(
            residual, training = self.training,
            p = self.dropout_factor * self.pair_transition.config.dropout_rate,
        )
        pair_act = pair_act + residual 

        return pair_act


    def forward_compute(self, msa_act, pair_act):
        c = self.config
        gc = self.global_config

        msa_mask = torch.ones_like(msa_act[0])
        pair_mask = torch.ones_like(pair_act[:, :, :, 0])

        # msa_act, pair_act = activations['msa'], activations['pair']
        # msa_mask, pair_mask = masks['msa'], masks['pair']

        # if c.outer_product_mean.first:
        #     residual = self.outer_product_mean(msa_act, msa_mask)
        #     residual = F.dropout(
        #         residual, training = self.training,
        #         p = self.dropout_factor * self.outer_product_mean.config.dropout_rate,
        #     )
        #     pair_act = pair_act + residual

        # residual = self.msa_row_attention_with_pair_bias(msa_act, msa_mask, pair_act)
        # residual = F.dropout(
        #     residual, training = self.training,
        #     p = self.dropout_factor * self.msa_row_attention_with_pair_bias.config.dropout_rate,
        # )
        # msa_act = msa_act + residual
        # if not self.is_extra_msa:
        #     residual = self.msa_column_attention(msa_act, msa_mask)
        #     residual = F.dropout(
        #         residual, training = self.training,
        #         p = self.dropout_factor * self.msa_column_attention.config.dropout_rate,
        #     )
        # else:
        #     residual = self.msa_column_global_attention(msa_act, msa_mask)
        #     residual = F.dropout(
        #         residual, training = self.training,
        #         p = self.dropout_factor * self.msa_column_global_attention.config.dropout_rate,
        #     )
        # msa_act = msa_act + residual 
        residual = self.msa_column_attention(msa_act, msa_mask)
        residual = F.dropout(
            residual, training = self.training,
            p = self.dropout_factor * self.msa_column_attention.config.dropout_rate,
        )

        msa_act = msa_act + residual 

        residual = self.msa_transition(msa_act, msa_mask)
        residual = F.dropout(
            residual, training = self.training,
            p = self.dropout_factor * self.msa_transition.config.dropout_rate,
        )
        msa_act = msa_act + residual 
        
        residual = self.outer_product(msa_act)
        residual = F.dropout(
            residual, training = self.training,
            p = self.dropout_factor * self.outer_product.config.dropout_rate,
        )
        pair_act = pair_act + residual
        
        residual = self.triangle_multiplication_outgoing(pair_act, pair_mask)
        residual = F.dropout(
            residual, training = self.training,
            p = self.dropout_factor * self.triangle_multiplication_outgoing.config.dropout_rate,
        )
        pair_act = pair_act + residual 

        residual = self.triangle_multiplication_incoming(pair_act, pair_mask)
        residual = F.dropout(
            residual, training = self.training,
            p = self.dropout_factor * self.triangle_multiplication_incoming.config.dropout_rate,
        )
        pair_act = pair_act + residual 
        import pdb; pdb.set_trace()
        ######################################### TBD #########################################
        residual = self.triangle_attention_starting_node(pair_act, pair_mask)
        residual = F.dropout(
            residual, training = self.training,
            p = self.dropout_factor * self.triangle_attention_starting_node.config.dropout_rate,
        )
        pair_act = pair_act + residual 

        residual = self.triangle_attention_ending_node(pair_act, pair_mask)
        residual = F.dropout(
            residual, training = self.training,
            p = self.dropout_factor * self.triangle_attention_ending_node.config.dropout_rate,
        )
        pair_act = pair_act + residual 

        residual = self.pair_transition(pair_act, pair_mask)
        residual = F.dropout(
            residual, training = self.training,
            p = self.dropout_factor * self.pair_transition.config.dropout_rate,
        )
        pair_act = pair_act + residual 

        return msa_act, pair_act


    def forward(self, msa_act, pair_act):
        if  self.global_config.use_checkpoint:
            msa_act, pair_act = checkpoint(self.forward_compute,  msa_act, pair_act)
        else:
            msa_act, pair_act = self.forward_compute(msa_act, pair_act)
        # if self.global_config.use_checkpoint:
        #     msa_act, pair_act = checkpoint(self.msa_block,  msa_act, pair_act, msa_mask, pair_mask)
        # else:
        #     msa_act, pair_act = self.msa_block(msa_act, pair_act, msa_mask, pair_mask)
        
        # if self.global_config.use_checkpoint:
        #     pair_act = checkpoint(self.pair_block, pair_act, pair_mask)
        # else:
        #     pair_act = self.pair_block(pair_act, pair_mask)
            

        return msa_act, pair_act




class EvoformerBlock(nn.Module):
    def __init__(self, config, global_config, msa_channel, pair_channel, is_extra_msa):
        super().__init__()
        self.config = config
        self.global_config = global_config
        self.is_extra_msa = is_extra_msa

        self.dropout_factor = 0.0 if global_config.deterministic else 1.0
        self.outer_product_mean = OuterProductMean(
            config.outer_product_mean, global_config, msa_channel, pair_channel
        )
        self.msa_row_attention_with_pair_bias = MSARowAttentionWithPairBias(
            config.msa_row_attention_with_pair_bias, global_config, msa_channel, pair_channel
        )
        if not is_extra_msa:
            self.msa_column_attention = MSAColumnAttention(
                config.msa_column_attention, global_config, msa_channel,
            )
        else:
            self.msa_column_global_attention = MSAColumnGlobalAttention(
                config.msa_column_attention, global_config, msa_channel
            )
        self.msa_transition = Transition(
            config.msa_transition, global_config, msa_channel
        )
        self.triangle_multiplication_outgoing = TriangleMultiplication(
            config.triangle_multiplication_outgoing, global_config, pair_channel
        )
        self.triangle_multiplication_incoming = TriangleMultiplication(
            config.triangle_multiplication_incoming, global_config, pair_channel
        )
        self.triangle_attention_starting_node = TriangleAttention(
            config.triangle_attention_starting_node, global_config, pair_channel
        )
        self.triangle_attention_ending_node = TriangleAttention(
            config.triangle_attention_ending_node, global_config, pair_channel
        )
        self.pair_transition = Transition(
            config.pair_transition, global_config, pair_channel
        )

    def msa_block(self, msa_act, pair_act, msa_mask, pair_mask):
        c = self.config
        gc = self.global_config

        # msa_act, pair_act = activations['msa'], activations['pair']
        # msa_mask, pair_mask = masks['msa'], masks['pair']

        if c.outer_product_mean.first:
            residual = self.outer_product_mean(msa_act, msa_mask)
            residual = F.dropout(
                residual, training = self.training,
                p = self.dropout_factor * self.outer_product_mean.config.dropout_rate,
            )
            pair_act = pair_act + residual

        residual = self.msa_row_attention_with_pair_bias(msa_act, msa_mask, pair_act)
        residual = F.dropout(
            residual, training = self.training,
            p = self.dropout_factor * self.msa_row_attention_with_pair_bias.config.dropout_rate,
        )
        msa_act = msa_act + residual
        if not self.is_extra_msa:
            residual = self.msa_column_attention(msa_act, msa_mask)
            residual = F.dropout(
                residual, training = self.training,
                p = self.dropout_factor * self.msa_column_attention.config.dropout_rate,
            )
        else:
            residual = self.msa_column_global_attention(msa_act, msa_mask)
            residual = F.dropout(
                residual, training = self.training,
                p = self.dropout_factor * self.msa_column_global_attention.config.dropout_rate,
            )
        msa_act = msa_act + residual 

        residual = self.msa_transition(msa_act, msa_mask)
        residual = F.dropout(
            residual, training = self.training,
            p = self.dropout_factor * self.msa_transition.config.dropout_rate,
        )
        msa_act = msa_act + residual 

        if not c.outer_product_mean.first:
            residual = self.outer_product_mean(msa_act, msa_mask)
            residual = F.dropout(
                residual, training = self.training,
                p = self.dropout_factor * self.outer_product_mean.config.dropout_rate,
            )
            pair_act = pair_act + residual
        return msa_act, pair_act

    def pair_block(self, pair_act, pair_mask):
        residual = self.triangle_multiplication_outgoing(pair_act, pair_mask)
        residual = F.dropout(
            residual, training = self.training,
            p = self.dropout_factor * self.triangle_multiplication_outgoing.config.dropout_rate,
        )
        pair_act = pair_act + residual 

        residual = self.triangle_multiplication_incoming(pair_act, pair_mask)
        residual = F.dropout(
            residual, training = self.training,
            p = self.dropout_factor * self.triangle_multiplication_incoming.config.dropout_rate,
        )
        pair_act = pair_act + residual 

        residual = self.triangle_attention_starting_node(pair_act, pair_mask)
        residual = F.dropout(
            residual, training = self.training,
            p = self.dropout_factor * self.triangle_attention_starting_node.config.dropout_rate,
        )
        pair_act = pair_act + residual 

        residual = self.triangle_attention_ending_node(pair_act, pair_mask)
        residual = F.dropout(
            residual, training = self.training,
            p = self.dropout_factor * self.triangle_attention_ending_node.config.dropout_rate,
        )
        pair_act = pair_act + residual 

        residual = self.pair_transition(pair_act, pair_mask)
        residual = F.dropout(
            residual, training = self.training,
            p = self.dropout_factor * self.pair_transition.config.dropout_rate,
        )
        pair_act = pair_act + residual 

        return pair_act

    # def forward(self, activations, masks):
    def forward_compute(self, msa_act, pair_act, msa_mask, pair_mask):
        c = self.config
        gc = self.global_config

        # msa_act, pair_act = activations['msa'], activations['pair']
        # msa_mask, pair_mask = masks['msa'], masks['pair']

        if c.outer_product_mean.first:
            residual = self.outer_product_mean(msa_act, msa_mask)
            residual = F.dropout(
                residual, training = self.training,
                p = self.dropout_factor * self.outer_product_mean.config.dropout_rate,
            )
            pair_act = pair_act + residual

        residual = self.msa_row_attention_with_pair_bias(msa_act, msa_mask, pair_act)
        residual = F.dropout(
            residual, training = self.training,
            p = self.dropout_factor * self.msa_row_attention_with_pair_bias.config.dropout_rate,
        )
        msa_act = msa_act + residual
        if not self.is_extra_msa:
            residual = self.msa_column_attention(msa_act, msa_mask)
            residual = F.dropout(
                residual, training = self.training,
                p = self.dropout_factor * self.msa_column_attention.config.dropout_rate,
            )
        else:
            residual = self.msa_column_global_attention(msa_act, msa_mask)
            residual = F.dropout(
                residual, training = self.training,
                p = self.dropout_factor * self.msa_column_global_attention.config.dropout_rate,
            )
        msa_act = msa_act + residual 

        residual = self.msa_transition(msa_act, msa_mask)
        residual = F.dropout(
            residual, training = self.training,
            p = self.dropout_factor * self.msa_transition.config.dropout_rate,
        )
        msa_act = msa_act + residual 

        if not c.outer_product_mean.first:
            residual = self.outer_product_mean(msa_act, msa_mask)
            residual = F.dropout(
                residual, training = self.training,
                p = self.dropout_factor * self.outer_product_mean.config.dropout_rate,
            )
            pair_act = pair_act + residual

        residual = self.triangle_multiplication_outgoing(pair_act, pair_mask)
        residual = F.dropout(
            residual, training = self.training,
            p = self.dropout_factor * self.triangle_multiplication_outgoing.config.dropout_rate,
        )
        pair_act = pair_act + residual 

        residual = self.triangle_multiplication_incoming(pair_act, pair_mask)
        residual = F.dropout(
            residual, training = self.training,
            p = self.dropout_factor * self.triangle_multiplication_incoming.config.dropout_rate,
        )
        pair_act = pair_act + residual 

        residual = self.triangle_attention_starting_node(pair_act, pair_mask)
        residual = F.dropout(
            residual, training = self.training,
            p = self.dropout_factor * self.triangle_attention_starting_node.config.dropout_rate,
        )
        pair_act = pair_act + residual 

        residual = self.triangle_attention_ending_node(pair_act, pair_mask)
        residual = F.dropout(
            residual, training = self.training,
            p = self.dropout_factor * self.triangle_attention_ending_node.config.dropout_rate,
        )
        pair_act = pair_act + residual 

        residual = self.pair_transition(pair_act, pair_mask)
        residual = F.dropout(
            residual, training = self.training,
            p = self.dropout_factor * self.pair_transition.config.dropout_rate,
        )
        pair_act = pair_act + residual 

        return msa_act, pair_act

    def forward(self, msa_act, pair_act, msa_mask, pair_mask):
        if  self.global_config.use_checkpoint:
            msa_act, pair_act = checkpoint(self.forward_compute,  msa_act, pair_act, msa_mask, pair_mask)
        else:
            msa_act, pair_act = self.forward_compute(msa_act, pair_act, msa_mask, pair_mask)
        # if self.global_config.use_checkpoint:
        #     msa_act, pair_act = checkpoint(self.msa_block,  msa_act, pair_act, msa_mask, pair_mask)
        # else:
        #     msa_act, pair_act = self.msa_block(msa_act, pair_act, msa_mask, pair_mask)
        
        # if self.global_config.use_checkpoint:
        #     pair_act = checkpoint(self.pair_block, pair_act, pair_mask)
        # else:
        #     pair_act = self.pair_block(pair_act, pair_mask)
            

        return msa_act, pair_act


class EmbeddingsAndEvoformer(nn.Module):
    def __init__(self, config, global_config):
        super().__init__()
        self.config = config
        self.global_config = global_config

        self.preprocess_1d = Linear(config.target_feat_dim, config.msa_channel)   # 22
        self.preprocess_msa = Linear(config.msa_feat_dim, config.msa_channel)  # 49

        self.left_single = Linear(config.target_feat_dim, config.pair_channel)
        self.right_single = Linear(config.target_feat_dim, config.pair_channel)

        self.prev_pos_linear = Linear(config.prev_pos.num_bins, config.pair_channel)
        self.pair_activiations = Linear(2 * config.max_relative_feature + 1, config.pair_channel)

        self.prev_msa_first_row_norm = nn.LayerNorm(config.msa_channel)
        self.prev_pair_norm = nn.LayerNorm(config.pair_channel)

        self.template_embedding = TemplateEmbedding(config.template, global_config, config.pair_channel)
        self.extra_msa_activations = Linear(config.extra_msa_feat_dim, config.extra_msa_channel)

        self.extra_msa_stack = nn.ModuleList()
        for _ in range(config.extra_msa_stack_num_block):
            self.extra_msa_stack.append(
                # a bug in config?
                EvoformerBlock(
                    config.evoformer, global_config, config.extra_msa_channel, config.pair_channel, is_extra_msa=True
                )
            )
        self.evoformer_iteration = nn.ModuleList()
        for _ in range(config.evoformer_num_block):
            self.evoformer_iteration.append(
                EvoformerBlock(
                    config.evoformer, global_config, config.msa_channel, config.pair_channel, is_extra_msa=False
                )
            )
            
        self.template_single_embedding = Linear(config.template_feat_dim, config.msa_channel, initializer='relu') # 57
        self.template_projection = Linear(config.msa_channel, config.msa_channel, initializer='relu')
        self.single_activations = Linear(config.msa_channel, config.seq_channel)

    def forward(self, batch):
        
        c = self.config
        gc = self.global_config
        
        preprocess_1d = self.preprocess_1d(batch['target_feat'])
        preprocess_msa = self.preprocess_msa(batch['msa_feat'])

        msa_activations = preprocess_1d.unsqueeze(0) + preprocess_msa
        
        left_single = self.left_single(batch['target_feat'])
        right_single = self.right_single(batch['target_feat'])
        pair_activations = left_single[:, None] + right_single[None]
        mask_2d = batch['seq_mask'][...,:, None] * batch['seq_mask'][...,None, :]

       
        # Inject previous outputs for recycling.
        # Jumper et al. (2021) Suppl. Alg. 2 "Inference" line 6
        # Jumper et al. (2021) Suppl. Alg. 32 "RecyclingEmbedder"
        if c.recycle_pos and 'prev_pos' in batch:
            prev_pseudo_beta = pseudo_beta_fn(
                batch['aatype'], batch['prev_pos'], None)
            dgram = dgram_from_positions(prev_pseudo_beta, **self.config.prev_pos)
            prev_pseudo_beta=prev_pseudo_beta.to(pair_activations.dtype)
            dgram= dgram.to(pair_activations.dtype)
            pair_activations = pair_activations + self.prev_pos_linear(dgram)

        
        # recycle_features
        if c.recycle_features:
            if 'prev_msa_first_row' in batch:
                msa_activations[0] = msa_activations[0] + self.prev_msa_first_row_norm(batch['prev_msa_first_row'])
            if 'prev_pair' in batch:
                pair_activations = pair_activations + self.prev_pair_norm(batch['prev_pair'])

        # pos embedding
        if c.max_relative_feature:
            pos = batch['residue_index']
            offset = pos[:, None] - pos[None, :]
            rel_pos = torch.clamp(
                offset + c.max_relative_feature,
                min = 0,
                max = 2 * c.max_relative_feature
            )
            rel_pos = F.one_hot(rel_pos, num_classes=2 * c.max_relative_feature + 1)
            pair_activations = pair_activations + self.pair_activiations(rel_pos.to(pair_activations))
        
        
        # template embedding
        pair_activations=pair_activations.half()

        if c.template.enabled:
            template_batch = {k: batch[k] for k in batch if k.startswith('template_')}
            template_pair_representation = self.template_embedding(
                pair_activations, template_batch, mask_2d
            )
            pair_activations = pair_activations + template_pair_representation
        
        # extra MSA features
        extra_msa_feat = create_extra_msa_feature(batch)
        extra_msa_activations = self.extra_msa_activations(extra_msa_feat)

        # Extra MSA Stack.
        # Jumper et al. (2021) Suppl. Alg. 18 "ExtraMsaStack"
        extra_msa_input = extra_msa_activations
        extra_pair_input = pair_activations
   
        for extra_msa_block in self.extra_msa_stack:
            extra_msa_input, extra_pair_input = extra_msa_block(
                extra_msa_input, extra_pair_input, batch['extra_msa_mask'], mask_2d
            )
        extra_msa_output = {
            'msa': extra_msa_input,
            'pair': extra_pair_input
        }

        pair_activations = extra_msa_output['pair']
        evoformer_input = {
            'msa': msa_activations,
            'pair': pair_activations,
        }

        evoformer_masks = {'msa': batch['msa_mask'], 'pair': mask_2d}

        # Append num_templ rows to msa_activations with template embeddings.
        # Jumper et al. (2021) Suppl. Alg. 2 "Inference" lines 7-8
        if c.template.enabled and c.template.embed_torsion_angles:
            num_templ, num_res = batch['template_aatype'].shape

            # Embed the templates aatypes.
            aatype_one_hot = F.one_hot(batch['template_aatype'], num_classes=22)
            ret = all_atom.atom37_to_torsion_angles(
                aatype=batch['template_aatype'],
                all_atom_pos=batch['template_all_atom_positions'],
                all_atom_mask=batch['template_all_atom_masks'],
                # Ensure consistent behaviour during testing:
                placeholder_for_undefined=not gc.zero_init)
            
            template_features = torch.cat([
                aatype_one_hot,
                ret['torsion_angles_sin_cos'].view(num_templ, num_res, 14),
                ret['alt_torsion_angles_sin_cos'].view(num_templ, num_res, 14),
                ret['torsion_angles_mask']], dim=-1)

            template_activations = self.template_single_embedding(template_features)
            template_activations = F.relu(template_activations)
            template_activations = self.template_projection(template_activations)

            # Concatenate the templates to the msa.
            evoformer_input['msa'] = torch.cat(
                [evoformer_input['msa'], template_activations], dim=0
            )
            # Concatenate templates masks to the msa masks.
            # Use mask from the psi angle, as it only depends on the backbone atoms
            # from a single residue.
            torsion_angle_mask = ret['torsion_angles_mask'][:, :, 2]
            torsion_angle_mask = torsion_angle_mask.type_as(evoformer_masks['msa'])
            evoformer_masks['msa'] = torch.cat(
                [evoformer_masks['msa'], torsion_angle_mask], dim=0)

        evoformer_input_msa = evoformer_input['msa']
        evoformer_input_pair = evoformer_input['pair']
        evoformer_mask_msa = evoformer_masks['msa']
        evoformer_mask_pair = evoformer_masks['pair']
        for evoformer_block in self.evoformer_iteration:
            evoformer_input_msa, evoformer_input_pair = evoformer_block(
                evoformer_input_msa, evoformer_input_pair, 
                evoformer_mask_msa, evoformer_mask_pair
            )
        evoformer_output = {
            'msa': evoformer_input_msa,
            'pair': evoformer_input_pair
        }

        msa_activations = evoformer_output['msa']
        pair_activations = evoformer_output['pair']

        single_activations = self.single_activations(msa_activations[0])
        num_sequences = batch['msa_feat'].shape[0]
        output = {
            'single': single_activations,
            'pair': pair_activations,
            # Crop away template rows such that they are not used in MaskedMsaHead.
            'msa': msa_activations[:num_sequences, :, :],
            'msa_first_row': msa_activations[0],
        }

        return output
        