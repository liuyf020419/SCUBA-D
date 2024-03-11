import torch
from torch import nn
from torch.nn import functional as F

from . import quat_affine
from .common import residue_constants

from .layers import *


class TemplatePairStack(nn.Module):
    def __init__(self, config, global_config):
        super().__init__()
        self.config = config
        self.global_config = global_config

        self.dropout_factor = 0.0 if global_config.deterministic else 1.0 #config.dropout_rate

        self.triangle_attention_starting_node = nn.ModuleList()
        self.triangle_attention_ending_node = nn.ModuleList()
        self.triangle_multiplication_outgoing = nn.ModuleList()
        self.triangle_multiplication_incoming = nn.ModuleList()
        self.pair_transition = nn.ModuleList()
        for _ in range(config.num_block):
            self.triangle_attention_starting_node.append(
                TriangleAttention(
                    config.triangle_attention_starting_node,
                    global_config,
                    config.triangle_attention_starting_node.value_dim,
                )
            )
            self.triangle_attention_ending_node.append(
                TriangleAttention(
                    config.triangle_attention_ending_node,
                    global_config,
                    config.triangle_attention_ending_node.value_dim,
                )
            )
            self.triangle_multiplication_outgoing.append(
                TriangleMultiplication(
                    config.triangle_multiplication_outgoing,
                    global_config,
                    config.triangle_attention_ending_node.value_dim
                )
            )
            self.triangle_multiplication_incoming.append(
                TriangleMultiplication(
                    config.triangle_multiplication_incoming,
                    global_config,
                    config.triangle_attention_ending_node.value_dim
                )
            )
            self.pair_transition.append(
                Transition(config.pair_transition, global_config, 
                    config.triangle_attention_starting_node.value_dim)
            )

    def forward(self, pair_act, pair_mask):
        gc = self.global_config
        c = self.config
        
        if not c.num_block:
            return pair_act

        def block(x, idx):
            """One block of the template pair stack."""
            pair_act = x

            # TODO: shared dropout
            residual = self.triangle_attention_starting_node[idx](pair_act, pair_mask)
            residual = F.dropout(
                residual, training = self.training,
                p = self.dropout_factor * self.triangle_attention_ending_node[idx].config.dropout_rate,
            )
            pair_act = pair_act + residual

            residual = self.triangle_attention_ending_node[idx](pair_act, pair_mask)
            residual = F.dropout(
                residual, training = self.training,
                p = self.dropout_factor * self.triangle_attention_ending_node[idx].config.dropout_rate,
            )
            pair_act = pair_act + residual

            residual = self.triangle_multiplication_outgoing[idx](pair_act, pair_mask)
            residual = F.dropout(
                residual, training = self.training,
                p = self.dropout_factor * self.triangle_multiplication_outgoing[idx].config.dropout_rate,
            )
            pair_act = pair_act + residual

            residual = self.triangle_multiplication_incoming[idx](pair_act, pair_mask)
            residual = F.dropout(
                residual, training = self.training,
                p = self.dropout_factor * self.triangle_multiplication_incoming[idx].config.dropout_rate,
            )
            pair_act = pair_act + residual

            residual = self.pair_transition[idx](pair_act, pair_mask)
            residual = F.dropout(
                residual, training = self.training,
                p = self.dropout_factor * self.pair_transition[idx].config.dropout_rate,
            )
            pair_act = pair_act + residual

            return pair_act

        for bidx in range(c.num_block):
            pair_act = block(pair_act, bidx)
        return pair_act


class SingleTemplateEmbedding(nn.Module):
    def __init__(self, config, global_config):
        super().__init__()
        self.config = config
        self.global_config = global_config

        num_channels = (self.config.template_pair_stack
                        .triangle_attention_ending_node.value_dim)

        # 88 = 39 + 1 + 22 + 22 + 1 + 1 + 1 + 1
        input_dim = config.template_dgrame_dim + config.template_aatype_dim * 2 + 5
        self.embedding2d = Linear(input_dim, num_channels, initializer='relu')

        self.template_pair_stack = TemplatePairStack(config.template_pair_stack, global_config)
        self.output_layer_norm = nn.LayerNorm(num_channels)

    def forward(self, query_embedding, batch, mask_2d):
        
        # assert(mask_2d.type() == query_embedding.type())
        num_res = batch['template_aatype'].size(0)
        
        
        template_mask = batch['template_pseudo_beta_mask']
        template_mask_2d = template_mask.unsqueeze(-1) * template_mask.unsqueeze(-2)
        template_mask_2d = template_mask_2d.type_as(query_embedding)

        # TODO: atoms
        template_dgram = dgram_from_positions(batch['template_pseudo_beta'],
                                              **self.config.dgram_features)
        template_dgram = template_dgram.type_as(query_embedding)

        to_concat = [template_dgram, template_mask_2d[:, :, None]]
        aatype = F.one_hot(batch['template_aatype'], num_classes=22).type_as(query_embedding)
        to_concat.append(aatype.unsqueeze(0).repeat(num_res, 1, 1))
        to_concat.append(aatype.unsqueeze(1).repeat(1, num_res, 1))

        n, ca, c = [residue_constants.atom_order[a] for a in ('N', 'CA', 'C')]
        
        rot, trans = quat_affine.make_transform_from_reference(
            n_xyz=batch['template_all_atom_positions'][:, n],
            ca_xyz=batch['template_all_atom_positions'][:, ca],
            c_xyz=batch['template_all_atom_positions'][:, c])
        affines = quat_affine.QuatAffine(
            quaternion=quat_affine.rot_to_quat(rot, unstack_inputs=True),
            translation=trans,
            rotation=rot,
            unstack_inputs=True)
        points = [x.unsqueeze(-2) for x in affines.translation]
        affine_vec = affines.invert_point(points, extra_dims=1)
        inv_distance_scalar = torch.rsqrt(
            1e-6 + sum([torch.square(x) for x in affine_vec]))

        # Backbone affine mask: whether the residue has C, CA, N
        # (the template mask defined above only considers pseudo CB).
        template_mask = (
            batch['template_all_atom_masks'][..., n] *
            batch['template_all_atom_masks'][..., ca] *
            batch['template_all_atom_masks'][..., c])
        template_mask_2d = template_mask[:, None] * template_mask[None, :]

        inv_distance_scalar = inv_distance_scalar * template_mask_2d.type_as(inv_distance_scalar)

        unit_vector = [(x * inv_distance_scalar)[..., None] for x in affine_vec]

        unit_vector = [x.type_as(query_embedding) for x in unit_vector]
        template_mask_2d = template_mask_2d.type_as(query_embedding)
        if not self.config.use_template_unit_vector:
            unit_vector = [torch.zeros_like(x) for x in unit_vector]
        to_concat.extend(unit_vector)

        to_concat.append(template_mask_2d[..., None])
        act = torch.cat(to_concat, dim=-1)

        # Mask out non-template regions so we don't get arbitrary values in the
        # distogram for these regions.
        act = act * template_mask_2d[..., None]

        # Jumper et al. (2021) Suppl. Alg. 2 "Inference" line 9
        act = self.embedding2d(act)

        # Jumper et al. (2021) Suppl. Alg. 2 "Inference" line 11
        act = self.template_pair_stack(act, mask_2d)
        act = self.output_layer_norm(act)

        return act


class TemplateEmbedding(nn.Module):
    def __init__(self, config, global_config, query_num_channels):
        super().__init__()
        self.config = config
        self.global_config = global_config

        self.single_template_embedding = SingleTemplateEmbedding(config, global_config)
                                                                                                                                                          
        self.attention = Attention(
            config.attention, global_config, 
            (query_num_channels, self.config.template_pair_stack.triangle_attention_ending_node.value_dim), 
            query_num_channels
        )

    def forward(self, query_embedding, template_batch, mask_2d):
        
        num_templates = template_batch['template_mask'].size(0)
        num_channels = (self.config.template_pair_stack
                    .triangle_attention_ending_node.value_dim)
        num_res = query_embedding.shape[0]

        template_mask = template_batch['template_mask']
        template_mask = template_mask.type_as(query_embedding)

        query_num_channels = query_embedding.size(-1)

        # Make sure the weights are shared across templates by constructing the
        # embedder here.
        # Jumper et al. (2021) Suppl. Alg. 2 "Inference" lines 9-12
        def slice_batch(i):
            b = {k: v[i] for k, v in template_batch.items()}
            return b
        template_pair_representation = []
       
        for i in range(num_templates):
            single_template_batch = slice_batch(i)
            sigle_template_pair_representation = self.single_template_embedding(
                query_embedding, single_template_batch, mask_2d
            )
            template_pair_representation.append(sigle_template_pair_representation)
        template_pair_representation = torch.stack(template_pair_representation)
           
        # Cross attend from the query to the templates along the residue
        # dimension by flattening everything else into the batch dimension.
        # Jumper et al. (2021) Suppl. Alg. 17 "TemplatePointwiseAttention"
        flat_query = query_embedding.view(num_res * num_res, 1, query_num_channels)

        flat_templates = template_pair_representation.permute(1,2,0,3).view(num_res * num_res, num_templates, num_channels)
        
        bias = (1e9 * (template_mask[None, None, None, :] - 1.))
        embedding = self.attention(
            flat_query, flat_templates, bias
        )
        embedding = embedding.view(num_res, num_res, query_num_channels)
        embedding = embedding * (torch.sum(template_mask) > 0.).float()

        return embedding
        

        