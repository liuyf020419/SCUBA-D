import sys
import torch
from torch import nn
from torch.nn import functional as F
from typing import Dict, List

from .layers import *
from . import quat_affine
from . import all_atom
from . import r3
from . import utils
sys.path.append("protdiff/models")
from protein_geom_utils import generate_pair_from_pos, preprocess_pair_feature, add_c_beta_from_crd


def squared_difference(x, y):
  return torch.square(x - y)


def permute_final_dims(tensor: torch.Tensor, inds: List[int]):
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])



class InvariantPointAttention(nn.Module):
    def __init__(self, config, global_config, msa_channel, pair_channel, dist_epsilon=1e-8):
        super().__init__()
        self._dist_epsilon = dist_epsilon
        self._zero_initialize_last = global_config.zero_init
        self.config = config
        self.global_config = global_config

        num_head = config.num_head
        num_scalar_qk = config.num_scalar_qk
        num_point_qk = config.num_point_qk
        num_scalar_v = config.num_scalar_v
        num_point_v = config.num_point_v
        num_output = config.num_channel
        assert num_scalar_qk > 0
        assert num_point_qk > 0
        assert num_point_v > 0

        self.q_scalar = Linear(msa_channel, num_head * num_scalar_qk)
        self.kv_scalar = Linear(msa_channel, num_head * (num_scalar_v + num_scalar_qk))
        self.q_point_local = Linear(msa_channel, num_head * 3 * num_point_qk)
        self.kv_point_local = Linear(msa_channel, num_head * 3 * (num_point_qk + num_point_v))

        weights = torch.ones((num_head)) * 0.541323855 # np.log(np.exp(1.) - 1.)
        self.trainable_point_weights = nn.Parameter(data=weights, requires_grad=True)

        self.attention_2d = Linear(pair_channel, num_head)
        num_final_input = num_head * num_scalar_v + num_head * num_point_v * 4 + num_head * pair_channel
        self.output_projection = Linear(num_final_input, num_output, initializer='final')

    def forward(self, inputs_1d, inputs_2d, mask, affine):
        batch_size, num_residues, _ = inputs_1d.shape

        num_head = self.config.num_head
        num_scalar_qk = self.config.num_scalar_qk
        num_point_qk = self.config.num_point_qk
        num_scalar_v = self.config.num_scalar_v
        num_point_v = self.config.num_point_v
        # num_output = self.config.num_channel
        dtype= inputs_1d.dtype

        # Construct scalar queries of shape:
        # [batch_size, num_query_residues, num_head, num_points]
        q_scalar = self.q_scalar(inputs_1d)
        q_scalar = q_scalar.view(batch_size, num_residues, num_head, num_scalar_qk)

        # Construct scalar keys/values of shape:
        # [num_target_residues, num_head, num_points]
        kv_scalar = self.kv_scalar(inputs_1d)
        kv_scalar = kv_scalar.view(batch_size, num_residues, num_head, num_scalar_v + num_scalar_qk)

        k_scalar, v_scalar = torch.split(kv_scalar, num_scalar_qk, dim=-1)
        
        # Construct query points of shape:
        # [num_residues, num_head, num_point_qk]

        # First construct query points in local frame.
        q_point_local = self.q_point_local(inputs_1d)
        q_point_local_dim = q_point_local.size(-1) // 3
        q_point_local = torch.split(q_point_local, q_point_local_dim, dim=-1)
        
        # Project query points into global frame.
        # import pdb; pdb.set_trace()
        q_point_global = affine.apply_to_point(q_point_local, extra_dims=1)
        # Reshape query point for later use.
        q_point = [x.view(batch_size, num_residues, num_head, num_point_qk).to(dtype) for x in q_point_global]

        # Construct key and value points.
        # Key points have shape [num_residues, num_head, num_point_qk]
        # Value points have shape [num_residues, num_head, num_point_v]

        # Construct key and value points in local frame.
        kv_point_local = self.kv_point_local(inputs_1d)
        kv_point_local_dim = kv_point_local.size(-1) // 3
        kv_point_local = torch.split(kv_point_local, kv_point_local_dim, dim=-1)
    
        # Project key and value points into global frame.
        kv_point_global = affine.apply_to_point(kv_point_local, extra_dims=1)
        kv_point_global = [x.view(batch_size, num_residues, num_head, (num_point_qk + num_point_v)).to(dtype) for x in kv_point_global]
        
        # Split key and value points.
        k_point, v_point = list(
            zip(*[
                [x[..., :num_point_qk], x[..., num_point_qk:]]
                for x in kv_point_global
            ]))

        # We assume that all queries and keys come iid from N(0, 1) distribution
        # and compute the variances of the attention logits.
        # Each scalar pair (q, k) contributes Var q*k = 1
        scalar_variance = max(num_scalar_qk, 1) * 1.
        # Each point pair (q, k) contributes Var [0.5 ||q||^2 - <q, k>] = 9 / 2
        point_variance = max(num_point_qk, 1) * 9. / 2

        # Allocate equal variance to scalar, point and attention 2d parts so that
        # the sum is 1.

        num_logit_terms = 3
        scalar_weights = math.sqrt(1.0 / (num_logit_terms * scalar_variance))
        point_weights = math.sqrt(1.0 / (num_logit_terms * point_variance))
        attention_2d_weights = math.sqrt(1.0 / (num_logit_terms))

        # Trainable per-head weights for points.
        trainable_point_weights = F.softplus(self.trainable_point_weights)
        point_weights = point_weights * trainable_point_weights

        v_point = [torch.swapaxes(x, -2, -3) for x in v_point]
        q_point = [torch.swapaxes(x, -2, -3) for x in q_point]
        k_point = [torch.swapaxes(x, -2, -3) for x in k_point]
        dist2 = [
            squared_difference(qx[..., None, :], kx[..., None, :, :])
            for qx, kx in zip(q_point, k_point)
        ]
        dist2 = sum(dist2)
        attn_qk_point = -0.5 * torch.sum(
            point_weights[..., None, None, None] * dist2, dim=-1)

        v = torch.swapaxes(v_scalar, -2, -3)
        q = torch.swapaxes(scalar_weights * q_scalar, -2, -3)
        k = torch.swapaxes(k_scalar, -2, -3)
        attn_qk_scalar = torch.matmul(q, torch.swapaxes(k, -2, -1))
        attn_logits = attn_qk_scalar + attn_qk_point

        # # attention_2d.shape == (r, r, 12)
        # ch2d_in = inputs_2d.size()[-1]
        # attention_2d = self.attention_2d(inputs_2d.reshape(-1, ch2d_in))
        # # attention_2d.shape == (12, r, r)
        # # attention_2d = torch.permute(attention_2d, [2, 0, 1])
        # attention_2d = attention_2d.reshape(batch_size, num_residues, num_residues, -1)
        # attention_2d = attention_2d.permute(0, 3, 1, 2)
        # attention_2d = attention_2d_weights * attention_2d
        # # attn_logits.shape == (12, r, r)
        # attn_logits = attn_logits + attention_2d

        attention_2d = self.attention_2d(inputs_2d)
        # attention_2d = torch.permute(attention_2d, [2, 0, 1])
        attention_2d = permute_final_dims(attention_2d, (2,0,1))
        attention_2d = attention_2d_weights * attention_2d
        attn_logits = attn_logits + attention_2d


        # import pdb; pdb.set_trace()
        mask_2d = mask * torch.swapaxes(mask, -1, -2)
       
        attn_logits = attn_logits - FP16_huge * (1. - mask_2d.unsqueeze(1))

        # [num_head, num_query_residues, num_target_residues]
        # import pdb; pdb.set_trace()
        attn = F.softmax(attn_logits, dim=-1)

        # [num_head, num_query_residues, num_head * num_scalar_v]
        result_scalar = torch.matmul(attn, v)

        # For point result, implement matmul manually so that it will be a float32
        # on TPU.  This is equivalent to
        # result_point_global = [jnp.einsum('bhqk,bhkc->bhqc', attn, vx)
        #                        for vx in v_point]
        # but on the TPU, doing the multiply and reduce_sum ensures the
        # computation happens in float32 instead of bfloat16.
        attn_f = attn.float()
        
        result_point_global = [torch.sum(
            attn_f[..., None] * vx.float()[..., None, :, :],
            dim=-2) for vx in v_point]

        # [num_query_residues, num_head, num_head * num_(scalar|point)_v]
        result_scalar = torch.swapaxes(result_scalar, -2, -3)
        result_point_global = [
            torch.swapaxes(x, -2, -3)
            for x in result_point_global]

        # Features used in the linear output projection. Should have the size
        # [num_query_residues, ?]
        output_features = []
        result_scalar = result_scalar.contiguous().view(batch_size, num_residues, num_head * num_scalar_v)
        output_features.append(result_scalar)

        result_point_global = [
            r.contiguous().view(batch_size, num_residues, num_head * num_point_v)
            for r in result_point_global]
        result_point_local = affine.invert_point(result_point_global, extra_dims=1)
        output_features.extend(result_point_local)

        output_features.append(torch.sqrt(self._dist_epsilon +
                                            torch.square(result_point_local[0].float()) +
                                            torch.square(result_point_local[1].float()) +
                                            torch.square(result_point_local[2].float())))

        # Dimensions: h = heads, i and j = residues,
        # c = inputs_2d channels
        # Contraction happens over the second residue dimension, similarly to how
        # the usual attention is performed.
        result_attention_over_2d = torch.einsum('...hij, ...ijc->...ihc', attn_f, inputs_2d.float())
        num_out = num_head * result_attention_over_2d.shape[-1]
        output_features.append(result_attention_over_2d.view(batch_size, num_residues, num_out))

        final_act = torch.cat(output_features, axis=-1)

        return self.output_projection(final_act)


def generate_pair_from_mergerigid(merge_rigid, batchsize, seqlen, degree=True, scale=1.0):
    B, L = batchsize, seqlen
    alanine_idx = residue_constants.restype_order_with_x["G"]
    pseudo_aatype = torch.LongTensor([alanine_idx] * L)
    batch_pseudo_aatype = torch.stack([pseudo_aatype for _ in range(B)]).to(merge_rigid.trans.x.device)
    # import pdb; pdb.set_trace()
    mergevec = all_atom.frames_and_literature_positions_to_atom14_pos(batch_pseudo_aatype.reshape(-1), merge_rigid)
    pos = r3.vecs_to_tensor(mergevec).reshape(B, L, 14, 3)
    gly_pos = pos[:, :, :3] / scale
    gly_pos = add_c_beta_from_crd(gly_pos)
    # import pdb; pdb.set_trace()
    pair = generate_pair_from_pos(gly_pos, degree)
    pair = preprocess_pair_feature(pair)
    return pair


def split_affine_batch(merge_affine, batchsize, seqlen):
    # try:
    split_affine_tensor = merge_affine.to_tensor().reshape(batchsize, seqlen, 7)
    # except:
    #     import pdb; pdb.set_trace()
    # import pdb; pdb.set_trace()
    split_affine_tensor = split_affine_tensor
    split_affine = quat_affine.QuatAffine.from_tensor(split_affine_tensor)
    split_rigid = r3.rigids_from_quataffine(split_affine)
    return split_affine, split_rigid


def pad_affine_flat12(affine_flat12):
    B, L = affine_flat12.shape[:2]
    pad_affine_flat12_rot = torch.eye(3).reshape([1, 1, 1, 9]).repeat(B, L, 7, 1)
    pad_affine_flat12_trans = torch.zeros(3).reshape([1, 1, 1, 3]).repeat(B, L, 7, 1) * 1e-10
    pad_affine_flat12_ = torch.cat([pad_affine_flat12_rot, pad_affine_flat12_trans], -1).to(affine_flat12.device)
    # import pdb; pdb.set_trace()
    return torch.cat([affine_flat12[:, :, None], pad_affine_flat12_], 2)


class FoldBlock(nn.Module):
    def __init__(self, config, global_config, update_affine, msa_channel):
        super().__init__()
        self.config = config
        self.global_config = global_config
        self.dropout_factor = 0.0 if global_config.deterministic else 1.0
        # self.encode_2d = encode_2d

        dropout_rate = 0.0 if global_config.deterministic else config.dropout
        self.dropout = nn.Dropout(p = dropout_rate)

        final_init = 'zeros' if self.global_config.zero_init else 'linear'

        self.invariant_point_attention = InvariantPointAttention(config, global_config, msa_channel, config.pair_channel)
        self.attention_layer_norm = nn.LayerNorm(msa_channel)

        layers=[]
        in_dim= msa_channel
        for i in range(config.num_layer_in_transition):
            if i < config.num_layer_in_transition -1:
                layers.append(Linear(in_dim, config.num_channel, initializer="relu"))
                layers.append(nn.ReLU())
            else:
                layers.append(Linear(in_dim, config.num_channel, initializer="zeros"))
            in_dim = config.num_channel
        self.transition= nn.Sequential(*layers)

        self.transition_layer_norm = nn.LayerNorm(config.num_channel)

        if update_affine:
            affine_update_size = 6
            self.affine_update = Linear(msa_channel, affine_update_size, initializer=final_init)
            

    def forward(self, 
                activations,
                update_affine,
                fix_region=None
                ):
        # import pdb; pdb.set_trace()
        affine = activations["affine"]
        pair = activations['pair_act']
        pair_mask = activations['pair_mask']
        seq_mask = activations['seq_mask']
        act = activations['act']

        B, L = affine.shape[:2]
        # import pdb; pdb.set_trace()
        affine = affine[:, :, 0]
        
        rigids = r3.rigids_from_tensor_flat12(affine.reshape(-1, 1, 12))
        affine = r3.rigids_to_quataffine_m(rigids) # (B, L, 1, 7)
        # import pdb; pdb.set_trace()
        affine, rigid = split_affine_batch(affine, B, L)

        act_pair = pair
        
        act_attn = self.invariant_point_attention(
            inputs_1d=act,
            inputs_2d=act_pair,
            mask=seq_mask,
            affine=affine)

        act = act + act_attn

        act = self.dropout(act)
        act = self.attention_layer_norm(act)

        act = self.transition(act) + act
        act = self.dropout(act)
        act = self.transition_layer_norm(act)
        # import pdb; pdb.set_trace()
        if update_affine:
            # This block corresponds to
            # Jumper et al. (2021) Alg. 23 "Backbone update"

            # Affine update
            affine_update = self.affine_update(act)
            affine = affine.pre_compose(affine_update, fix_region)
            # if stepsize is not None:
            #     import pdb; pdb.set_trace()

        ## frame and postion for output
        outputs = {'affine': affine.to_tensor(),
                   'single_act': act
                  }

        affine = affine.apply_rotation_tensor_fn(torch.detach)

        affine_flat12 = r3.rigids_to_tensor_flat12(r3.rigids_from_quataffine(affine))
        # import pdb; pdb.set_trace()
        affine_flat12 = pad_affine_flat12(affine_flat12)

        new_activations = {
            'act': act,
            'affine': affine_flat12,
            'pair_act': act_pair,
            'pair_mask': pair_mask,
            'seq_mask': seq_mask
        }
        
        return new_activations, outputs



class AffineGenerator(nn.Module):
    def __init__(self, config, global_config, msa_channel, pair_channel):
        super().__init__()
        self.config = config
        self.global_config = global_config

        self.single_layer_norm = nn.LayerNorm(msa_channel)
        self.initial_projection = Linear(msa_channel, config.num_channel)

        self.fold_iterations = nn.ModuleList([FoldBlock(config, 
                                                        global_config, 
                                                        True, 
                                                        msa_channel,
                                                        )
                                                for _ in range(config.num_layer)])

        self.pair_layer_norm = nn.LayerNorm(pair_channel)

        self.affine_out = Linear(2 * 7, 7, initializer="relu")

    def forward(self, representations, fix_region=None):
        c = self.config
        
        act = self.single_layer_norm(representations['single'])
        act = self.initial_projection(act)
        # import pdb; pdb.set_trace()
        act_2d = self.pair_layer_norm(representations['pair'])
        affine_scaler = torch.FloatTensor([1.] * 9 + \
                        [self.global_config.position_scale] * 3).to(representations['single'].device)

        activations = {'act': act,
                       'affine': representations['frame'] * affine_scaler,
                       'pair_act': act_2d,
                       'pair_mask': representations['pair_mask'],
                       'seq_mask': representations['seq_mask'][:, :, None]
                      }

        outputs = []

        for l_id in range(c.num_layer):
            fold_iterations = self.fold_iterations[l_id]
            # if torch.any(torch.isnan(activations["affine"])):
            #     import pdb; pdb.set_trace()
            activations, output = fold_iterations(
                activations,
                update_affine=True, 
                fix_region=fix_region)

            outputs.append(output)

        output = {
            'affine': torch.stack([out['affine'] for out in outputs]),
            'single_act': torch.stack([out['single_act'] for out in outputs])
        }


        # Include the activations in the output dict for use by the LDDT-Head.
        output['act'] = activations['act']

        return output


class StructureModule(nn.Module):
    def __init__(self, config, global_config):
        super().__init__()
        self.config = config
        self.global_config = global_config
        single_channel = config.single_channel
        pair_channel = config.pair_channel
        self.affine_generator = AffineGenerator(config, global_config, single_channel, pair_channel)

    def forward(self, representations, fix_region=None):
        ret = {}
        # import pdb; pdb.set_trace()
        output = self.affine_generator(representations, fix_region)
        # import pdb; pdb.set_trace()
        ret['representations'] = {'structure_module': output['act'],
                                  'single_acts': output['single_act']}

        ret['traj'] = output['affine'] * torch.FloatTensor([1.] * 4 +
                                                [1. /self.global_config.position_scale] * 3).to(output['act'].device)

        return ret