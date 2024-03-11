import numbers
from typing import Union, Sequence, List
import torch
from torch import nn
from torch.nn import functional as F
import math
from typing import Tuple, List, Callable, Any, Dict
from .common import residue_constants

import torch.nn.init as torch_init
from functools import partialmethod
from .utils import checkpoint_function, sharded_layer


FP16_huge=2**15
FP16_tiny=2**-20


def permute_final_dims(tensor: torch.Tensor, inds: List[int]):
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])


def flatten_final_dims(t: torch.Tensor, no_dims: int):
    return t.reshape(t.shape[:-no_dims] + (-1,))


def flatten_prev_dims(t:torch.Tensor, no_dims:int):
    return t.reshape((-1,) + t.shape[no_dims:])


# call this function with config seems too heavy, and zero_init seems tobe always
# so add final to zero function, and we can change it in code if required
# def final_init(config):
#     return 'zeros' if config.zero_init else 'linear'

def param_init_(tensor:torch.Tensor, method='linear'):
    init_fns= {
        "linear":lambda x: torch_init.kaiming_normal_(x, nonlinearity='linear'),
        "relu":lambda x: torch_init.kaiming_normal_(x, nonlinearity='relu'),
        "gating":lambda x: torch_init.constant_(x, 0.),
        "final":lambda x: torch_init.constant_(x, 0.),
        "glorot":lambda x: torch_init.xavier_uniform_(x,gain =1)
    }
    if method not in init_fns:
        raise NotImplementedError(f'unknown init function {method}')
    return init_fns[method](tensor)

class Linear(nn.Linear):
    def __init__(self, in_dim: int, out_dim: int, bias: bool=True, initializer:str="linear") -> None:
        super().__init__(in_dim, out_dim, bias)
        param_init_(self.weight, initializer)
        if self.bias is not None and initializer == 'gating':
            torch_init.constant_(self.bias, 1.)


def dgram_from_positions(positions, num_bins, min_bin, max_bin):
    """Compute distogram from amino acid positions.

    Arguments:
        positions: [N_res, 3] Position coordinates.
        num_bins: The number of bins in the distogram.
        min_bin: The left edge of the first bin.
        max_bin: The left edge of the final bin. The final bin catches
            everything larger than `max_bin`.

    Returns:
        Distogram with the specified number of bins.
    """

    def squared_difference(x, y):
        return torch.square(x - y)
    device = positions.device
    lower_breaks = torch.linspace(min_bin, max_bin, num_bins)
    lower_breaks = torch.square(lower_breaks).to(device)
    upper_breaks = torch.cat([lower_breaks[1:],
                                torch.FloatTensor([1e8]).to(device)], axis=-1)

    dist2 = torch.sum(
        squared_difference(
            positions.unsqueeze(-2), positions.unsqueeze(-3)),
        dim=-1, keepdims=True)

    dgram = ((dist2 > lower_breaks).float() *
            (dist2 < upper_breaks).float())
    return dgram


def create_extra_msa_feature(batch):
    """Expand extra_msa into 1hot and concat with other extra msa features.

    We do this as late as possible as the one_hot extra msa can be very large.

    Arguments:
        batch: a dictionary with the following keys:
        * 'extra_msa': [N_extra_seq, N_res] MSA that wasn't selected as a cluster
        centre. Note, that this is not one-hot encoded.
        * 'extra_has_deletion': [N_extra_seq, N_res] Whether there is a deletion to
        the left of each position in the extra MSA.
        * 'extra_deletion_value': [N_extra_seq, N_res] The number of deletions to
        the left of each position in the extra MSA.

    Returns:
        Concatenated tensor of extra MSA features.
    """
    # 23 = 20 amino acids + 'X' for unknown + gap + bert mask
    msa_1hot = F.one_hot(batch['extra_msa'], num_classes=23)
    msa_feat = [msa_1hot,
                batch['extra_has_deletion'].unsqueeze(-1),
                batch['extra_deletion_value'].unsqueeze(-1)]
    return torch.cat(msa_feat, dim=-1)


# TODO
def pseudo_beta_fn(aatype, all_atom_positions, all_atom_masks):
    """Create pseudo beta features."""
    is_gly = (aatype ==  residue_constants.restype_order['G']) #torch.equal(aatype, residue_constants.restype_order['G'])
    ca_idx = residue_constants.atom_order['CA']
    cb_idx = residue_constants.atom_order['CB']

    new_shape = [1] * len(is_gly.shape) + [3]
    pseudo_beta = torch.where(
        is_gly[..., None].repeat(*new_shape),
        all_atom_positions[..., ca_idx, :],
        all_atom_positions[..., cb_idx, :])

    if all_atom_masks is not None:
        pseudo_beta_mask = torch.where(
            is_gly, all_atom_masks[..., ca_idx], all_atom_masks[..., cb_idx])
        pseudo_beta_mask = pseudo_beta_mask.float()
        return pseudo_beta, pseudo_beta_mask
    else:
        return pseudo_beta





class Attention(nn.Module):
    """Multihead attention."""
    def __init__(self, config, global_config, qm_dims, output_dim):
        super().__init__()
        self.config = config
        self.global_config = global_config
        self.output_dim = output_dim

        q_dim, m_dim = qm_dims
        # Sensible default for when the config keys are missing
        key_dim = self.config.get('key_dim', q_dim)
        value_dim = self.config.get('value_dim',m_dim)

        num_head = self.config.num_head
        self.num_head= num_head
        assert key_dim % num_head == 0 and value_dim % num_head == 0
        key_dim = key_dim // num_head
        value_dim = value_dim // num_head

        self.key_scale = key_dim ** (-0.5)

        self.linear_q= Linear(q_dim,num_head*key_dim, bias=False, initializer='glorot')
        self.linear_k = Linear(m_dim, num_head*key_dim, bias=False, initializer="glorot")
        self.linear_v= Linear(m_dim, num_head*value_dim, bias= False, initializer="glorot")
        self.linear_o= Linear(num_head*value_dim, output_dim, initializer = "final")
        self.linear_g=None
        if config.gating:
            self.linear_g= Linear(q_dim, num_head*value_dim, bias=True, initializer= "gating")
        self.sigmoid= nn.Sigmoid()
        self.softmax= nn.Softmax(dim=-1)

    def forward(self, q_data, m_data, bias,nonbatched_bias= None):
        """Builds Attention module.

        Arguments:
        q_data: A tensor of queries, shape [batch_size, NRes, NRes, dim].
        m_data: A tensor of memories from which the keys and values are
            projected, shape [batch_size, N_keys, m_channels].
        bias: A bias for the attention, shape [batch_size, N_Res, 1,1, N_Res].
        nonbatched_bias: Shared bias, shape [batch-size, Head, NRes, NRes]
        Returns:
        A float32 tensor of shape [batch_size, N_queries, output_dim].
        """

        # B, NRes, NRes, head_num
        q= self.linear_q(q_data)*self.key_scale
        # B, NRes, NRes, head_num
        k= self.linear_k(m_data)
        # B, NRes, NRes, head_num
        v= self.linear_v(m_data)

        # *, T, H, D
        q=q.view(q.shape[:-1]+(self.num_head, -1))
        k=k.view(k.shape[:-1]+(self.num_head, -1))
        v=v.view(v.shape[:-1]+(self.num_head, -1))


        # *THD->*HTD,*HDT
        q=permute_final_dims(q, (1,0,2))
        k=permute_final_dims(k, (1,2,0))
        v= permute_final_dims(v,(1,0,2))
        # def atten_function(q,k, bias, bias1=None):
        #     logits= torch.matmul(q,k) +bias
        #     if bias1 is not None:
        #         logits += bias1
        #     return logits
        
        # def weight_avg_function(logits, v):
        #     weight = F.softmax(logits, dim=-1)
        #     weighted_avg = torch.matmul(weight, v).transpose(-2,-3)
        #     return weighted_avg
        # bias1= nonbatched_bias
        # if bias1 is not None:
        #     bias1= bias1.unsqueeze(-4)
        # logits = checkpoint_function(atten_function, q,k, bias, bias1)
        # weighted_avg= checkpoint_function(weight_avg_function, logits, v)

        # *HTT
        logits= torch.matmul(q,k)+bias
        
        if nonbatched_bias is not None:
            logits += nonbatched_bias
        weight = self.softmax(logits)
        # *HTD
        weighted_avg= torch.matmul(weight, v)
        # *HTD ->*THD
        weighted_avg= weighted_avg.transpose(-2,-3)

        if self.config.gating:
            g= self.sigmoid(self.linear_g(q_data))
            g= g.view(g.shape[:-1]+(self.num_head, -1))
            weighted_avg = weighted_avg*g
        weighted_avg = flatten_final_dims(weighted_avg, 2)
        output= self.linear_o(weighted_avg)
        return output



class GlobalAttention(nn.Module):
    """Multihead attention."""
    def __init__(self, config, global_config, qm_dims, output_dim):
        super().__init__()
        self.eps= FP16_tiny
        self.config = config
        self.global_config = global_config
        self.output_dim = output_dim

        q_dim, m_dim = qm_dims
        key_dim = self.config.get('key_dim', q_dim)
        value_dim = self.config.get('value_dim',m_dim)

        num_head = self.config.num_head
        self.num_head=num_head
        assert key_dim % num_head == 0
        assert value_dim % num_head == 0
        key_dim = key_dim // num_head
        value_dim = value_dim // num_head

        self.key_scalar = key_dim ** (-0.5)

        self.linear_q= Linear(q_dim, num_head*key_dim, bias=False, initializer="glorot")
        self.linear_k= Linear(m_dim, key_dim, bias=False, initializer= "glorot")
        self.linear_v= Linear(m_dim, value_dim,bias=False, initializer="glorot")
        self.linear_o= Linear(num_head*value_dim, output_dim,initializer="final")
        self.linear_g= None
        if config.gating:
            self.linear_g = Linear(q_dim, num_head*value_dim, bias=True, initializer='gating')
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, q_data, m_data, q_mask, bias):
        """Builds Attention module.

        Arguments:
        q_data: A tensor of queries, shape [batch_size, N_res, N_Seq, m_channels].
        m_data: same as q_data
        q_mask: shape[batch_size, N_res, N_seq]
        bias: same shape as qmask

        Returns:
        A float32 tensor of shape [batch_size, N_res, N_seq, output_dim].
        """
        # BTND ->BTD, mask: BTN
        
        # q_avg= torch.sum(q_data*q_mask.unsqueeze(-1), dim=-2) / (
        #     torch.sum(q_mask,dim=-1).unsqueeze(-1)+self.eps
        # )
        
        q_avg= torch.sum(q_data*q_mask, dim=-2) / (
            torch.sum(q_mask,dim=-2)+self.eps
        )
        # BTHD
        q= self.linear_q(q_avg)*self.key_scalar
        q= q.view(q.shape[:-1]+(self.num_head, -1))
        # BTND
        k= self.linear_k(m_data)
        v = self.linear_v(m_data)
        # BTHN
        
        logits= torch.matmul(q, k.transpose(-1,-2))+bias
        weight= self.softmax(logits)
        #BTHD
        weighted_avg = torch.matmul(weight, v)
        if self.config.gating:
            #BTND->BTNHD
            g= self.sigmoid( self.linear_g(q_data))
            g= g.view(g.shape[:-1]+(self.num_head,-1))
            weighted_avg = weighted_avg.unsqueeze(-3)*g
            weighted_avg = weighted_avg.view(weighted_avg.shape[:-2] +(-1,))
            #BTND
            out= self.linear_o(weighted_avg)
        else:
            weighted_avg = weighted_avg.view(weighted_avg.shape[:-2]+(-1,))
            out= self.linear_o(weighted_avg)
            out = out[...,None,:]
        return out


class TriangleAttention(nn.Module):
    """Triangle Attention.

    Jumper et al. (2021) Suppl. Alg. 13 "TriangleAttentionStartingNode"
    Jumper et al. (2021) Suppl. Alg. 14 "TriangleAttentionEndingNode"
    """
    def __init__(self, config, global_config, num_channels, is_template_stack=False):
        super().__init__()
        self.config = config
        self.global_config = global_config
        assert config.orientation in ['per_row', 'per_column'], f"bad orientation {config.orientation}"
        self.is_per_column = config.orientation == "per_column"
        self.is_template_stack = is_template_stack
        self.query_norm = LayerNormFP32(num_channels)
        self.linear_2d= Linear(num_channels, config.num_head, bias=False, initializer= "linear")
        value_dim = config.value_dim if hasattr(config, 'value_dim') else num_channels
        qm_dims = (value_dim, value_dim)
        self.attention = Attention(config, global_config, qm_dims, num_channels)
        self.out_single = self.config.out_single
        if self.out_single:
            self.out_single= Linear(num_channels, num_channels, bias=False, initializer= "linear")

    def process(self, pair_act, pair_mask, num_batch_dims=2):
        """
            pair_act: batch_size, N_Res, NRes, D
            pair_mask: shape batch_size, N_Res, N_Res
        """
        # import pdb;pdb.set_trace()
        if self.is_per_column:
            pair_act= torch.swapaxes(pair_act, -2, -3)
            pair_mask= torch.swapaxes(pair_mask, -1,-2)
        # B, NRes, 1, 1, NRes
        bias=  (FP16_huge * (pair_mask - 1.))[..., None, None, :]
        pair_act = self.query_norm(pair_act)
        # B, NRes, NRes, head_num
        nonbatched_bias = self.linear_2d(pair_act)
        # B, head_num, NRes, NRes
        nonbatched_bias = permute_final_dims(nonbatched_bias, (2,0,1))
        # B, 1, head_num, NRes, NRes
        nonbatched_bias= nonbatched_bias.unsqueeze(-4)
        if self.global_config.subbatch_size >0:
            pair_act = sharded_layer(
                self.attention,
                (pair_act, pair_act, bias, nonbatched_bias),
                self.global_config.subbatch_size,
                num_batch_dims=num_batch_dims
            )
        else:
            pair_act = self.attention(pair_act, pair_act, bias, nonbatched_bias)
        if self.is_per_column:
            pair_act = torch.swapaxes(pair_act, -2,-3)
            pair_mask= torch.swapaxes(pair_mask, -1,-2)
        pair_act = pair_act * pair_mask[..., None]
        if self.out_single:
            # single_act = self.out_single(pair_act.sum(-2))[:, None]
            # import pdb; pdb.set_trace()
            single_act = self.out_single(pair_act[:, 0])[:, None]
            return pair_act, single_act
        else:
            return pair_act

    def forward(self, pair_act, pair_mask):
        """
            pair_act: batch_size, N_Res, NRes, D
            pair_mask: shape batch_size, N_Res, N_Res
        """
        if not self.global_config.is_inference:
            # training
            return self.process(pair_act, pair_mask)

        if self.is_template_stack:
            act_shape = pair_act.shape
            # B*t, N, N, D
            pair_act = pair_act.reshape(-1, *act_shape[2:])
            # B, t, N, D
            pair_mask = pair_mask.repeat(1, act_shape[1], 1, 1)
            mask_shape = pair_mask.shape
            # B*t, N, N, D
            pair_mask = pair_mask.reshape(-1, *mask_shape[2:])

        outputs = []
        for act, mask in zip(pair_act, pair_mask):
            # if not is_template_stack
            # act.shape == (N, N, D)
            # mask.shape == (N, N)
            out = self.process(act, mask, num_batch_dims=1)
            outputs.append(out)
        pair_act = torch.stack(outputs)
        if self.is_template_stack:
            pair_act = pair_act.reshape(*act_shape)
        return pair_act

# class TriangleAttention(nn.Module):
#     """Triangle Attention.

#     Jumper et al. (2021) Suppl. Alg. 13 "TriangleAttentionStartingNode"
#     Jumper et al. (2021) Suppl. Alg. 14 "TriangleAttentionEndingNode"
#     """
#     def __init__(self, config, global_config, num_channels, is_template_stack=False):
#         super().__init__()
#         self.config = config
#         self.global_config = global_config
#         assert config.orientation in ['per_row', 'per_column'], f"bad orientation {config.orientation}"
#         self.is_per_column = config.orientation == "per_column"
#         self.is_template_stack = is_template_stack
#         self.query_norm = LayerNormFP32(num_channels)
#         self.linear_2d= Linear(num_channels, config.num_head, bias=False, initializer= "linear")
#         value_dim = config.value_dim if hasattr(config, 'value_dim') else num_channels
#         qm_dims = (value_dim, value_dim)
#         self.attention = Attention(config, global_config, qm_dims, num_channels)

#     def process(self, pair_act, pair_mask, num_batch_dims=2):
#         """
#             pair_act: batch_size, N_Res, NRes, D
#             pair_mask: shape batch_size, N_Res, N_Res
#         """
        
#         if self.is_per_column:
#             pair_act= torch.swapaxes(pair_act, -2, -3)
#             pair_mask= torch.swapaxes(pair_mask, -1,-2)
#         # B, NRes, 1, 1, NRes
#         bias=  (FP16_huge * (pair_mask - 1.))[..., None, None, :]
#         pair_act = self.query_norm(pair_act)
#         # B, NRes, NRes, head_num
#         nonbatched_bias = self.linear_2d(pair_act)
#         # B, head_num, NRes, NRes
#         nonbatched_bias = permute_final_dims(nonbatched_bias, (2,0,1))
#         # B, 1, head_num, NRes, NRes
#         nonbatched_bias= nonbatched_bias.unsqueeze(-4)
#         if self.global_config.subbatch_size >0:
#             pair_act = sharded_layer(
#                 self.attention,
#                 (pair_act, pair_act, bias, nonbatched_bias),
#                 self.global_config.subbatch_size,
#                 num_batch_dims=num_batch_dims
#             )
#         else:
#             pair_act = self.attention(pair_act, pair_act, bias, nonbatched_bias)
#         if self.is_per_column:
#             pair_act = torch.swapaxes(pair_act, -2,-3)
#             pair_mask= torch.swapaxes(pair_mask, -1,-2)
#         pair_act = pair_act * pair_mask[..., None]
#         return pair_act

#     def forward(self, pair_act, pair_mask):
#         """
#             pair_act: batch_size, N_Res, NRes, D
#             pair_mask: shape batch_size, N_Res, N_Res
#         """
#         if not self.global_config.is_inference:
#             # training
#             return self.process(pair_act, pair_mask)

#         if self.is_template_stack:
#             act_shape = pair_act.shape
#             # B*t, N, N, D
#             pair_act = pair_act.reshape(-1, *act_shape[2:])
#             # B, t, N, D
#             pair_mask = pair_mask.repeat(1, act_shape[1], 1, 1)
#             mask_shape = pair_mask.shape
#             # B*t, N, N, D
#             pair_mask = pair_mask.reshape(-1, *mask_shape[2:])

#         outputs = []
#         for act, mask in zip(pair_act, pair_mask):
#             # if not is_template_stack
#             # act.shape == (N, N, D)
#             # mask.shape == (N, N)
#             out = self.process(act, mask, num_batch_dims=1)
#             outputs.append(out)
#         pair_act = torch.stack(outputs)
#         if self.is_template_stack:
#             pair_act = pair_act.reshape(*act_shape)
#         return pair_act


class LayerNormFP32(nn.LayerNorm):
    def half(self):
        return self
    def bfloat16(self):
        return self
 
    def forward(self, input):
        dtype= input.dtype

        x= super().forward(input.float())
        return x.to(dtype)

class TriangleMultiplication(nn.Module):
    def __init__(self, config, global_config, input_dim, is_template_stack=False):
        super().__init__()
        self.config = config
        self.global_config = global_config
        self.is_template_stack = is_template_stack

        self.layer_norm_input = LayerNormFP32(input_dim)
        self.center_layer_norm =LayerNormFP32(input_dim)
        self.left_projection = Linear(input_dim, config.num_intermediate_channel)
        self.right_projection = Linear(input_dim, config.num_intermediate_channel)
        self.left_gate = Linear(
            input_dim, config.num_intermediate_channel,
            bias=True,
            initializer= "gating"
        )
        self.right_gate = Linear(
            input_dim, config.num_intermediate_channel,
            bias=True,
            initializer= "gating"
        )
        self.output_projection = Linear(
            config.num_intermediate_channel, input_dim,
            initializer="final"
        )
        self.gating_linear = Linear(
            config.num_intermediate_channel, input_dim, 
            initializer="final"
        )
        
    def process(self, act, mask):
        c = self.config
        gc = self.global_config

        mask = mask.unsqueeze(-1)
        act = self.layer_norm_input(act)
        input_act = act
        left_proj_act = mask * self.left_projection(act)
        right_proj_act = mask * self.right_projection(act)
        left_gate_values = torch.sigmoid(self.left_gate(act))
        right_gate_values = torch.sigmoid(self.right_gate(act))
        
        left_proj_act = left_proj_act * left_gate_values
        right_proj_act = right_proj_act * right_gate_values
        dtype= act.dtype
        if self.global_config.is_inference:
            del act, left_gate_values, right_gate_values
            torch.cuda.empty_cache()
            # def compute_chunk(left):
            #     return torch.einsum(c.equation, left.float(), right_proj_act.float())
            sb_size = self.global_config.subbatch_size
            act = []
            start = 0
            while start < left_proj_act.shape[-1]:
                end = start + sb_size
                act.append(
                    torch.einsum(c.equation, left_proj_act[..., start:end], right_proj_act[..., start:end])
                )
                start = end
            act = torch.cat(act, dim=-1)
        else:
            act = torch.einsum(c.equation, left_proj_act.float(), right_proj_act.float())
        
        act = self.center_layer_norm(act)
        act= act.to(dtype)
        act = self.output_projection(act)
        gate_values = torch.sigmoid(self.gating_linear(input_act))

        act = act * gate_values
        return act

    def forward(self, act, mask):
        if not self.global_config.is_inference:
            return self.process(act, mask)

        if self.is_template_stack:
            act_shape = act.shape
            act = act.reshape(-1, *act_shape[2:])
            mask = mask.repeat(1, act_shape[1], 1, 1)
            mask_shape = mask.shape
            mask = mask.reshape(-1, *mask_shape[2:])
        outputs = []
        for a, m in zip(act, mask):
            out = self.process(a, m)
            outputs.append(out)
        act = torch.cat(outputs)
        if self.is_template_stack:
            act = act.reshape(*act_shape)
        return act
        



class Transition(nn.Module):
    def __init__(self, config, global_config, input_dim):
        super().__init__()
        self.config = config
        self.global_config = global_config

        num_intermediate = int(input_dim * self.config.num_intermediate_factor)
        
        self.input_layer_norm = LayerNormFP32(input_dim)
        self.transition1 = Linear(input_dim, num_intermediate, initializer='relu')
        self.transition2 = Linear(num_intermediate, input_dim, initializer="final")
    
    def forward(self, act, mask):
        mask = mask.unsqueeze(-1)
        act = self.input_layer_norm(act)
        act = self.transition1(act)
        act = F.relu(act)
        act = self.transition2(act)
        act= act*mask
        return act

class LinearFp32(Linear):
    def half(self):
        return self
    def bfloat16(self):
        return self
 
    def forward(self, input):
        dtype= input.dtype

        x= super().forward(input.float())
        return x.to(dtype)


class OuterProduct(nn.Module):
    def __init__(self, config, global_config, num_input_channel, num_output_channel):
        super().__init__()
        self.global_config = global_config
        self.config = config
        self.num_output_channel = num_output_channel

        self.layer_norm_input = nn.LayerNorm(num_input_channel)
        self.left_projection = Linear(num_input_channel, config.num_outer_channel, initializer='linear')
        self.right_projection = Linear(num_input_channel, config.num_outer_channel, initializer='linear')

        self.act_projections = Linear(1, num_output_channel, initializer='linear')

    def forward(self, act):
        
        act = self.layer_norm_input(act)
        left_act = self.left_projection(act)
        right_act = self.right_projection(act)
        act = torch.einsum("...ia,...jb->...ij", left_act, right_act)[:, :, :, None]

        b, n, _, _ = act.shape
        act = self.act_projections(act.reshape(-1, 1)).reshape(b, n, n, -1)

        return act       



class OuterProductMean(nn.Module):
    def __init__(self, config, global_config, num_input_channel, num_output_channel):
        super().__init__()
        self.global_config = global_config
        self.config = config
        self.num_output_channel = num_output_channel

        self.layer_norm_input = LayerNormFP32(num_input_channel)

        self.left_projection = Linear(num_input_channel, config.num_outer_channel, initializer='linear')
        self.right_projection = Linear(num_input_channel, config.num_outer_channel, initializer='linear')

        self.linear_o = LinearFp32(config.num_outer_channel*config.num_outer_channel,num_output_channel, initializer='final')

    def process(self, act, mask):
        gc = self.global_config
        c = self.config
        
        act = permute_final_dims(act, (1, 0, 2))
        mask = mask[..., None].to(act.dtype)
        act = self.layer_norm_input(act)
        # B, 1, N_res, ch
        left_act = mask * self.left_projection(act)
        right_act = mask * self.right_projection(act)
        # here float32 now
        dtype= act.dtype
        # left_act.shape == (B, 1, ch, N_res)
        # right_act.shape == (B, 1, N_res, ch)
        left_act= permute_final_dims(left_act, (0,2,1))
        act = torch.einsum('...acb,...ade->...dbce', left_act.float(), right_act.float())
        # act.shape == (B, N_res, N_res, ch, ch)
        act = flatten_final_dims(act, 2)
        # act.shape == (B, N_res, N_res, ch*ch)
        act= self.linear_o(act)
        # act.shape == (B, N_res, N_res, ch*ch)
        act= permute_final_dims(act, (1,0,2))
        epsilon = 1e-3
        norm = torch.einsum('...abc,...adc->...bdc', mask.float(), mask.float())
        act= act/(epsilon +norm)
        act= act.to(dtype)
        return act

    def forward(self, act, mask):
        if not self.global_config.is_inference:
            return self.process(act, mask)
        
        gc = self.global_config
        c = self.config
        
        act = permute_final_dims(act, (1, 0, 2))
        mask = mask[..., None].to(act.dtype)
        act = self.layer_norm_input(act)
        left_act = mask * self.left_projection(act)
        right_act = mask * self.right_projection(act)

        dtype= left_act.dtype
        def compute_chunk(left_act):
            
            left_act= permute_final_dims(left_act, (0,2,1))
            act = torch.einsum('...acb,...ade->...dbce', left_act.float(), right_act.float())
            
            act = flatten_final_dims(act, 2)
            act= self.linear_o(act)
            return permute_final_dims(act, (1,0,2))

        torch.cuda.empty_cache()
            
        if self.global_config.subbatch_size >0:
            # act = sharded_layer(
            #     compute_chunk,
            #     (left_act),
            #     self.global_config.subbatch_size,
            #     num_batch_dims=3
            # )
            sb_size = self.global_config.subbatch_size
            start = 0
            outputs = []
            while start < left_act.shape[2]:
                left_act_chunk = left_act[:, :, start:start+sb_size]
                outputs.append(compute_chunk(left_act_chunk))
                start = start + sb_size
            act = torch.cat(outputs, 1)
        else:
            act = compute_chunk(left_act)

        # here float32 now
        # dtype= act.dtype
        # left_act= permute_final_dims(left_act, (0,2,1))
        # act = torch.einsum('...acb,...ade->...dbce', left_act.float(), right_act.float())
        
        # act = flatten_final_dims(act, 2)
        # act= self.linear_o(act)
        # act= permute_final_dims(act, (1,0,2))
        epsilon = 1e-3
        norm = torch.einsum('...abc,...adc->...bdc', mask.float(), mask.float())
        act= act/(epsilon +norm)
        act= act.to(dtype)
        return act



class MSARowAttentionWithPairBias(nn.Module):
    def __init__(self, config, global_config, msa_channel, pair_channel):
        super().__init__()
        self.global_config = global_config
        self.config = config
       
        assert self.config.orientation == 'per_row',\
            f"MSARowAttentionWithPairBias wit orient {self.config.orientation}"
        self.query_norm = LayerNormFP32(msa_channel)
        self.feat_2d_norm = LayerNormFP32(pair_channel)

        self.linear_2d = Linear(pair_channel, config.num_head, bias= False, initializer='linear')
        self.attention = Attention(
            config, global_config, (msa_channel, msa_channel),  msa_channel
        )

    def forward(self, msa_act, msa_mask, pair_act):
        bias = (FP16_huge * (msa_mask - 1.))[..., None, None, :]
        bias= bias.to(msa_act.dtype)
        msa_act = self.query_norm(msa_act)
        pair_act = self.feat_2d_norm(pair_act)
        nonbatched_bias = self.linear_2d(pair_act)
        nonbatched_bias= permute_final_dims(nonbatched_bias, (2,0,1))
        nonbatched_bias= nonbatched_bias.unsqueeze(-4)
     
        if self.global_config.subbatch_size >0:
            msa_act = sharded_layer(
                self.attention,
                (msa_act, msa_act, bias, nonbatched_bias),
                self.global_config.subbatch_size,
                num_batch_dims=2
            )
        else:
            msa_act = self.attention(msa_act, msa_act, bias, nonbatched_bias)
        msa_act = msa_act * msa_mask.unsqueeze(-1)		
        return msa_act

class MSAColumnAttention(nn.Module):
    def __init__(self, config, global_config, msa_channel, pair_channel=None):
        super().__init__()
        self.global_config = global_config
        self.config = config
        self.pair_channel = pair_channel
        assert config.orientation == 'per_column', f'MSAColumnAttention should not with orient {config.orientation}'
        
        self.query_norm = LayerNormFP32(msa_channel)
        self.attention = Attention(config, global_config, (msa_channel, msa_channel), msa_channel)
        if pair_channel is not None:
            self.feat_2d_norm = LayerNormFP32(pair_channel)
            self.linear_2d = Linear(pair_channel, config.num_head, bias= False, initializer='linear')

    def forward(self, msa_act, msa_mask, pair_act=None):
        msa_act = torch.swapaxes(msa_act, -2, -3)
        msa_mask = torch.swapaxes(msa_mask, -1, -2)

        bias = (FP16_huge * (msa_mask - 1.))[..., None, None, :]
        bias = bias.to(msa_act.dtype)

        msa_act = self.query_norm(msa_act)
        if self.pair_channel:
            assert pair_act is not None
            pair_act = self.feat_2d_norm(pair_act)
            nonbatched_bias = self.linear_2d(pair_act)
            nonbatched_bias = permute_final_dims(nonbatched_bias, (2,0,1))
            nonbatched_bias = nonbatched_bias.unsqueeze(-4)

        if self.global_config.subbatch_size >0:
            msa_act = sharded_layer(
                self.attention,
                (msa_act, msa_act, bias, nonbatched_bias) if self.pair_channel else (msa_act, msa_act, bias)
                (msa_act, msa_act, bias),
                self.global_config.subbatch_size,
                num_batch_dims=2
            )
        else:
            msa_act = self.attention(msa_act, msa_act, bias, \
                nonbatched_bias if self.pair_channel else None)

        msa_act = torch.swapaxes(msa_act, -2, -3)
        msa_mask = torch.swapaxes(msa_mask, -1, -2)
        msa_act = msa_act * msa_mask.unsqueeze(-1)
        return msa_act


class MSAColumnGlobalAttention(nn.Module):
    def __init__(self, config, global_config, msa_channel):
        super().__init__()
        self.global_config = global_config
        self.config = config
        assert config.orientation == 'per_column', f'MSAColumnAttention should not with orient {config.orientation}'
        self.query_norm =LayerNormFP32(msa_channel)
        self.attention = GlobalAttention(config, global_config, (msa_channel, msa_channel), msa_channel)

    def forward(self, msa_act, msa_mask):
       
        msa_act = torch.swapaxes(msa_act, -2, -3)
        msa_mask = torch.swapaxes(msa_mask, -1, -2)
        msa_mask= msa_mask.to(msa_act.dtype)
        bias = (FP16_huge * (msa_mask - 1.))[..., None,:]
        bias= bias.to(msa_act.dtype)
        msa_act = self.query_norm(msa_act)
        msa_mask = msa_mask.unsqueeze(-1)
        if self.global_config.subbatch_size >0:
            msa_act = sharded_layer(
                self.attention,
                (msa_act, msa_act, msa_mask, bias),
                self.global_config.subbatch_size,
                num_batch_dims=2
            )
        else:
            msa_act = self.attention(msa_act, msa_act, msa_mask, bias)
        msa_act = torch.swapaxes(msa_act, -2, -3)
        msa_mask = torch.swapaxes(msa_mask, -2, -3)
        msa_act = msa_act * msa_mask
        return msa_act


# class Dropout(nn.Module):
#     """
#     Implementation of dropout with the ability to share the dropout mask
#     along a particular dimension.

#     If not in training mode, this module computes the identity function.
#     """

#     def __init__(self, r: float, batch_dim: Union[int, List[int]]):
#         """
#         Args:
#             r:
#                 Dropout rate
#             batch_dim:
#                 Dimension(s) along which the dropout mask is shared
#         """
#         super(Dropout, self).__init__()

#         self.r = r
#         if type(batch_dim) == int:
#             batch_dim = [batch_dim]
#         self.batch_dim = batch_dim
#         self.dropout = nn.Dropout(self.r)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             x:
#                 Tensor to which dropout is applied. Can have any shape
#                 compatible with self.batch_dim
#         """
#         shape = list(x.shape)
#         if self.batch_dim is not None:
#             for bd in self.batch_dim:
#                 shape[bd] = 1
#         mask = x.new_ones(shape)
#         mask = self.dropout(mask)
#         x = x * mask
#         return x

class Dropout(nn.Module):
    """
    Implementation of dropout with the ability to share the dropout mask
    along a particular dimension.

    If not in training mode, this module computes the identity function.
    """

    def __init__(self, r: float, batch_dim: Union[int, List[int]]):
        """
        Args:
            r:
                Dropout rate
            batch_dim:
                Dimension(s) along which the dropout mask is shared
        """
        super(Dropout, self).__init__()

        self.r = r
        if type(batch_dim) == int:
            batch_dim = [batch_dim]
        self.batch_dim = batch_dim
        self.dropout = nn.Dropout(self.r)

    def process(self, x: torch.Tensor) -> torch.Tensor:
        shape = list(x.shape)
        if self.batch_dim is not None:
            for bd in self.batch_dim:
                shape[bd] = 1
        mask = x.new_ones(shape)
        mask = self.dropout(mask)
        x = x * mask
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:
                Tensor to which dropout is applied. Can have any shape
                compatible with self.batch_dim
        """

        if isinstance(x, torch.Tensor):
            x = self.process(x)
            return x
        else:
            out_list = []
            for i in x:
                out_list.append(self.process(i))
            return out_list


class DropoutRowwise(Dropout):
    """
    Convenience class for rowwise dropout as described in subsection
    1.11.6.
    """

    __init__ = partialmethod(Dropout.__init__, batch_dim=-3)


class DropoutColumnwise(Dropout):
    """
    Convenience class for columnwise dropout as described in subsection
    1.11.6.
    """

    __init__ = partialmethod(Dropout.__init__, batch_dim=-2)
