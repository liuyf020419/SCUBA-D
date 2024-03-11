import numbers
from typing import Union, Sequence
import torch
from torch import nn
from torch.nn import functional as F
import math

from . import residue_constants
from . import utils

FP16_huge=2**15

def final_init(config):
    return 'zeros' if config.zero_init else 'linear'

def get_initializer_scale(initializer_name, input_shape):
    """Get Initializer for weights and scale to multiply activations by."""

    if initializer_name == 'zeros':
        stddev = 0.0
    else:
        # fan-in scaling
        scale = 1.
        for channel_dim in input_shape:
            scale /= channel_dim
        if initializer_name == 'relu':
            scale *= 2

        noise_scale = scale
        stddev = math.sqrt(noise_scale)
    return stddev


class Linear(nn.Module):
    """Protein folding specific Linear module.

    This differs from the standard Haiku Linear in a few ways:
        * It supports inputs and outputs of arbitrary rank
        * Initializers are specified by strings
    """

    def __init__(self,
                num_input: Union[int, Sequence[int]],
                num_output: Union[int, Sequence[int]],
                initializer: str = 'linear',
                num_input_dims: int = 1,
                use_bias: bool = True,
                bias_init: float = 0.,
                precision = None):
        """Constructs Linear Module.

        Args:
        num_output: Number of output channels. Can be tuple when outputting
            multiple dimensions.
        initializer: What initializer to use, should be one of {'linear', 'relu',
            'zeros'}
        num_input_dims: Number of dimensions from the end to project.
        use_bias: Whether to include trainable bias
        bias_init: Value used to initialize bias.
        precision: What precision to use for matrix multiplication, defaults
            to None.
        name: Name of module, used for name scopes.
        """
        super().__init__()
        if isinstance(num_output, numbers.Integral):
            self.output_shape = (num_output,)
        else:
            self.output_shape = tuple(num_output)
        
        if isinstance(num_output, numbers.Integral):
            self.input_shape = (num_input,)
        else:
            self.input_shape = tuple(num_input)

        self.initializer = initializer
        self.use_bias = use_bias
        self.bias_init = bias_init
        self.num_input_dims = num_input_dims
        self.num_output_dims = len(self.output_shape)
        self.precision = precision

        stddev = get_initializer_scale(initializer, self.input_shape)
        in_letters = 'abcde'[:self.num_input_dims]
        out_letters = 'hijkl'[:self.num_output_dims]
        self.equation = f'...{in_letters}, {in_letters}{out_letters}->...{out_letters}'

        weight_shape = self.input_shape + self.output_shape
        init_weights = torch.randn(weight_shape) * stddev
        self.weights = nn.Parameter(init_weights, requires_grad=True)

        if use_bias:
            init_bias = torch.ones(self.output_shape) * bias_init
            self.bias = nn.Parameter(init_bias, requires_grad=True)

    def forward(self, x):
        output = torch.einsum(self.equation, x, self.weights)
        if self.use_bias:
            output = output + self.bias
        return output


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
        assert key_dim % num_head == 0
        assert value_dim % num_head == 0
        key_dim = key_dim // num_head
        value_dim = value_dim // num_head

        self.key_scale = key_dim ** (-0.5)

        q_weights = torch.zeros((q_dim, num_head, key_dim))
        nn.init.xavier_uniform_(q_weights)
        self.query_w = nn.Parameter(data=q_weights, requires_grad=True)

        k_weights = torch.zeros((m_dim, num_head, key_dim))
        nn.init.xavier_uniform_(k_weights)
        self.key_w = nn.Parameter(data=k_weights, requires_grad=True)

        v_weights = torch.zeros((m_dim, num_head, value_dim))
        nn.init.xavier_uniform_(v_weights)
        self.value_w = nn.Parameter(data=v_weights, requires_grad=True)

        if config.gating:
            gating_weights = torch.zeros((q_dim, num_head, value_dim))
            self.gating_w = nn.Parameter(data=gating_weights, requires_grad=True)
            gating_bias = torch.ones((num_head, value_dim))
            self.gating_b = nn.Parameter(data=gating_bias, requires_grad=True)
        
        o_weights = torch.zeros((num_head, value_dim, output_dim))
        if not global_config.zero_init:
            nn.init.xavier_uniform_(o_weights)
        self.output_w = nn.Parameter(data=o_weights, requires_grad=True)
        o_bias = torch.zeros((output_dim,))
        self.output_b = nn.Parameter(data=o_bias, requires_grad=True)

    def forward(self, q_data, m_data, bias, nonbatched_bias=None, batched_bias=None):
        """Builds Attention module.

        Arguments:
        q_data: A tensor of queries, shape [batch_size, N_queries, q_channels].
        m_data: A tensor of memories from which the keys and values are
            projected, shape [batch_size, N_keys, m_channels].
        bias: A bias for the attention, shape [batch_size, N_queries, N_keys].
        nonbatched_bias: Shared bias, shape [N_queries, N_keys].

        Returns:
        A float32 tensor of shape [batch_size, N_queries, output_dim].
        """
        # import pdb; pdb.set_trace()
        # q = torch.einsum('bqwa,ahc->bqwhc', q_data, self.query_w) * self.key_scale
        # k = torch.einsum('bkwa,ahc->bkwhc', m_data, self.key_w)
        # v = torch.einsum('bkwa,ahc->bkwhc', m_data, self.value_w)

        # logits = torch.einsum('bqwhc,bkwhc->bhqk', q, k) + bias

        q = torch.einsum('bqa,ahc->bqhc', q_data, self.query_w) * self.key_scale
        k = torch.einsum('bka,ahc->bkhc', m_data, self.key_w)
        v = torch.einsum('bka,ahc->bkhc', m_data, self.value_w)

        logits = torch.einsum('bqhc,bkhc->bhqk', q, k) + bias
        
        if nonbatched_bias is not None:
            logits = logits + nonbatched_bias.unsqueeze(0)
        if nonbatched_bias is not None:
            logits = logits + batched_bias
        weights = F.softmax(logits, dim=-1)

        weighted_avg = torch.einsum('bhqk,bkhc->bqhc', weights, v)

        if self.config.gating:
            gate_values = torch.einsum('bqc, chv->bqhv', q_data, self.gating_w) + self.gating_b
            gate_values = torch.sigmoid(gate_values)
            weighted_avg = weighted_avg * gate_values

        output = torch.einsum('bqhc,hco->bqo', weighted_avg, self.output_w) + self.output_b

        return output



class AttentionBatch(nn.Module):
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
        assert key_dim % num_head == 0
        assert value_dim % num_head == 0
        key_dim = key_dim // num_head
        value_dim = value_dim // num_head

        self.key_scale = key_dim ** (-0.5)

        q_weights = torch.zeros((q_dim, num_head, key_dim))
        nn.init.xavier_uniform_(q_weights)
        self.query_w = nn.Parameter(data=q_weights, requires_grad=True)

        k_weights = torch.zeros((m_dim, num_head, key_dim))
        nn.init.xavier_uniform_(k_weights)
        self.key_w = nn.Parameter(data=k_weights, requires_grad=True)

        v_weights = torch.zeros((m_dim, num_head, value_dim))
        nn.init.xavier_uniform_(v_weights)
        self.value_w = nn.Parameter(data=v_weights, requires_grad=True)

        if config.gating:
            gating_weights = torch.zeros((q_dim, num_head, value_dim))
            self.gating_w = nn.Parameter(data=gating_weights, requires_grad=True)
            gating_bias = torch.ones((num_head, value_dim))
            self.gating_b = nn.Parameter(data=gating_bias, requires_grad=True)
        
        o_weights = torch.zeros((num_head, value_dim, output_dim))
        if not global_config.zero_init:
            nn.init.xavier_uniform_(o_weights)
        self.output_w = nn.Parameter(data=o_weights, requires_grad=True)
        o_bias = torch.zeros((output_dim,))
        self.output_b = nn.Parameter(data=o_bias, requires_grad=True)

    def forward(self, q_data, m_data, bias, nonbatched_bias=None, batched_bias=None):
        """Builds Attention module.

        Arguments:
        q_data: A tensor of queries, shape [batch_size, N_queries, q_channels].
        m_data: A tensor of memories from which the keys and values are
            projected, shape [batch_size, N_keys, m_channels].
        bias: A bias for the attention, shape [batch_size, N_queries, N_keys].
        nonbatched_bias: Shared bias, shape [N_queries, N_keys].

        Returns:
        A float32 tensor of shape [batch_size, N_queries, output_dim].
        """
        # import pdb; pdb.set_trace()
        q = torch.einsum('bqwa,ahc->bqwhc', q_data, self.query_w) * self.key_scale
        k = torch.einsum('bkwa,ahc->bkwhc', m_data, self.key_w)
        v = torch.einsum('bkwa,ahc->bkwhc', m_data, self.value_w)

        logits = torch.einsum('bqwhc,bkwhc->bhqk', q, k) + bias
        
        if nonbatched_bias is not None:
            logits = logits + nonbatched_bias.unsqueeze(0)
        if nonbatched_bias is not None:
            logits = logits + batched_bias
        weights = F.softmax(logits, dim=-1)

        weighted_avg = torch.einsum('bhqk,bkhc->bqhc', weights, v)

        if self.config.gating:
            gate_values = torch.einsum('bqc, chv->bqhv', q_data, self.gating_w) + self.gating_b
            gate_values = torch.sigmoid(gate_values)
            weighted_avg = weighted_avg * gate_values

        output = torch.einsum('bqhc,hco->bqo', weighted_avg, self.output_w) + self.output_b

        return output


class GlobalAttention(nn.Module):
    """Multihead attention."""
    def __init__(self, config, global_config, qm_dims, output_dim):
        super().__init__()
        self.config = config
        self.global_config = global_config
        self.output_dim = output_dim

        q_dim, m_dim = qm_dims
        key_dim = self.config.get('key_dim', q_dim)
        value_dim = self.config.get('value_dim',m_dim)

        num_head = self.config.num_head
        assert key_dim % num_head == 0
        assert value_dim % num_head == 0
        key_dim = key_dim // num_head
        value_dim = value_dim // num_head

        self.key_scalar = key_dim ** (-0.5)

        q_weights = torch.rand(q_dim, num_head, key_dim)
        nn.init.xavier_uniform_(q_weights)
        self.query_w = nn.Parameter(data=q_weights, requires_grad=True)

        k_weights = torch.rand(m_dim, key_dim)
        nn.init.xavier_uniform_(k_weights)
        self.key_w = nn.Parameter(data=k_weights, requires_grad=True)

        v_weights = torch.rand(m_dim, value_dim)
        nn.init.xavier_uniform_(v_weights)
        self.value_w = nn.Parameter(data=v_weights, requires_grad=True)

        if config.gating:
            gating_weights = torch.zeros((q_dim, num_head, value_dim))
            self.gating_w = nn.Parameter(data=gating_weights, requires_grad=True)
            gating_bias = torch.ones((num_head, value_dim))
            self.gating_b = nn.Parameter(data=gating_bias, requires_grad=True)
        
        o_weights = torch.zeros((num_head, value_dim, output_dim))
        if not global_config.zero_init:
            nn.init.xavier_uniform_(o_weights)
        self.output_w = nn.Parameter(data=o_weights, requires_grad=True)
        o_bias = torch.zeros((output_dim,))
        self.output_b = nn.Parameter(data=o_bias, requires_grad=True)

    def forward(self, q_data, m_data, q_mask, bias):
        """Builds Attention module.

        Arguments:
        q_data: A tensor of queries, shape [batch_size, N_queries, q_channels].
        m_data: A tensor of memories from which the keys and values are
            projected, shape [batch_size, N_keys, m_channels].
        bias: A bias for the attention, shape [batch_size, N_queries, N_keys].
        nonbatched_bias: Shared bias, shape [N_queries, N_keys].

        Returns:
        A float32 tensor of shape [batch_size, N_queries, output_dim].
        """
        
        v = torch.einsum('bka,ac->bkc', m_data, self.value_w)
        q_avg = utils.mask_mean(q_mask, q_data, dims=[1])

        q = torch.einsum('ba,ahc->bhc', q_avg, self.query_w) * self.key_scalar
        k = torch.einsum('bka,ac->bkc', m_data, self.key_w)
        bias = (FP16_huge * (q_mask[:, None, :, 0] - 1.))
        logits = torch.einsum('bhc,bkc->bhk', q, k) + bias

        weights = F.softmax(logits, dim=-1)
        weighted_avg = torch.einsum('bhk,bkc->bhc', weights, v)

        if self.config.gating:
            gate_values = torch.einsum('bqc, chv->bqhv', q_data, self.gating_w)
            gate_values = torch.sigmoid(gate_values + self.gating_b)
            weighted_avg = weighted_avg[:, None] * gate_values
            output = torch.einsum('bqhc,hco->bqo', weighted_avg, self.output_w) + self.output_b
        else:
            output = torch.einsum('bhc,hco->bo', weighted_avg, self.output_w) + self.output_b
            output = output[:, None]
        
        return output


class TriangleAttention(nn.Module):
    """Triangle Attention.

    Jumper et al. (2021) Suppl. Alg. 13 "TriangleAttentionStartingNode"
    Jumper et al. (2021) Suppl. Alg. 14 "TriangleAttentionEndingNode"
    """
    def __init__(self, config, global_config, num_channels):
        super().__init__()
        self.config = config
        self.global_config = global_config

        self.query_norm = nn.LayerNorm(num_channels)
        weights = torch.randn((num_channels, config.num_head)) * 1. / math.sqrt(num_channels)
        self.feat_2d_weights = nn.Parameter(data=weights, requires_grad=True)
        value_dim = config.value_dim if hasattr(config, 'value_dim') else num_channels
        qm_dims = (value_dim, value_dim)
        self.attention = Attention(config, global_config, qm_dims, num_channels)

    def forward(self, pair_act, pair_mask):
        c = self.config

        assert len(pair_act.shape) == 4
        assert len(pair_mask.shape) == 3
        assert c.orientation in ['per_row', 'per_column']
      
        if c.orientation == 'per_column':
            pair_act = torch.swapaxes(pair_act, -1, -2)
            pair_mask = torch.swapaxes(pair_mask, -1, -2)
        
        bias = (FP16_huge * (pair_mask - 1.))[:, :, :, None]
        assert len(bias.shape) == 4

        pair_act = self.query_norm(pair_act)
        import pdb; pdb.set_trace()
        batched_bias = torch.einsum('...qkc,ch->...hqk', pair_act, self.feat_2d_weights)

        pair_act = self.attention(pair_act, pair_act, bias, batched_bias)

        if c.orientation == 'per_column':
            pair_act = torch.swapaxes(pair_act, -2, -3)

        return pair_act


class TriangleMultiplication(nn.Module):
    def __init__(self, config, global_config, input_dim):
        super().__init__()
        self.config = config
        self.global_config = global_config

        self.layer_norm_input = nn.LayerNorm(input_dim)
        self.center_layer_norm = nn.LayerNorm(input_dim)
        self.left_projection = Linear(input_dim, config.num_intermediate_channel)
        self.right_projection = Linear(input_dim, config.num_intermediate_channel)
        self.left_gate = Linear(
            input_dim, config.num_intermediate_channel,
            initializer=final_init(global_config),
            bias_init=1.
        )
        self.right_gate = Linear(
            input_dim, config.num_intermediate_channel,
            initializer=final_init(global_config),
            bias_init=1.
        )
        self.output_projection = Linear(
            config.num_intermediate_channel, input_dim,
            initializer=final_init(global_config),
        )
        self.gating_linear = Linear(
            config.num_intermediate_channel, input_dim, 
            initializer=final_init(global_config),
            bias_init=1.
        )

    def forward(self, act, mask):
        c = self.config
        gc = self.global_config

        mask = mask[..., None]
        act = self.layer_norm_input(act)
        input_act = act
        left_proj_act = mask * self.left_projection(act)
        right_proj_act = mask * self.right_projection(act)
        left_gate_values = torch.sigmoid(self.left_gate(act))
        right_gate_values = torch.sigmoid(self.right_gate(act))

        left_proj_act = left_proj_act * left_gate_values
        right_proj_act = right_proj_act * right_gate_values

        act = torch.einsum(c.equation, left_proj_act, right_proj_act)
        act = self.center_layer_norm(act)
        act = self.output_projection(act)
        gate_values = torch.sigmoid(self.gating_linear(input_act))

        act = act * gate_values
        return act


class Transition(nn.Module):
    def __init__(self, config, global_config, input_dim):
        super().__init__()
        self.config = config
        self.global_config = global_config

        num_intermediate = int(input_dim * self.config.num_intermediate_factor)
        
        self.input_layer_norm = nn.LayerNorm(input_dim)
        self.transition1 = Linear(input_dim, num_intermediate, initializer='relu')
        self.transition2 = Linear(num_intermediate, input_dim, initializer=final_init(global_config),)
    
    def forward(self, act, mask):
        mask = mask.unsqueeze(-1)
        act = self.input_layer_norm(act)
        act = self.transition1(act)
        act = F.relu(act)
        act = self.transition2(act)
        return act


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

        self.layer_norm_input = nn.LayerNorm(num_input_channel)

        self.left_projection = Linear(num_input_channel, config.num_outer_channel, initializer='linear')
        self.right_projection = Linear(num_input_channel, config.num_outer_channel, initializer='linear')

        output_weight = torch.zeros(
            (config.num_outer_channel, config.num_outer_channel, num_output_channel)
        )
        if not global_config.zero_init:
            nn.init.xavier_uniform_(output_weight, 2.0)
        self.output_w = nn.Parameter(data = output_weight, requires_grad=True)

        output_bias = torch.zeros((num_output_channel,))
        self.output_b = nn.Parameter(data = output_bias, requires_grad=True)

    def forward(self, act, mask):
        gc = self.global_config
        c = self.config
        import pdb; pdb.set_trace()
        mask = mask[..., None]
        act = self.layer_norm_input(act)
        left_act = mask * self.left_projection(act)
        right_act = mask * self.right_projection(act)

        def compute_chunk(left_act):
            # This is equivalent to
            #
            # act = jnp.einsum('abc,ade->dceb', left_act, right_act)
            # act = jnp.einsum('dceb,cef->bdf', act, output_w) + output_b
            #
            # but faster.
            # left_act = torch.permute(left_act, [0, 2, 1])
            left_act = left_act.permute(0, 2, 1)
            # ncs,nsc->sccs
            act = torch.einsum('acb,ade->dceb', left_act, right_act)
            # sccs,cch->ssh
            act = torch.einsum('dceb,cef->dbf', act, self.output_w) + self.output_b
            return act.permute(1, 0, 2) # torch.permute(act, [1, 0, 2])

        act = compute_chunk(left_act)
        epsilon = 1e-3
        # import pdb;pdb.set_trace()
        norm = torch.einsum('abc,adc->bdc', mask, mask)
        act /= epsilon + norm

        return act


class MSARowAttentionWithPairBias(nn.Module):
    def __init__(self, config, global_config, msa_channel, pair_channel):
        super().__init__()
        self.global_config = global_config
        self.config = config

        self.query_norm = nn.LayerNorm(msa_channel)
        self.feat_2d_norm = nn.LayerNorm(pair_channel)

        weights = torch.randn((pair_channel, config.num_head)) * 1. / math.sqrt(pair_channel)
        self.feat_2d_weights = nn.Parameter(data = weights, requires_grad=True)

        self.attention = Attention(
            config, global_config, (msa_channel, msa_channel),  msa_channel
        )

    def forward(self, msa_act, msa_mask, pair_act):
        c = self.config
        assert len(msa_act.shape) == 3
        assert len(msa_mask.shape) == 2
        assert c.orientation == 'per_row'

        bias = (FP16_huge * (msa_mask - 1.))[:, None, None, :]
        assert len(bias.shape) == 4
       
        msa_act = self.query_norm(msa_act)
        pair_act = self.feat_2d_norm(pair_act)

        nonbatched_bias = torch.einsum('qkc,ch->hqk', pair_act, self.feat_2d_weights)
        msa_act = self.attention(msa_act, msa_act, bias, nonbatched_bias)

        return msa_act

class MSAColumnAttention(nn.Module):
    def __init__(self, config, global_config, msa_channel):
        super().__init__()
        self.global_config = global_config
        self.config = config
        self.query_norm = nn.LayerNorm(msa_channel)
        self.attention = Attention(config, global_config, (msa_channel, msa_channel), msa_channel)

    def forward(self, msa_act, msa_mask):

        c = self.config
        assert len(msa_act.shape) == 3
        assert len(msa_mask.shape) == 2
        assert c.orientation == 'per_column'

        # msa_act = torch.swapaxes(msa_act, -2, -3)
        msa_act = torch.swapaxes(msa_act, -1, -2)
        msa_mask = torch.swapaxes(msa_mask, -1, -2)

        bias = (FP16_huge * (msa_mask - 1.))[None, None, :, :]
        assert len(bias.shape) == 4

        msa_act = self.query_norm(msa_act)
        msa_act = self.attention(msa_act, msa_act, bias)

        msa_act = torch.swapaxes(msa_act, -1, -2)

        return msa_act


class MSAColumnGlobalAttention(nn.Module):
    def __init__(self, config, global_config, msa_channel):
        super().__init__()
        self.global_config = global_config
        self.config = config

        self.query_norm = nn.LayerNorm(msa_channel)
        self.attention = GlobalAttention(config, global_config, (msa_channel, msa_channel), msa_channel)

    def forward(self, msa_act, msa_mask):
        c = self.config

        assert len(msa_act.shape) == 3
        assert len(msa_mask.shape) == 2
        assert c.orientation == 'per_column'

        msa_act = torch.swapaxes(msa_act, -2, -3)
        msa_mask = torch.swapaxes(msa_mask, -1, -2)

        bias = (FP16_huge * (msa_mask - 1.))[:, None, None, :]
        assert len(bias.shape) == 4

        msa_act = self.query_norm(msa_act)
        msa_mask = msa_mask.unsqueeze(-1)
        msa_act = self.attention(msa_act, msa_act, msa_mask, bias)
        msa_act = torch.swapaxes(msa_act, -2, -3)

        return msa_act


