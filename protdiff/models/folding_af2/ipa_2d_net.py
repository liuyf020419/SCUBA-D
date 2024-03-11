import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops.layers.torch import Rearrange

# from .evoformer import *
# from .layers import *
from .layers_batch import *
from .utils import checkpoint, checkpoint_sequential, MultiArgsSequential, ResModule

# import sys
# sys.path.append("/train14/superbrain/yfliu25/structure_refine/ProtDiff_new2d_inpainting_denoising/protdiff/models")
# from conv_block import STD_Resnet_block, Usequential, Upsampling
 
def build_block(
    config, 
    global_config, 
    pair_channel, 
    ):
    pair_dropout_row= config.triangle_attention_starting_node.dropout_rate
    if global_config.deterministic:
        pair_dropout_row= 0.
    modules=[]

    modules.append(
        ResModule(
            TriangleMultiplication(
                config.triangle_multiplication_outgoing, global_config, pair_channel
            ),
            DropoutRowwise(pair_dropout_row),
            input_indices=(0,1),output_index=0,
            name = "triangle_multiplication_outgoing"
        )
    )
    modules.append(
        ResModule(
            TriangleMultiplication(
                config.triangle_multiplication_incoming, global_config, pair_channel
            ),
            DropoutRowwise(pair_dropout_row),
            input_indices=(0,1),output_index=0,
            name= "triangle_multiplication_incoming"
        )
    )
    modules.append(
        ResModule(
            TriangleAttention(
                config.triangle_attention_starting_node, global_config, pair_channel
            ),
            DropoutRowwise(pair_dropout_row),
            input_indices=(0,1),output_index=0,
            name= "triangle_attention_starting_node"
        )
    )
    modules.append(
        ResModule(
            TriangleAttention(
                config.triangle_attention_ending_node, global_config, pair_channel
            ),
            DropoutColumnwise(pair_dropout_row),
            input_indices=(0,1),output_index=0,
            name= "triangle_attention_ending_node"
        )
    )
    modules.append(
        ResModule(
            Transition(
                config.pair_transition, global_config, pair_channel
            ),
            None,
            input_indices=(0,1),output_index=0,
            name= "pair_transition"
        )
    )
    return modules



class EvoformerPairBlock_(nn.Module):
    def __init__(self, config, global_config, input_channel, pair_channel):
        super().__init__()
        self.config = config
        self.global_config = global_config

        self.pair_act = Linear(input_channel, pair_channel, initializer="relu")

        evoiter_list=[]
        for _ in range(config.layers_2d):
            evoiter_list.extend(
                build_block(
                    config.evoformer_block, global_config,
                    pair_channel
                )
            )
        self.evoformer_iteration = MultiArgsSequential(*evoiter_list)

    def forward(self, pair, pair_mask):
        # import pdb; pdb.set_trace()
        pair_act = self.pair_act(pair)

        evoiter_out = checkpoint_sequential(
            self.evoformer_iteration,
            self.config.evo_former_checkpoint,
            (pair_act, pair_mask)
        )

        return evoiter_out



# class Tokenization(nn.Module):
#     def __init__(self, patch_size, input_channel, out_channel):
#         super().__init__()

#         patch_height, patch_width = patch_size
#         self.patch_height = patch_height
#         self.patch_width = patch_width

#         patch_dim = input_channel * patch_height * patch_width

#         self.to_patch_embedding = nn.Sequential(
#             Rearrange('b c (h p1) (w p2) -> b h w (p1 p2 c)', p1 = patch_height, p2 = patch_width),
#             nn.Linear(patch_dim, out_channel),
#         )

#     def forward(self, pair_act):
#         batchsize, channel, image_height, image_width = pair_act.shape
#         assert image_height % self.patch_height == 0 and image_width % self.patch_width == 0, 'Image dimensions must be divisible by the patch size.'
#         pair_token = self.to_patch_embedding(pair_act)

#         return pair_token


# class UpsamplerBlock(nn.Module):
#     def __init__(self, input_channel, channel_mult, num_res_blocks, base_channel, num_group, dropout, conv_up=True) -> None:
#         super().__init__()
#         num_resolutions = len(channel_mult)
#         self.upnets = nn.ModuleList([])
#         # Upsampling
#         for i_level in range(num_resolutions):
#             # Residual blocks for this resolution
#             for i_block in range(num_res_blocks):
#                 if i_level == 0: 
#                     in_ch = input_channel
#                 else:
#                     if i_block == 0: 
#                         in_ch = base_channel * channel_mult[i_level - 1]
#                     else: 
#                         in_ch = base_channel * channel_mult[i_level]
#                 layer = [STD_Resnet_block(in_ch=in_ch, 
#                                           out_ch=base_channel*channel_mult[i_level], 
#                                           dropout=dropout, num_group=num_group)]
#                 self.upnets.append(Usequential(*layer))
#             if i_level != num_resolutions-1:
#                 self.upnets.append(Usequential(Upsampling(base_channel * channel_mult[i_level], 
#                                                           conv_up=conv_up)))

#     def forward(self, x, temb=None):
#         for upnet in self.upnets:
#             x = upnet(x, temb)

#         return x


# class TokenEvoformerUpsampler(nn.Module):
#     def __init__(self, config, global_config):
#         super().__init__()

#         self.config = config
#         self.global_config = global_config

#         patch_size = self.config.patch_size
#         hidden_channel = self.config.hidden_channel

#         patch_height, patch_width = patch_size
#         upsampler_layer_num = patch_height//2
#         channel_mult = np.power(2, np.arange(upsampler_layer_num))

#         num_res_blocks = self.config.num_res_blocks
#         num_group = self.config.num_group
#         dropout = self.config.dropout
#         self.upsampler_block = UpsamplerBlock(
#             hidden_channel, channel_mult, num_res_blocks, hidden_channel, num_group, dropout)

#         out_channel = self.config.out_channel
#         self.out_projection = Linear(hidden_channel[-1] * hidden_channel, out_channel)

#     def forward(self, pair_token_act):
#         pair_up = self.upsampler_block(pair_token_act)
#         pair_out = self.out_projection(pair_up)

#         return pair_out


# def upsample(mat, rate):
#     B, L, L, D = mat.shape
#     assert(D % (rate*rate) == 0)
#     L1 = L * rate

#     mat1 = torch.reshape(mat, (B, L, L, rate, rate, -1))
#     mat1 = torch.permute(mat1, (0, 1, 3, 2, 4, 5))
#     mat1 = torch.reshape(mat1, (B, L1, L1, -1))
#     return mat1


# class Upsampler(nn.Module):
#     def __init__(self, config, global_config, input_channel, upsample_rate, out_channel=None):
#         super().__init__()

#         self.config = config
#         self.global_config = global_config
#         self.upsample_rate = upsample_rate
#         self.is_decoder_evoformer = self.config.is_decoder_evoformer
#         self.proj = nn.Linear(input_channel, input_channel * upsample_rate * upsample_rate)

#         if self.is_decoder_evoformer:
#             if out_channel is None:
#                 out_channel = input_channel
#             self.decoder_evoformer = EvoformerPairBlock_(
#                 self.config.decoder_evoformer, self.global_config, input_channel, out_channel)

#     def forward(self, x, evoformer_mask=None):
#         x1 = self.proj(x)
#         if not self.is_decoder_evoformer:
#             return upsample(x1, self.upsample_rate)
#         else:
#             assert evoformer_mask is not None
#             x = upsample(x1, self.upsample_rate)
#             pair_act = self.decoder_evoformer(x, evoformer_mask)[0]

#             return pair_act
        

# class TokenEvoformerEncoder(nn.Module):
#     def __init__(self, config, global_config, input_channel):
#         super().__init__()
#         self.config = config
#         self.global_config = global_config

#         patch_size = self.config.patch_size
#         self.patch_size = patch_size

#         if self.config.encoder_evoformer.layers_2d > 0:
#             encoder_channel = self.config.encoder_evoformer.pair_channel
#             self.encoder_evoformer = EvoformerPairBlock_(
#                 self.config.encoder_evoformer, self.global_config, input_channel, encoder_channel)
#         else:
#             encoder_channel = self.config.bottle_evoformer.pair_channel

#         bottle_channel = self.config.bottle_evoformer.pair_channel
#         self.bottle_channel = bottle_channel
#         self.tokenization = Tokenization(patch_size, encoder_channel, bottle_channel)

#         self.bottle_evoformer = EvoformerPairBlock_(
#             self.config.bottle_evoformer, self.global_config, bottle_channel, bottle_channel)

#     def forward(self, pair_act, pair_mask):
#         if self.config.encoder_evoformer.layers_2d > 0:
#             pair_act = self.encoder_evoformer(pair_act, pair_mask)[0]
#         pair_token = self.tokenization(pair_act.permute(0, 3, 1, 2))
#         # import pdb; pdb.set_trace()
#         pair_token_act = self.bottle_evoformer(pair_token, torch.ones_like(pair_token)[..., 0])[0]

#         return pair_token_act
