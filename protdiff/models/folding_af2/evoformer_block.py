import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers_batch import *
from .utils import checkpoint, checkpoint_sequential, MultiArgsSequential, ResModule

def build_block(
    config,
    global_config,
    msa_channel,
    pair_channel
):
    msa_dropout= config.msa_row_attention_with_pair_bias.dropout_rate
    pair_dropout_row= config.triangle_attention_starting_node.dropout_rate
    if global_config.deterministic:
        msa_dropout= pair_dropout_row= 0.
    modules=[]
    

    modules.append(
        ResModule(
            MSARowAttentionWithPairBias(
                config.msa_row_attention_with_pair_bias, global_config, msa_channel, pair_channel
            ),
            DropoutRowwise(msa_dropout),
            input_indices=(0,2,1),output_index=0,
            name= "msa_row_attention_with_pair_bias"
        )
    )
    modules.append(
        ResModule(
            Transition(
                config.msa_transition, global_config, msa_channel
            ),
            None,
            input_indices=(0,2),output_index=0,
            name = "msa_transition"
        )
    )
    # modules.append(
    #     ResModule(
    #         OuterProductMean(
    #             config.outer_product_mean, global_config, msa_channel, pair_channel
    #         ),
    #         None,input_indices=(0,2),output_index=1,
    #         name= "outer_product_mean"
    #     )
    # )
    modules.append(
        ResModule(
            TriangleMultiplication(
                config.triangle_multiplication_outgoing, global_config, pair_channel
            ),
            DropoutRowwise(pair_dropout_row),
            input_indices=(1,3),output_index=1,
            name = "triangle_multiplication_outgoing"
        )
    )
    modules.append(
        ResModule(
            TriangleMultiplication(
                config.triangle_multiplication_incoming, global_config, pair_channel
            ),
            DropoutRowwise(pair_dropout_row),
            input_indices=(1,3),output_index=1,
            name= "triangle_multiplication_incoming"
        )
    )
    modules.append(
        ResModule(
            TriangleAttention(
                config.triangle_attention_starting_node, global_config, pair_channel
            ),
            DropoutRowwise(pair_dropout_row),
            input_indices=(1,3),output_index=1,
            name= "triangle_attention_starting_node"
        )
    )
    modules.append(
        ResModule(
            TriangleAttention(
                config.triangle_attention_ending_node, global_config, pair_channel
            ),
            DropoutColumnwise(pair_dropout_row),
            input_indices=(1,3),output_index=1,
            name= "triangle_attention_ending_node"
        )
    )
    modules.append(
        ResModule(
            Transition(
                config.pair_transition, global_config, pair_channel
            ),
            None,
            input_indices=(1,3),output_index=1,
            name= "pair_transition"
        )
    )
    return modules


class DiffEvoformer(nn.Module):
    def __init__(self, config, global_config, input_channel, hidden_channel) -> None:
        super().__init__()
        self.config = config
        self.global_config = global_config

        self.pair_act = Linear(input_channel, hidden_channel, initializer="relu")
        self.single_act = Linear(input_channel, hidden_channel, initializer="relu")

        evoiter_list=[]
        for _ in range(config.layers_2d):
            evoiter_list.extend(
                build_block(
                    config.evoformer_block, 
                    global_config,
                    hidden_channel,
                    hidden_channel
                )
            )
        self.evoformer_iteration = MultiArgsSequential(*evoiter_list)

    def forward(self, single, pair, single_mask, pair_mask):
        single_act = F.relu(self.single_act(single))
        pair_act = F.relu(self.pair_act(pair))

        evoiter_out = checkpoint_sequential(
            self.evoformer_iteration,
            self.config.evoformer_checkpoint,
            (single_act[:, None], pair_act, single_mask[:, None], pair_mask)
        )

        return evoiter_out[0][:, 0], evoiter_out[1]