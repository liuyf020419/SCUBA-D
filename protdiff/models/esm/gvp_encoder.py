# Copyright (c) Facebook, Inc. and its affiliates.
#
# Contents of this file were adapted from the open source fairseq repository.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor


from .modules import SinusoidalPositionalEmbedding
from .inverse_folding.features import GVPInputFeaturizer, DihedralFeatures
from .inverse_folding.gvp_encoder import GVPEncoder
from .inverse_folding.transformer_layer import TransformerEncoderLayer


def rotate(v, R):
    """
    Rotates a vector by a rotation matrix.
    
    Args:
        v: 3D vector, tensor of shape (length x batch_size x channels x 3)
        R: rotation matrix, tensor of shape (length x batch_size x 3 x 3)

    Returns:
        Rotated version of v by rotation matrix R.
    """
    R = R.unsqueeze(-3)
    v = v.unsqueeze(-1)
    return torch.sum(v * R, dim=-2)


def get_rotation_frames(coords):
    """
    Returns a local rotation frame defined by N, CA, C positions.

    Args:
        coords: coordinates, tensor of shape (batch_size x length x 3 x 3)
        where the third dimension is in order of N, CA, C

    Returns:
        Local relative rotation frames in shape (batch_size x length x 3 x 3)
    """
    v1 = coords[:, :, 2] - coords[:, :, 1]
    v2 = coords[:, :, 0] - coords[:, :, 1]
    e1 = normalize(v1, dim=-1)
    u2 = v2 - e1 * torch.sum(e1 * v2, dim=-1, keepdim=True)
    e2 = normalize(u2, dim=-1)
    e3 = torch.cross(e1, e2, dim=-1)
    R = torch.stack([e1, e2, e3], dim=-2)
    return R


def nan_to_num(ts, val=0.0):
    """
    Replaces nans in tensor with a fixed value.    
    """
    val = torch.tensor(val, dtype=ts.dtype, device=ts.device)
    return torch.where(~torch.isfinite(ts), val, ts)


def rbf(values, v_min, v_max, n_bins=16):
    """
    Returns RBF encodings in a new dimension at the end.
    """
    rbf_centers = torch.linspace(v_min, v_max, n_bins, device=values.device)
    rbf_centers = rbf_centers.view([1] * len(values.shape) + [-1])
    rbf_std = (v_max - v_min) / n_bins
    v_expand = torch.unsqueeze(values, -1)
    z = (values.unsqueeze(-1) - rbf_centers) / rbf_std
    return torch.exp(-z ** 2)


def norm(tensor, dim, eps=1e-8, keepdim=False):
    """
    Returns L2 norm along a dimension.
    """
    return torch.sqrt(
            torch.sum(torch.square(tensor), dim=dim, keepdim=keepdim) + eps)


def normalize(tensor, dim=-1):
    """
    Normalizes a tensor along a dimension after removing nans.
    """
    return nan_to_num(
        torch.div(tensor, norm(tensor, dim=dim, keepdim=True))
    )


class GVPTransformerEncoder(nn.Module):
    """
    Transformer encoder consisting of *args.encoder.layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, args):
        super().__init__()
        self.args = args

        self.dropout_module = nn.Dropout(args.dropout)
        embed_tokens = self.build_embedding(args)

        self.embed_dim = embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = SinusoidalPositionalEmbedding(
            embed_dim,
            self.padding_idx,
        )
        self.embed_gvp_input_features = nn.Linear(15, embed_dim)
        self.embed_confidence = nn.Linear(16, embed_dim)
        self.embed_dihedrals = DihedralFeatures(embed_dim)

        self.aatype_embedding = nn.Embedding(22, embed_dim)


        gvp_args = args.gvp
        self.gvp_encoder = GVPEncoder(gvp_args)
        gvp_out_dim = gvp_args.node_hidden_dim_scalar + (3 *
                gvp_args.node_hidden_dim_vector)
        self.embed_gvp_output = nn.Linear(gvp_out_dim, embed_dim)

        self.layers = nn.ModuleList([])
        self.layers.extend(
            [self.build_encoder_layer(args) for i in range(args.encoder_layers)]
        )
        self.num_layers = len(self.layers)
        self.layer_norm = nn.LayerNorm(embed_dim)


    def build_encoder_layer(self, args):
        return TransformerEncoderLayer(args)

    def build_embedding(self, args):
        num_embeddings = args.num_aatypes
        padding_idx = args.aatype_padding_idx
        embed_dim = args.encoder_embed_dim
        emb = nn.Embedding(num_embeddings, embed_dim, padding_idx)
        nn.init.normal_(emb.weight, mean=0, std=embed_dim ** -0.5)
        nn.init.constant_(emb.weight[padding_idx], 0)
        return emb

    # def forward_embedding(self, coords, padding_mask, confidence):
    def forward_embedding(self, data, padding_mask):
        """
        Args:
            coords: N, CA, C backbone coordinates in shape length x 3 (atoms) x 3 
            padding_mask: boolean Tensor (true for padding) of shape length
            confidence: confidence scores between 0 and 1 of shape length
        """
        coords = data['coord'][..., :3, :]
        padding_mask = padding_mask
        confidence = data['confidence']
        aatype = data['aatype']
        R = data['rot']

        res_idx = data['res_idx']
        res_idx = res_idx * ~padding_mask + 5120*padding_mask

        components = dict()
        coord_mask = torch.all(torch.all(torch.isfinite(coords), dim=-1), dim=-1)
        coords = nan_to_num(coords)

        mask_tokens = aatype
        components["tokens"] = self.embed_tokens(mask_tokens) * self.embed_scale

        # actually, this is the axial angles
        components["diherals"] = self.embed_dihedrals(coords)

        # GVP encoder
        gvp_out_scalars, gvp_out_vectors = self.gvp_encoder(coords,
                coord_mask, res_idx, padding_mask, confidence)
        gvp_out_features = torch.cat([
            gvp_out_scalars,
            rotate(gvp_out_vectors, R.transpose(-2, -1)).flatten(-2, -1),
        ], dim=-1)
        components["gvp_out"] = self.embed_gvp_output(gvp_out_features)

        components["confidence"] = self.embed_confidence(
             rbf(confidence, 0., 1.))

        # In addition to GVP encoder outputs, also directly embed GVP input node
        # features to the Transformer
        scalar_features, vector_features = GVPInputFeaturizer.get_node_features(
            coords, coord_mask, with_coord_mask=False)
        features = torch.cat([
            scalar_features,
            rotate(vector_features, R.transpose(-2, -1)).flatten(-2, -1),
        ], dim=-1)
        components["gvp_input_features"] = self.embed_gvp_input_features(features)

        embed = sum(components.values())
        x = embed
        x = x + self.embed_positions(mask_tokens, positions=res_idx)
        x = self.dropout_module(x)
        return x, components 

    def forward(
        self,
        data,
        return_all_hiddens: bool = False,
    ):
        """
        Args:
            coords (Tensor): backbone coordinates
                shape batch_size x num_residues x num_atoms (3 for N, CA, C) x 3
            encoder_padding_mask (ByteTensor): the positions of
                  padding elements of shape `(batch_size x num_residues)`
            confidence (Tensor): the confidence score of shape (batch_size x
                num_residues). The value is between 0. and 1. for each residue
                coordinate, or -1. if no coordinate is given
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(num_residues, batch_size, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch_size, num_residues)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch_size, num_residues, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(num_residues, batch_size, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """

        encoder_padding_mask = ~data['encoder_padding_mask']

        x, encoder_embedding = self.forward_embedding(data, encoder_padding_mask)

        # account for padding while computing the representation
        x = x * (1 - encoder_padding_mask.unsqueeze(-1).type_as(x))

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_states = []

        if return_all_hiddens:
            encoder_states.append(x)

        # encoder layers
        for idx, layer in enumerate(self.layers):
            x = layer(
                x, encoder_padding_mask=encoder_padding_mask
            )
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [encoder_embedding],  # dictionary
            "encoder_states": encoder_states,  # List[T x B x C]
            # "out_logits": out_logits
        }
