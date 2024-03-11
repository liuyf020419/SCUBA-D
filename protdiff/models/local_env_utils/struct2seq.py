import numpy as np
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .self_attention import *
from .protein_features import ProteinFeatures, ProteinMPNNFeatures, ProteinMPNNFeaturesNew


class StructureTransformer(nn.Module):
    def __init__(self, node_features, edge_features, hidden_dim, 
        num_encoder_layers=3, vocab=22, k_neighbors=30, 
        protein_features='mpnn', augment_eps=0., dropout=0.1):
        super().__init__()
        
        # Hyperparameters
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim

        # Featurization layers
        self.features = ProteinMPNNFeatures(
            node_features, edge_features, top_k=k_neighbors,
            features_type=protein_features, augment_eps=augment_eps,
            dropout=dropout
        )

        # Embedding layers
        self.W_v = nn.Linear(node_features, hidden_dim, bias=True)
        self.W_e = nn.Linear(edge_features, hidden_dim, bias=True)
        # self.W_s = nn.Embedding(vocab, hidden_dim)

        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerLayer(hidden_dim, hidden_dim*2, dropout=dropout)
            for _ in range(num_encoder_layers)
        ])
        
        # Initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, X, L, mask, single_res_rel):
        """ Graph-conditioned sequence model """
        # mask [B, L]
        # Prepare node and edge embeddings
        V, E, E_idx = self.features(X, L, mask, single_res_rel)
        # import pdb; pdb.set_trace()
        h_V = self.W_v(V)
        h_E = self.W_e(E)
        # h_S = self.W_s(S)

        # h_V = torch.cat([h_V, h_S], -1)
        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1),  E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend

        hidden_list = []
        for layer in self.encoder_layers:
            h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
            # import pdb; pdb.set_trace()
            h_V = layer(h_V, h_EV, mask_attend=mask_attend)
            hidden_list.append(h_V)

        feature_dict = {
            'out_feature': h_V,
            'stacked_hidden': torch.stack(hidden_list)
        }

        return feature_dict



class MPNNEncoder(nn.Module):
    def __init__(self, node_features, edge_features, hidden_dim, 
        num_encoder_layers=3, vocab=22, k_neighbors=30, 
        protein_features='mpnn', augment_eps=0., dropout=0.1):
        super().__init__()
        
        # Hyperparameters
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim

        # Featurization layers
        self.features = ProteinMPNNFeaturesNew(
            node_features, edge_features, top_k=k_neighbors,
            features_type=protein_features, augment_eps=augment_eps,
            dropout=dropout
        )
        

        # Embedding layers
        self.W_v = nn.Linear(node_features, hidden_dim, bias=True)
        self.W_e = nn.Linear(edge_features, hidden_dim, bias=True)
        # self.W_s = nn.Embedding(vocab, hidden_dim)

        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            EncLayer(hidden_dim, hidden_dim*2, dropout=dropout)
            for _ in range(num_encoder_layers)
        ])
        
        # Initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, X, L, mask, single_res_rel):
        """ Graph-conditioned sequence model """
        # Prepare node and edge embeddings
        V, E, E_idx = self.features(X, L, mask, single_res_rel)
        # h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
        h_V = self.W_v(V)
        h_E = self.W_e(E)

        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1),  E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        hidden_list = []
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)
            # import pdb; pdb.set_trace()
            h_VE_encoder = cat_neighbors_nodes(h_V, h_E, E_idx)
            hidden_list.append(h_VE_encoder)

        feature_dict = {
            'out_feature': h_VE_encoder,
            'stacked_hidden': torch.stack(hidden_list)
        }

        return feature_dict

