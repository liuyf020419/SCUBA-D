import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from .diff_encoder_module import DiffPairEncoder, DiffSingleEncoder
from .protein_geom_utils import generate_pair_from_pos, preprocess_pair_feature_advance
from .protein_utils.rigid import affine_to_frame12, affine_to_pos

from .folding_af2.evoformer_block import DiffEvoformer
from .folding_af2.ipa_rigid_net import StructureModule
from .folding_af2.layers_batch import Linear

logger = logging.getLogger(__name__)


class EvoformerIPA(nn.Module):
    def __init__(self, config, global_config) -> None:
        super().__init__()
        self.config = config
        self.global_config = global_config

        self.single_encoder = DiffSingleEncoder(self.config.single_encoder, self.global_config)
        self.pair_encoder = DiffPairEncoder(self.config.pair_encoder, self.global_config)

        self.evoformer = DiffEvoformer(
            self.config.evoformer, self.global_config, 
            self.single_encoder.single_channel, self.pair_encoder.pair_channel)
        self.IPA_module = StructureModule(self.config.structure_module, self.global_config)

        self.esm_predictor = nn.Sequential(
            Linear(self.single_encoder.single_channel, global_config.esm_num, initializer='relu'),
            nn.ReLU(),
            nn.LayerNorm(global_config.esm_num),
            Linear(global_config.esm_num, global_config.esm_num))

        if self.global_config.loss_weight.diff_esm_single_ce_loss > 0.0:
            self.esm_ce_head = nn.Sequential(
                Linear(global_config.esm_num, self.single_encoder.single_channel, initializer='relu'),
                nn.ReLU(),
                nn.LayerNorm(self.single_encoder.single_channel),
                Linear(self.single_encoder.single_channel, 20))

        self.distogram_predictor = self.build_distogram_predictor()
        self.distogram_aux_predictor = self.build_distogram_aux_predictor()


    def forward(self, batch):
        xt_dict = batch['xt_dict']
        act_single = self.single_encoder(batch, xt_dict['esm'])
        batch_size, res_num = xt_dict['affine'].shape[:2]

        geom_pair = generate_pair_from_pos(affine_to_pos(xt_dict['affine'].reshape(-1, 7)).reshape(batch_size, res_num, -1, 3))
        pair_config = self.config.pair_encoder
        geom_pair = preprocess_pair_feature_advance(
            geom_pair, rbf_encode=pair_config.rbf_encode, 
            num_rbf=pair_config.num_rbf, 
            tri_encode=pair_config.tri_encode, 
            tri_num=pair_config.tri_num)
        act_pair = self.pair_encoder(batch, geom_pair)

        single_rep, pair_rep = self.evoformer(act_single, act_pair, batch['seq_mask'], batch['pair_mask'])
        # import pdb; pdb.set_trace()
        frame = affine_to_frame12(xt_dict['affine'])[:, :, None, :]
        representations = {
            "single": torch.nan_to_num(single_rep),
            "pair": pair_rep,
            "frame": frame,
            "seq_mask": batch["seq_mask"],
            "pair_mask": batch["pair_mask"]
            }
        # import pdb; pdb.set_trace()
        pred_dict = self.IPA_module(representations=representations)

        pred_esm = self.esm_predictor(single_rep)
        pred_dict.update({'esm': pred_esm})

        if self.global_config.loss_weight.diff_esm_single_ce_loss > 0.0:
            pred_aatype = self.esm_ce_head(pred_esm)
            pred_dict.update({'aatype': pred_aatype})

        pred_distogram = self.distogram_predictor(pair_rep)
        pred_dict.update({'distogram': pred_distogram})

        pred_aux_distogram = self.distogram_aux_predictor(pair_rep)
        pred_dict.update({'aux_distogram': pred_aux_distogram})

        return pred_dict


    def build_distogram_predictor(self):
        out_num = self.config.distogram_pred.distogram_args[-1]
        distogram_predictor = nn.Sequential(
            Linear(self.pair_encoder.pair_channel, out_num, initializer='relu'),
            nn.ReLU(),
            Linear(out_num, out_num))
        
        return distogram_predictor


    def build_distogram_aux_predictor(self):
        distogram_num = self.config.distogram_pred.distogram_args[-1]
        out_num = distogram_num * 5

        distogram_aux_predictor = nn.Sequential(
            Linear(self.pair_encoder.pair_channel, out_num, initializer='relu'),
            nn.ReLU(),
            Linear(out_num, out_num))
        
        return distogram_aux_predictor

