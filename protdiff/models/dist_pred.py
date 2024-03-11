import torch
import torch.nn as nn
import torch.nn.functional as F

from .folding_af2.layers_batch import Linear

class DistogramClassifier(nn.Module):
    def __init__(self, config, pair_channel) -> None:
        super().__init__()
        self.distogram_num = config.distogram_args[-1]
        self.atom3_dist = config.atom3_dist
        pred_all_dist = config.pred_all_dist
        if pred_all_dist:
            if self.atom3_dist:
                out_num = self.distogram_num * 6
            else:
                out_num = self.distogram_num
        else:
            out_num = self.distogram_num * 3 + self.distogram_num//2

        self.distogram_projection = Linear(pair_channel, out_num, initializer='linear') 

    def forward(self, act_pair):
        distogram_pair = self.distogram_projection(act_pair)
        return distogram_pair

