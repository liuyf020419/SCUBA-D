import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np

esm1b_aatype_to_index = {'L': 4, 'A': 5, 'G': 6, 'V': 7, 'S': 8, 'E': 9, 'R': 10, 'T': 11, 'I': 12, 'D': 13, 'P': 14, 'K': 15, 'Q': 16, 'N': 17, 'F': 18, 'Y': 19, 'M': 20, 'H': 21, 'W': 22, 'C': 23}
esm1b_index_to_aatype = {v:k for k, v in esm1b_aatype_to_index.items()}
esm1b_aatype_index_range = [4, 24]

esm1b_aatype_to_index_from0 = {k:v-4 for k, v in esm1b_aatype_to_index.items()}
esm1b_index_to_aatype_from0 = {v:k for k, v in esm1b_aatype_to_index_from0.items()}

af2_restypes = [
    'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P',
    'S', 'T', 'W', 'Y', 'V'
]
af2_aatype_to_index = {restype: i for i, restype in enumerate(af2_restypes)}
af2_index_to_aatype = {v:k for k, v in af2_aatype_to_index.items()}

af2_index_to_esm1b_from0_index = {k: esm1b_aatype_to_index_from0[v] for k, v in af2_index_to_aatype.items()}


esm_ckpt_file = 'protdiff/models/esm_pred/esm1b_projection_head.pt'
esm_mean_std_dict = 'protdiff/models/esm_pred/esm1b_None_dict.npy'


def load_esm1b_projection_head(ckpt_file=esm_ckpt_file, mean_std_dict=esm_mean_std_dict):
    esm1b_aatype_head = Esm1bProjection(mean_std_dict=mean_std_dict)
    esm1b_aatype_head.embed_tokens.load_state_dict(torch.load(ckpt_file)['embed_tokens'])
    esm1b_aatype_head.lm_head.load_state_dict(torch.load(ckpt_file)['lm_head'])
    return esm1b_aatype_head


def calc_simi_ident_seqs(myseqs):
    from itertools import combinations
    import numpy as np

    def seqs_similarity(seq1, seq2):
        # simi_list = ["GAVLI", "FYW", "CM", "ST", "KRH", "DENQ", "P"]
        simi_list = ["AVLICM", "FYW", "ST", "KRHDENQ", "PG"]
        def pos_simi(x, simi_list):
            # print(x)
            if x[0] == x[1]:
                return [1, 1]
            elif (x[0] in simi_list[0] and x[1] in simi_list[0]) or (x[0] in simi_list[1] and x[1] in simi_list[1]) or \
                 (x[0] in simi_list[2] and x[1] in simi_list[2]) or (x[0] in simi_list[3] and x[1] in simi_list[3]) or \
                 (x[0] in simi_list[4] and x[1] in simi_list[4]):
                return [1, 0]
            else:
                return [0, 0]

        correct_list = list(map(lambda x: pos_simi(x, simi_list), zip(*[seq1, seq2])))
        simi = np.array(correct_list)[:, 0].tolist()
        ident = np.array(correct_list)[:, 1].tolist()
        return np.sum(simi) / len(simi), np.sum(ident) / len(ident)

    simi_list = []
    ident_list = []
    for comb in list(combinations(np.arange(0, len(myseqs)).tolist(), 2)):

        simi, ident = seqs_similarity(myseqs[comb[0]], myseqs[comb[1]])
        simi_list.append(simi)
        ident_list.append(ident)

    return simi_list, ident_list



def gelu(x):
    """Implementation of the gelu activation function.

    For information: OpenAI GPT's gelu is slightly different
    (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class ESM1LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12, affine=True):
        """Construct a layernorm layer in the TF style (eps inside the sqrt)."""
        super().__init__()
        self.hidden_size = (hidden_size,) if isinstance(hidden_size, int) else tuple(hidden_size)
        self.eps = eps
        self.affine = bool(affine)
        if self.affine:
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zeros(hidden_size))
        else:
            self.weight, self.bias = None, None

    def forward(self, x):
        dims = tuple(-(i + 1) for i in range(len(self.hidden_size)))
        means = x.mean(dims, keepdim=True)
        x_zeromean = x - means
        variances = x_zeromean.pow(2).mean(dims, keepdim=True)
        x = x_zeromean / torch.sqrt(variances + self.eps)
        if self.affine:
            x = (self.weight * x) + self.bias
        return x


class RobertaLMHead(nn.Module):
    """Head for masked language modeling."""

    def __init__(self, embed_dim, output_dim, weight):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = ESM1LayerNorm(embed_dim)
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, features):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)
        # project back to size of vocabulary with bias
        x = F.linear(x, self.weight) + self.bias
        return x


class Esm1bProjection(nn.Module):
    def __init__(self, alphabet_size=33, embed_dim=1280, padding_idx=1, mean_std_dict=None) -> None:
        super().__init__()  
        self.alphabet_size = alphabet_size
        self.embed_dim = embed_dim
        self.padding_idx = padding_idx
        self.mean_std_dict = mean_std_dict
        if mean_std_dict is not None:
            # import pdb; pdb.set_trace()
            self.mean_std_dict = np.load(mean_std_dict, allow_pickle=True).item()
            self.esm1b_rep_mean = torch.from_numpy(self.mean_std_dict['mean'])
            self.esm1b_rep_std = torch.from_numpy(self.mean_std_dict['std'])

        self.embed_tokens = nn.Embedding(
            self.alphabet_size, self.embed_dim, padding_idx=self.padding_idx
        )
        self.lm_head = RobertaLMHead(
            embed_dim=self.embed_dim,
            output_dim=self.alphabet_size,
            weight=self.embed_tokens.weight,
        )

    def forward(self, esm_single_rep):
        device = esm_single_rep.device
        if self.mean_std_dict is not None:
            # import pdb; pdb.set_trace()
            logits = esm_single_rep * self.esm1b_rep_std.to(device) + self.esm1b_rep_mean.to(device)
        logits = self.lm_head(esm_single_rep)

        return logits


def predict_aatype(predictor, esm1b:torch.Tensor):
    pred_logits = predictor(esm1b)[..., esm1b_aatype_index_range[0]:esm1b_aatype_index_range[1]]
    pred_aatype_esmidx = torch.argmax(pred_logits, -1).reshape((-1, )).detach().cpu().numpy()
    pred_aatype = [esm1b_index_to_aatype_from0[aa] for aa in pred_aatype_esmidx]
    return ''.join(pred_aatype)


if __name__ == "__main__":
    gt_aatype_af2idx = [19, 10, 15, 14,  0,  3, 11, 16,  2, 19, 11,  0,  0, 17,  7, 11, 19,  7,
                        0,  8,  0,  7,  6, 18,  7,  0,  6,  0, 10,  6,  1, 12, 13, 10, 15, 13,
                        14, 16, 16, 11, 16, 18, 13, 14,  8, 13,  3, 10, 15,  8,  7, 15,  0,  5,
                        19, 11,  7,  8,  7, 11, 11, 19,  0,  3,  0, 10, 16,  2,  0, 19,  0,  8,
                        19,  3,  3, 12, 14,  2,  0, 10, 15,  0, 10, 15,  3, 10,  8,  0,  8, 11,
                        10,  1, 19,  3, 14, 19,  2, 13, 11, 10, 10, 15,  8,  4, 10, 10, 19, 16,
                        10,  0,  0,  8, 10, 14,  0,  6, 13, 16, 14,  0, 19,  8,  0, 15, 10,  3,
                        11, 13, 10,  0, 15, 19, 15, 16, 19, 10, 16, 15, 11, 18]
    gt_aatype = [af2_index_to_aatype[aa] for aa in gt_aatype_af2idx]
    test_esm1b_pred = 'test_esm_pred.npy'
    esm1b_projection_head = load_esm1b_projection_head()

    pred_esm1b = np.load(test_esm1b_pred, allow_pickle=True)
    pred_logits = esm1b_projection_head(torch.from_numpy(pred_esm1b))[..., esm1b_aatype_index_range[0]:esm1b_aatype_index_range[1]]
    pred_aatype_esmidx = torch.argmax(pred_logits, -1).reshape((-1, )).detach().cpu().numpy()
    pred_aatype = [esm1b_index_to_aatype_from0[aa] for aa in pred_aatype_esmidx]

    import pdb; pdb.set_trace()
