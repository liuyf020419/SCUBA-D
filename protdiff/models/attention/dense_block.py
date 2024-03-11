import math
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F


NOISE_SCALE = 5000

logger = logging.getLogger(__name__)

class TransformerPositionEncoding(nn.Module):
    def __init__(self, max_len, d_model):
        super(TransformerPositionEncoding, self).__init__()

        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len)
        half_dim = d_model // 2
        ## emb.shape (hald_dim, )
        emb = torch.exp(torch.arange(half_dim) * -(math.log(10000) / half_dim))
        # Compute the positional encodings once in log space.
        pe[: ,: half_dim] = torch.sin(position[:, None] * emb)
        pe[: ,half_dim: ] = torch.cos(position[:, None] * emb)

        self.register_buffer("pe", pe, persistent=True)

    def forward(self, timesteps, index_select=False):
        """
        return [:seqlen, d_model]
        """
        if not index_select:
            assert len(timesteps.shape) == 1
            return self.pe[:timesteps.shape[0]]
        else:
            B, L = timesteps.shape
            return self.pe[timesteps.reshape(-1, 1)].reshape(B, L, self.d_model)


class ContinousNoiseSchedual(nn.Module):
    """
    noise.shape (batch_size, )
    """
    def __init__(self, d_model):
        super(ContinousNoiseSchedual, self).__init__()

        half_dim = d_model // 2
        emb = math.log(10000) / float(half_dim - 1)
        emb = torch.exp(torch.arange(half_dim) * -emb)
        # emb.shape (half_dim, )
        self.register_buffer("emb", emb, persistent=True)

    def forward(self, noise):
        """
        noise [B, 1]
        return [:seqlen, d_model]
        """
        if len(noise.shape) > 1:
            noise = noise.squeeze(-1)
        assert len(noise.shape) == 1

        exponents = NOISE_SCALE * noise[:, None] * self.emb[None, :]
        return torch.cat([exponents.sin(), exponents.cos()], dim=-1)


class FeaturewiseAffine(nn.Module):
    def __init__(self):
        super(FeaturewiseAffine, self).__init__()

    def forward(self, x, scale, shift):
        return x * scale + shift


class DenseResBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(DenseResBlock, self).__init__()

        self.ln1 = nn.LayerNorm(in_dim)
        self.FA1 = FeaturewiseAffine()
        self.swish1 = nn.SiLU()
        self.dense1 = nn.Linear(in_dim, out_dim)

        self.ln2 = nn.LayerNorm(out_dim)
        self.FA2 = FeaturewiseAffine()
        self.swish2 = nn.SiLU()
        self.dense2 = nn.Linear(out_dim, out_dim)

        self.skipbranch = nn.Linear(in_dim, out_dim)

    def forward(self, x, scale = 1., shift = 0.):
        """
        x.shape [B, L, mlp_dim]
        scale.shape [B, 1, mlp_dim]
        shift.shape [B, 1, mlp_dim]
        """
        input = x

        x = self.ln1(x)
        x = self.swish1(self.FA1(self.ln1(x), scale, shift))
        x = self.dense1(x)
        x = self.ln2(x)
        x = self.swish2(self.FA2(self.ln1(x), scale, shift))
        x = self.dense2(x)

        return x + self.skipbranch(input)


class DenseFiLM(nn.Module):
    def __init__(self, emb_dim, out_dim, sequence = False):
        super(DenseFiLM, self).__init__()
        self.sequence = sequence
        self.noiselayer = ContinousNoiseSchedual(emb_dim)
        self.branch = nn.Sequential(nn.Linear(emb_dim, 4*emb_dim),
                                    nn.SiLU(),
                                    nn.Linear(4*emb_dim, 4*emb_dim))

        self.scale_layer = nn.Linear(4*emb_dim, out_dim)
        self.shift_layer = nn.Linear(4*emb_dim, out_dim)

    def forward(self, t):
        """
        t.shape [B, 1, 1]
        """
        assert len(t.shape) == 3
        t = t.squeeze(-1)
        t = self.branch(self.noiselayer(t))
        if self.sequence == True:
            t = t[: ,None ,: ]

        scale = self.scale_layer(t)
        shift = self.shift_layer(t)

        return scale, shift


############################# Transformer ############################

class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k,v in kwargs.items():
            setattr(self, k, v)


class SelfAttention(nn.Module):
    """
    A vanilla multi-head self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head

    def forward(self, x, mask=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if mask is not None:
            att = att.masked_fill(mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class GPTBlock(nn.Module):
    """ an unassuming Transformer block """
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = SelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),  # nice
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size=20, block_size=128, n_layer=12, n_head=8, n_embd=256,
                 embd_pdrop=0.1, resid_pdrop=0.1, attn_pdrop=0.1, n_unmasked=0):
        super(GPT, self).__init__()

        ### minGPT
        config = GPTConfig(block_size=block_size, vocab_size=vocab_size,
                           embd_pdrop=embd_pdrop, resid_pdrop=resid_pdrop, attn_pdrop=attn_pdrop,
                           n_layer=n_layer, n_head=n_head, n_embd=n_embd, n_unmasked=n_unmasked)
        # input embedding stem
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer
        self.gptblocks = nn.Sequential(*[GPTBlock(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        # self.head = nn.Linear(config.n_embd, config.n_output, bias=False)
        self.block_size = config.block_size
        self.apply(self._init_weights)
        self.config = config

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, embeddings):
        # forward the GPT model
        x = self.drop(embeddings)
        x = self.gptblocks(x)
        x = self.ln_f(x)
        # logits = self.head(x)

        # return logits
        return x

############################# scale & shift encoder output ############################

class DenseEncoderOut(nn.Module):
    def __init__(self, n_embd, mlp_dims, sequence = True):
        super(DenseEncoderOut, self).__init__()
        self.densefilm = DenseFiLM(emb_dim=n_embd, out_dim=mlp_dims, sequence=sequence)
        self.denseres = DenseResBlock(n_embd, mlp_dims)

    def forward(self, x, t):
        """
        t.shape (B, 1, 1)
        """
        scale, shift = self.densefilm(t)
        x = self.denseres(x, scale, shift)

        return x


class BB_embedding(nn.Module):
    def __init__(self, n_embd=128, vocab_emb=True, torsion_bin=1, joint_phipsi=False,
                 joint_nbins=None, bb_func_dim=1, triangle_encode=False):
        super(BB_embedding, self).__init__()

        self.joint_phipsi = joint_phipsi
        self.bb_func_dim = bb_func_dim
        self.vocab_emb = vocab_emb
        self.triangle_encode = triangle_encode

        if vocab_emb:
            assert isinstance(torsion_bin, int)
            n_bins = 360 // torsion_bin
            if joint_phipsi:
                assert isinstance(joint_nbins, int)
                self.phi_psi_bb_tors = nn.Embedding(joint_nbins + 1, n_embd)
                self.omega_bb_tors = nn.Embedding(n_bins + 1, n_embd)
            else:
                self.phi_bb_tors = nn.Embedding(n_bins + 1, n_embd)
                self.psi_bb_tors = nn.Embedding(n_bins + 1, n_embd)
                self.omega_bb_tors = nn.Embedding(n_bins + 1, n_embd)
        else:
            if self.triangle_encode:
                assert isinstance(bb_func_dim, int)
                self.phi_bb_tors = nn.Linear(bb_func_dim * 2, n_embd, bias=False)
                self.psi_bb_tors = nn.Linear(bb_func_dim * 2, n_embd, bias=False)
                self.omega_bb_tors = nn.Linear(bb_func_dim * 2, n_embd, bias=False)

            else:
                self.bb_tors = nn.Linear(3, n_embd, bias=False)

    def forward(self, bbs_inf):
        if self.vocab_emb:
            if self.joint_phipsi:
                joint_emb = self.phi_psi_bb_tors(bbs_inf[:, :, 0])
                omega_emb = self.omega_bb_tors(bbs_inf[:, :, 1])
                embedinigs = torch.cat([joint_emb, omega_emb], dim=2)
            else:
                phi_emb = self.phi_bb_tors(bbs_inf[:, :, 0])
                psi_emb = self.psi_bb_tors(bbs_inf[:, :, 1])
                omega_emb = self.omega_bb_tors(bbs_inf[:, :, 2])
                embedinigs = torch.cat([phi_emb, psi_emb, omega_emb], dim=2)
        else:
            if self.triangle_encode:
                phi_emb = self.phi_bb_tors(bbs_inf[:, :, : 2*self.bb_func_dim])
                psi_emb = self.psi_bb_tors(bbs_inf[:, :, 2 * self.bb_func_dim: 2 * 2 * self.bb_func_dim])
                omega_emb = self.omega_bb_tors(bbs_inf[:, :, 2 * 2 * self.bb_func_dim: 3 * 2 * self.bb_func_dim])
                embedinigs = torch.cat([phi_emb, psi_emb, omega_emb], dim=2)

            else:
                embedinigs = self.bb_tors(bbs_inf)

        return embedinigs

############################# Diffusion model ############################

class transffusion(nn.Module):
    def __init__(self, d_model: int, mlp_dims: int, num_mlp_layers: int, data_channels: int, vocab_emb ,
                 torsion_bin, joint_phipsi , joint_nbins, bb_func_dim, triangle_encode,
                 max_len = 5000, n_layer=12, n_head=8, embd_pdrop=0.1, resid_pdrop=0.1, attn_pdrop=0.1, n_unmasked=0):
        super(transffusion, self).__init__()
        if triangle_encode:
            model_nembd = 2 * d_model if joint_phipsi else 3 * d_model
        else:
            model_nembd = d_model

        self.BB_embd = BB_embedding(n_embd=d_model, vocab_emb=vocab_emb, torsion_bin=torsion_bin,
                                    joint_phipsi=joint_phipsi, joint_nbins=joint_nbins, bb_func_dim=bb_func_dim,
                                    triangle_encode=triangle_encode)

        self.PositionEncoding = TransformerPositionEncoding(max_len=max_len, d_model=model_nembd)

        self.Transformer = GPT(n_layer=n_layer, n_head=n_head, n_embd=model_nembd, embd_pdrop=embd_pdrop,
                               resid_pdrop=resid_pdrop, attn_pdrop=attn_pdrop, n_unmasked=n_unmasked)

        self.interlayer = nn.Sequential(nn.LayerNorm(model_nembd),
                                        nn.Linear(model_nembd, mlp_dims))

        self.denseoutblocks = nn.ModuleList([DenseEncoderOut(n_embd=mlp_dims, mlp_dims=mlp_dims, sequence=True)
                                            for _ in range(num_mlp_layers)])

        self.lnout = nn.LayerNorm(mlp_dims)
        self.denseout = nn.Linear(mlp_dims, data_channels)

    def forward(self, x, t):
        """
        x.shape (B, L, C)
        t.shape (B, 1, 1)
        """

        x = self.BB_embd(x)
        batch_size, seq_len, data_channels = x.shape
        pemb = self.PositionEncoding(torch.arange(seq_len))
        pemb = pemb[None, :, :]
        x = x + pemb
        x = self.Transformer(x)
        x = self.interlayer(x)
        for layer in self.denseoutblocks:
            x = layer(x, t)
        x = self.lnout(x)
        x = self.denseout(x)

        return x

if __name__ == "__main__":
    cns = ContinousNoiseSchedual(4)
    t = torch.randn(1).repeat(10,1)
    print(cns(t).shape)
