import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td

from .savi import build_savi

# Replace to our slot-encoder
# TODO: dataloader convert to (c, 64, 64), NN interpolate
# input (T, bs, c, 10, 10)
# output (T, bs, embedding_size)
# embedding_size = N x D


class SlotObsEncoder(nn.Module):

    def __init__(self, params):
        super(SlotObsEncoder, self).__init__()

        savi = build_savi(params).eval().cuda()
        for p in savi.parameters():
            p.requires_grad = False

        cur_dir = os.path.dirname(os.path.realpath(__file__))
        w = os.path.join(cur_dir, params.savi_path)  # './savi/weights/minatar.pth'
        savi.load_weight(w)
        self.savi = savi

    def forward(self, obs):
        """obs: [T, B, C, H, W]"""
        # to [B, T, C, H, W]
        T, B, _, _, _ = obs.shape
        obs = 2 * obs - 1
        obs = torch.clamp(obs, min=-1, max=1)
        obs = obs.flatten(0,1)
        obs = F.interpolate(obs, (64, 64), mode='nearest')
        obs = obs.unflatten(0, (T, B))

        obs = obs.transpose(0, 1).contiguous()
        embed = self.savi({'img': obs})['slots']  # [B, T, N, D]
        # back to [T, B, N, D]
        embed = embed.transpose(0, 1).contiguous()
        embed = embed.flatten(2, 3)  # [T, B, (N*D)]
        return embed

    @property
    def embed_size(self):
        return self.savi.slot_size * self.savi.num_slots

    def train(self, mode=True):
        super().train(mode)
        # fix SAVi
        self.savi.eval()
        return self


class SlotObsDecoder(nn.Module):

    def __init__(self, params):
        super(SlotObsDecoder, self).__init__()
        emb_dim = params.slot_size * params.num_slots
        self.fc = nn.Linear(2 * emb_dim, emb_dim)

    def forward(self, x):
        mean = self.fc(x)
        obs_dist = td.Independent(td.Normal(mean, 1), 1)
        return obs_dist
