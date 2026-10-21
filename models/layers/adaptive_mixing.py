import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from torch.utils.checkpoint import checkpoint as cp


class AdaptiveMixing(BaseModule):
    """AdaMixer-style adaptive channel and point mixing.
    """

    def __init__(
            self,
            in_dim,
            in_points,
            n_groups=1,
            query_dim=None,
            out_dim=None,
            out_points=None,
            init_cfg=None):
        super().__init__(init_cfg)

        out_dim = out_dim if out_dim is not None else in_dim
        out_points = out_points if out_points is not None else in_points
        query_dim = query_dim if query_dim is not None else in_dim

        self.query_dim = query_dim
        self.in_dim = in_dim
        self.in_points = in_points
        self.n_groups = n_groups
        self.out_dim = out_dim
        self.out_points = out_points

        self.eff_in_dim = in_dim // n_groups
        self.eff_out_dim = out_dim // n_groups

        self.m_parameters = self.eff_in_dim * self.eff_out_dim
        self.s_parameters = self.in_points * self.out_points
        self.total_parameters = self.m_parameters + self.s_parameters

        self.parameter_generator = nn.Linear(
            self.query_dim, self.n_groups * self.total_parameters)
        self.out_proj = nn.Linear(
            self.eff_out_dim * self.out_points * self.n_groups,
            self.query_dim,
        )
        self.act = nn.ReLU(inplace=True)

    @torch.no_grad()
    def init_weights(self):
        nn.init.zeros_(self.parameter_generator.weight)

    def inner_forward(self, x, query):
        B, Q, G, P, C = x.shape
        assert G == self.n_groups
        assert P == self.in_points
        assert C == self.eff_in_dim

        '''generate mixing parameters'''
        params = self.parameter_generator(query)
        params = params.reshape(B * Q, G, -1)
        out = x.reshape(B * Q, G, P, C)

        M, S = params.split([self.m_parameters, self.s_parameters], 2)
        M = M.reshape(B * Q, G, self.eff_in_dim, self.eff_out_dim)
        S = S.reshape(B * Q, G, self.out_points, self.in_points)

        '''adaptive channel mixing'''
        out = torch.matmul(out, M)
        out = F.layer_norm(out, [out.size(-2), out.size(-1)])
        out = self.act(out)

        '''adaptive point mixing'''
        out = torch.matmul(S, out)  # implicitly transpose and matmul
        out = F.layer_norm(out, [out.size(-2), out.size(-1)])
        out = self.act(out)

        '''linear transfomation to query dim'''
        out = out.reshape(B, Q, -1)
        out = self.out_proj(out)
        out = query + out

        return out

    def forward(self, x, query):
        if self.training and x.requires_grad:
            return cp(self.inner_forward, x, query, use_reentrant=False)
        return self.inner_forward(x, query)
