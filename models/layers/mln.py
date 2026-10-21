import torch.nn as nn


class MLN(nn.Module):
    """Modulated layer normalization conditioned on a latent code.

    The code is mapped to a per-channel scale/shift applied on top of a
    non-affine LayerNorm. ``gamma``/``beta`` are initialised to the identity so
    the module is a no-op at the start of training.

    Args:
        c_dim: Dimension of the conditioning code.
        f_dim: Feature dimension being modulated.
    """

    def __init__(self, c_dim, f_dim=256, use_ln=True):
        super().__init__()
        self.c_dim = c_dim
        self.f_dim = f_dim
        self.use_ln = use_ln

        self.reduce = nn.Sequential(
            nn.Linear(c_dim, f_dim),
            nn.ReLU(),
        )
        self.gamma = nn.Linear(f_dim, f_dim)
        self.beta = nn.Linear(f_dim, f_dim)
        if self.use_ln:
            self.ln = nn.LayerNorm(f_dim, elementwise_affine=False)
        self.init_weights()

    def init_weights(self):
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)
        nn.init.ones_(self.gamma.bias)
        nn.init.zeros_(self.beta.bias)

    def forward(self, x, c):
        if self.use_ln:
            x = self.ln(x)
        c = self.reduce(c)
        return self.gamma(c) * x + self.beta(c)
