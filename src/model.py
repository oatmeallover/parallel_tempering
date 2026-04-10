import torch
import torch.nn as nn
import torch.nn.functional as F
import math

torch.manual_seed(42)

class SinusoidalTimeEmbedding(nn.Module):
	def __init__(self, dim):
		super().__init__()
		self.dim = dim

	def forward(self, t):
		if t.dim() == 2:
			t = t.squeeze(-1)

		half = self.dim // 2
		freqs = torch.exp(
			-math.log(10000) * torch.arange(half, device=t.device) / half
		)
		args = t[:, None] * freqs[None, :]
		emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

		if self.dim % 2 == 1:
			emb = F.pad(emb, (0, 1))

		return emb


class MLP(nn.Module):
    def __init__(
        self,
        x_dim=1,
        hidden_dim=512,   # 128 -> 512
        time_dim=64,      # 32 -> 64
        n_layers=8,       # 4 -> 8
    ):
        super().__init__()

        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        self.input = nn.Linear(x_dim + time_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers)]
        )
        self.output = nn.Linear(hidden_dim, x_dim)

    def forward(self, x, t):
        t_emb = self.time_embed(t)
        h = torch.cat([x, t_emb], dim=-1)
        h = F.silu(self.input(h))
        for layer in self.layers:
            h = h + F.silu(layer(h))
        return self.output(h)
