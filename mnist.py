import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm

import sys

from torchvision import datasets as tv_datasets #mnist support
from torchvision import transforms #mnist support

from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_diffusion_steps = 100

TRAINING = {
	"n_steps": 500_000,
	"batch_size": 512,
	"lr": 2e-4
}

TRAINING_IMG = {
	"n_steps": 1_000_000,
	"batch_size": 64,
	"lr": 2e-4
}

datasets = {
	"single": {"dataset_shape": (50_000, 1), "means": [0.0], "stds": 1.0},
	"barrier": {"dataset_shape": (50_000, 1), "means": [-3.0, 3.0], "stds": 0.5},
	"composed": {
		"dataset_shape": (100_000, 1),
		"means": [-3.0, 0.0, 3.0],
		"stds": [0.5, 1.0, 0.5],
	},
}

img_datasets = {
	"mnist": {"sample_shape": (1, 28, 28)}
}

ckpt_dir = "model_checkpoints"


@torch.no_grad()
def generate_gaussian_mixture(dataset_name, device='cpu'):
	"""Generates mixture of gaussians according to inputted means and standard deviations"""

	dataset_config = img_datasets[dataset_name]
	dataset_shape = dataset_config["dataset_shape"]
	n_samples = dataset_shape[0]

	means = torch.as_tensor(dataset_config['means'], dtype=torch.float32)
	stds = dataset_config['stds']
	
	n_gaussians = len(means)
	
	if isinstance(stds, (int, float)):
		stds = torch.full((n_gaussians,), float(stds))
	else:
		stds = torch.as_tensor(stds, dtype=torch.float32)
		assert len(stds) == n_gaussians, f"stds length {len(stds)} != n_gaussians {n_gaussians}"
		
	component_ids = np.random.choice(n_gaussians, size=n_samples)
	samples = torch.zeros(n_samples, 1, device=device)
	
	for i in range(n_gaussians):
		mask = component_ids == i
		samples[mask] = torch.normal(
			mean=float(means[i]),
			std=float(stds[i]),
			size=(mask.sum(), 1)
		).to(device)
	
	return samples


@torch.no_grad()
def load_mnist_tensor(train=True, normalize_to_minus1_1=True):
	tfms = [transforms.ToTensor()]
	if normalize_to_minus1_1:
		tfms.append(transforms.Lambda(lambda x: (x - 0.1307) / 0.3081))

	ds = tv_datasets.MNIST(
		root="./data",
		train=train,
		download=True,
		transform=transforms.Compose(tfms),
	)
	x = torch.stack([img for img, _ in ds], dim=0)
	return x


@torch.no_grad()
def build_training_tensor(dataset_name, n_samples=None, train=True):
	if dataset_name in datasets:
		return generate_gaussian_mixture(dataset_name, n_samples=n_samples, device='cpu')
	elif dataset_name in img_datasets:
		x = load_mnist_tensor(train=train)
		if n_samples is not None:
			x = x[:n_samples]
		return x
	raise ValueError(f"Unsupported dataset name")


@torch.no_grad()
def compute_mixture_pdf(dataset_name, x_axis, k=1.0):
	"""Computes analytical pdf of training dataset from dataset config file, used for plotting"""

	dataset_config = datasets[dataset_name]
	means = np.array(dataset_config['means'])
	stds = dataset_config['stds']
		
	if isinstance(stds, (int, float)):
		stds = np.full(len(means), stds)
	else:
		stds = np.array(stds)
		
	stds = stds / np.sqrt(k)
	
	n_gaussians = len(means)
	pdf = np.zeros_like(x_axis)
	
	for mu, sigma in zip(means, stds):
		pdf += norm.pdf(x_axis, loc=mu, scale=sigma)
	
	pdf /= n_gaussians  
	
	return pdf 

@torch.no_grad()
def cosine_beta_schedule(timesteps, s=0.008):
	"""Cosine noise schedule, taken from reduce reuse recycle code"""
	steps = timesteps + 1
	t = torch.linspace(0, timesteps, steps, dtype=torch.float32) / timesteps
	alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
	alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
	betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
	return torch.clip(betas, 0, 0.999)

betas = cosine_beta_schedule(n_diffusion_steps).to(device)
alphas = 1.0 - betas
alpha_bars = torch.cumprod(alphas, dim=0)  # alphā_t

ts_desc = torch.arange(n_diffusion_steps - 1, -1, -1, device=device)


@torch.no_grad()
def compute_tsr_schedule(k, sigma, t):
	"""Computes temporal score rescaling coefficient. Outpit will be shape (N_DIFFUSION_STEPS, n_langevin_steps)"""

	a_bar = alpha_bars[t]
	sigma_t = torch.sqrt(1.0 - a_bar)
	alpha_t = torch.sqrt(a_bar)

	eta_t = (alpha_t**2) / (sigma_t**2)
	num = eta_t * (sigma ** 2) + 1
	den = (eta_t * (sigma ** 2)) / k + 1
	tsr = num / den

	return tsr

@torch.no_grad()
def compute_score(model, x, t, k, sigma):
	"""Computes score = - epsilon * temp / √(1 - α_bar)"""
	
	x_shape = x.shape
	ones = torch.ones((x_shape[0], 1), device=device)
	eps_hat = model(x, t * ones)   
	a_bar = alpha_bars[t]
	temp_t = compute_tsr_schedule(k, sigma, t)
	score_hat= - eps_hat * temp_t / torch.sqrt(1.0 - a_bar)
	return score_hat


@torch.no_grad()
def ddpm_tsr(model, dataset_shape, k=1.0, sigma=1.0):
	"""Sampling algorithm for DDPM, ULA, and MALA"""

	x = torch.randn(dataset_shape, device=device)
		
	for t in ts_desc: 

		alpha_t = alphas[t]
		beta_t = betas[t]
		sqrt_alpha_t = torch.sqrt(alpha_t)
		sqrt_beta_t = torch.sqrt(beta_t)
		noise = torch.randn(dataset_shape, device=device)

		score_hat = compute_score(model, x, t, k, sigma)
		x = (x + beta_t * score_hat) / sqrt_alpha_t + sqrt_beta_t * noise

	return x


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


class ResBlock(nn.Module):
    def __init__(self, channels, time_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, channels)

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


class UNet(nn.Module):
    def __init__(self, in_channels=1, base_channels=64, time_dim=128):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4), nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim)
        )
        C = base_channels
        # Encoder
        self.conv_in  = nn.Conv2d(in_channels, C, 3, padding=1)
        self.enc1     = ResBlock(C, time_dim)
        self.down1    = nn.Conv2d(C, C*2, 4, stride=2, padding=1)   # 28->14
        self.enc2     = ResBlock(C*2, time_dim)
        self.down2    = nn.Conv2d(C*2, C*4, 4, stride=2, padding=1) # 14->7
        # Bottleneck
        self.mid1     = ResBlock(C*4, time_dim)
        self.mid2     = ResBlock(C*4, time_dim)
        # Decoder
        self.up1      = nn.ConvTranspose2d(C*4, C*2, 4, stride=2, padding=1) # 7->14
        self.dec1     = ResBlock(C*4, time_dim)  # C*4 due to skip
        self.up2      = nn.ConvTranspose2d(C*4, C, 4, stride=2, padding=1)   # 14->28
        self.dec2     = ResBlock(C*2, time_dim)  # C*2 due to skip
        self.norm_out = nn.GroupNorm(8, C*2)
        self.conv_out = nn.Conv2d(C*2, in_channels, 3, padding=1)

    def forward(self, x, t):
        t_emb = self.time_mlp(self.time_embed(t))
        # Encoder
        h0 = F.silu(self.conv_in(x))
        h1 = self.enc1(h0, t_emb)          # (B, C, 28, 28)
        h2 = self.enc2(self.down1(h1), t_emb)  # (B, 2C, 14, 14)
        h  = self.mid2(self.mid1(self.down2(h2), t_emb), t_emb)  # (B, 4C, 7, 7)
        # Decoder with skip connections
        h  = self.dec1(torch.cat([self.up1(h), h2], dim=1), t_emb)
        h  = self.dec2(torch.cat([self.up2(h), h1], dim=1), t_emb)
        return self.conv_out(F.silu(self.norm_out(h)))
	
def load_model(path, dataset_name):
	"""Load trained model from checkpoint for a specific dataset."""
	model = build_model_for_dataset(dataset_name)
	state = torch.load(path, map_location=device)
	if dataset_name in datasets:
		model.load_state_dict(state, strict=True)
	else:
		model.load_state_dict(state, strict=False)

	model.eval()
	return model

def build_model_for_dataset(dataset_name):
	if dataset_name in datasets:
		return MLP().to(device)
	elif dataset_name in img_datasets:
		cfg = img_datasets[dataset_name]
		return UNet(in_channels=cfg["sample_shape"][0]).to(device)


def train_model(dataset_name, existing_checkpoint=None, save_name=None, k=1.0, log_every=1000):
	"""Trains model according to a dataset defined by dataset_config"""
		
	x0_all = build_training_tensor(dataset_name)

	if dataset_name in datasets:
		training_setup = TRAINING
		
	elif dataset_name in img_datasets:
		training_setup = TRAINING_IMG

	n_steps = training_setup["n_steps"]
	batch_size = training_setup["batch_size"]
	lr = training_setup["lr"]

	loader = DataLoader(
		TensorDataset(x0_all),
		batch_size=batch_size,
		shuffle=True,
		drop_last=True,
		num_workers=0,      # keep simple; set >0 if you want
		pin_memory=True,    # helps H2D transfer
	)

	model = build_model_for_dataset(dataset_name).to(device)

	if existing_checkpoint is not None:
		print(f"Loading {existing_checkpoint}")
		checkpoint = torch.load(existing_checkpoint, map_location=device)
		model.load_state_dict(checkpoint)

	opt = optim.Adam(model.parameters(), lr=lr)
	model.train()

	data_iter = iter(loader)

	for step in range(1, n_steps + 1):
		try:
			(x0,) = next(data_iter)
		except StopIteration:
			data_iter = iter(loader)  # reshuffles because shuffle=True
			(x0,) = next(data_iter)

		x0 = x0.to(device, non_blocking=True)

		t = torch.randint(0, n_diffusion_steps, (batch_size, 1), device=device)
		a_bar = alpha_bars[t]

		if dataset_name in img_datasets:
			a_bar = a_bar.view(-1, 1, 1, 1)

		noise = torch.randn_like(x0)
		xt = torch.sqrt(a_bar) * x0 + torch.sqrt(1.0 - a_bar) * noise

		xt = xt.detach().requires_grad_(True)  # we need grads wrt xt
		eps_hat = model(xt, t)   # energy values from EBM: shape [B, 1] or [B]

		loss = ((noise - eps_hat) ** 2).mean()

		opt.zero_grad()
		loss.backward()
		opt.step()

		if step % log_every == 0:
			print(f"dataset = {dataset_name} temperature={k} step={step} loss={loss.item():.4f}")
			sys.stdout.flush()

	save_name = dataset_name if save_name is None else save_name
	save_path = f"{ckpt_dir}/{save_name}_{k:.1f}.pt"
	torch.save(model.state_dict(), save_path)
	print(f"Trained model saved to {save_path}")
	return model


def plot_temperature_triptych(
	dataset_name="composed",
	sigma=1.0,
	x_limit=8,
	n_bins=220,
	n_samples=4
):
	"""Create side-by-side visuals for original, flattened, and sharpened sampling."""
	model = load_model(f"{ckpt_dir}/{dataset_name}_1.0.pt", dataset_name)
	n_rows = 1

	if dataset_name in datasets:
		dataset_shape = datasets[dataset_name]["dataset_shape"]
		x_axis = np.linspace(-x_limit, x_limit, n_bins)
		bins = np.linspace(-x_limit, x_limit, n_bins)
	elif dataset_name in img_datasets:
		sample_shape = img_datasets[dataset_name]["sample_shape"]  # (1, 28, 28)
		dataset_shape = (n_samples, *sample_shape)                 # (n_samples, 1, 28, 28)
		n_rows = n_samples

	ks = [1.0, 0.5, 2.0]
	titles = ["Original (k = 1.0)", "Flattened (k = 0.5)", "Sharpened (k = 2.0)"]

	fig, axes = plt.subplots(n_rows, len(ks), figsize=(17, 4 * n_rows), sharey=True)

	# Normalize axes to always be 2D: (n_rows, 3)
	if n_rows == 1:
		axes = axes[np.newaxis, :]  # (1, 3)

	for col, (k, title) in enumerate(zip(ks, titles)):
		samples = ddpm_tsr(model, dataset_shape, k=k, sigma=sigma)

		if n_rows ==1:
			ax = axes[0, col]
			samples_np = samples.detach().cpu().numpy().reshape(-1)
			pdf = compute_mixture_pdf(dataset_name, x_axis, k=k)
			ax.hist(samples_np, bins=bins, density=True, alpha=0.45, label="Samples")
			ax.plot(x_axis, pdf, linewidth=2.0, label="Analytic target")
			ax.set_xlabel("x")
			ax.grid(alpha=0.2)
			ax.set_title(title)
		else:
			samples = samples * 0.3081 + 0.1307
			
			print(samples.mean().item())
			print("std",samples.std().item())
			samples = torch.clamp(samples, 0.0, 1.0)

			for row in range(n_samples):
				ax = axes[row, col]
				ax.imshow(samples[row, 0].detach().cpu().numpy(), cmap="gray", vmin=0.0, vmax=1.0)
				ax.axis("off")
				if row == 0:
					ax.set_title(title)

	if n_rows == 1:
		axes[0, 0].set_ylabel("density")
		axes[0, -1].legend(loc="upper right")

	fig.suptitle(f"TSR temperature comparison on '{dataset_name}' template", y=1.03)
	plt.tight_layout()

	return fig, axes

train_model(dataset_name="mnist")
