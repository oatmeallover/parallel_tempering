import torch
import matplotlib.pyplot as plt
import numpy as np

from .model import load_model     
from .dataset import compute_mixture_pdf  
from .config import DEVICE, DATASETS, DATASETS_IMG, CKPT_DIR, N_DIFFUSION_STEPS
from .sample import ddpm_tsr_swapped

torch.manual_seed(42)
np.random.seed(42)

device = DEVICE
ckpt_dir = CKPT_DIR

def plot_temperature_triptych(
	dataset_name="composed",
	k=2.0,
	n_samples=4,
	n_replicas=3,
	replica_swaps=False,
	swap_algorithm= {
		"analytical" : False,
		"swap_towards_k" : False,
		"even": {90, 70, 50, 30},
		"odd": {80, 60, 40, 20},
		"debug" : False
	},
	x_limit=8,
	n_bins=220,
):
	"""Create side-by-side visuals for original, flattened, and sharpened sampling."""
	model = load_model(f"{ckpt_dir}/{dataset_name}_1.0.pt", dataset_name)
	n_rows = 1

	if dataset_name in DATASETS:
		dataset_shape = DATASETS[dataset_name]["dataset_shape"]
		x_axis = np.linspace(-x_limit, x_limit, n_bins)
		bins = np.linspace(-x_limit, x_limit, n_bins)
	elif dataset_name in DATASETS_IMG:
		sample_shape = DATASETS_IMG[dataset_name]["sample_shape"]  # (1, 28, 28)
		dataset_shape = (n_samples, *sample_shape)                 # (n_samples, 1, 28, 28)
		n_rows = n_samples

	k_ladder = np.linspace(1.0, k, n_replicas)
	titles = [f"k={k:.2f}" for k in k_ladder]

	samples_ladder = ddpm_tsr_swapped(model, dataset_shape, k, k_ladder, replica_swaps=replica_swaps, swap_algorithm=swap_algorithm)

	fig, axes = plt.subplots(n_rows, len(k_ladder), figsize=(len(k_ladder)*5, 4*n_rows), sharey=True)
	# Normalize axes to always be 2D: (n_rows, 3)
	if n_rows == 1:
		axes = axes[np.newaxis, :]  # (1, 3)
		
	for col, (k, title) in enumerate(zip(k_ladder, titles)):

		samples = samples_ladder[k]

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

	parts = []

	if replica_swaps: 

		if swap_algorithm['p_ratio'] == 's':
			parts.append(f"Swap with source ratio")
		elif swap_algorithm['p_ratio'] == 't':
			parts.append(f"Swap with target ratio")
		elif swap_algorithm['p_ratio'] == 'p':
			parts.append(f"Parallel Swap")

		if swap_algorithm['analytical']:
			parts.append("Analytical")
	
	else:
		parts.append("DDPM TSR")

	fig.suptitle(" | ".join(parts), y=1.03)
	plt.tight_layout()

	return fig, axes