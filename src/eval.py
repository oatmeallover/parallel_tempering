import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import defaultdict

from .model import MLP       
from .dataset import compute_mixture_pdf  
from .config import DEVICE, DATASETS, CKPT_DIR, N_DIFFUSION_STEPS
from .sample import sampling

torch.manual_seed(42)
np.random.seed(42)

device = DEVICE
ckpt_dir = CKPT_DIR

def load_model(path):
	"""Load trained model from checkpoint"""
	model = MLP().to(device)
	model.load_state_dict(torch.load(path, map_location=device))
	model.eval()
	return model


def ladder_ddpm(dataset_name, k, sigma, step_scale, n_langevin_steps, n_replicas, x_limit=6, save_dir="figures", figsize_per_panel=(5,4), filename=None):
	os.makedirs(save_dir, exist_ok=True)
	n_rows = n_replicas
	n_cols = 2

	y_max = 0.0

	fig_width = figsize_per_panel[0] * n_cols
	fig_height = figsize_per_panel[0] * n_rows

	fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), squeeze=False)

	axes[0, 0].set_title("With Replica Swaps - ULA ", fontsize=11, fontweight="bold")
	axes[0, 1].set_title("No Replica Swaps", fontsize=11, fontweight="bold")

	overall_title = f"Comparison of Exchanged Chains and DDPM"
	fig.suptitle(overall_title, fontsize=14, fontweight='bold')
	
	x_axis = np.linspace(-x_limit, x_limit, 500)
	bins = np.linspace(-x_limit, x_limit, 200)

	y_max = 0.0

	model = load_model(f"{ckpt_dir}/{dataset_name}_1.0.pt") # dataset name would always be first if it is a param
	dataset_config = DATASETS[dataset_name]
	dataset_shape = dataset_config["dataset_shape"]

	x_ladder, a_ladder = sampling(
		model=model,
		dataset_shape=dataset_shape,
		k=k,
		sigma=sigma,
		step_scale=step_scale,
		n_langevin_steps=n_langevin_steps,
		n_replicas=n_replicas
	)

	k_ladder = list(x_ladder.keys())

	for i in range(n_replicas):
		ax = axes[i, 0]
		k_val = k_ladder[i]
		pdf = compute_mixture_pdf(dataset_config, x_axis, k_val)

		ax.hist(x_ladder[k_val].cpu().numpy(),
		  bins = bins,
		  density=True,
		  alpha=0.5, 
		  label=f"k = {k_val}")
	
		ax.plot(x_axis, pdf, label=f"True k={k_val} PDF")
		ax.set_xlim(-x_limit, x_limit)

		ax.legend(fontsize=8)
		y_max = max(y_max, ax.get_ylim()[1])
		axes[i, 0].set_ylabel(f"k = {k_val}", fontsize=11)

	for i, k_val in enumerate(k_ladder):

		x_ladder_tsr, _ = sampling(
			model=model,
			dataset_shape=dataset_shape,
			k=k_val,
			sigma=sigma,
			step_scale=step_scale,
			n_langevin_steps=0,
			n_replicas=1
		)

		ax = axes[i, 1]
		pdf = compute_mixture_pdf(dataset_config, x_axis, k_val)

		ax.hist(x_ladder_tsr[k_val].cpu().numpy(),
		  bins = bins,
		  density=True,
		  alpha=0.5, 
		  label=f"k = {k_val}")
	
		ax.plot(x_axis, pdf, label=f"True k={k_val} PDF")
		ax.set_xlim(-x_limit, x_limit)

		ax.legend(fontsize=8)
		y_max = max(y_max, ax.get_ylim()[1])


	for row in axes:
		for ax in row:
			ax.set_ylim(0,y_max)

	if filename is None:
		filename = f"swaps_{n_replicas}_{k}.png"

	save_path = os.path.join(save_dir, filename)
	
	plt.tight_layout(rect=[0, 0.03, 1, 0.95])
	plt.savefig(save_path, dpi=200)
	plt.show()

	return x_ladder, a_ladder

if __name__ == "__main__":
	pass