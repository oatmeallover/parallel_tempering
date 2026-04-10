import torch
import matplotlib.pyplot as plt
import numpy as np
import os

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


def plot_samples_grid(dataset_name, method, k, sigma, step_scale, n_langevin_steps, x_limit=6, save_dir="figures", figsize_per_panel=(5,4), filename=None):
	"""Generate samples and plot against true distribution. Two parameters must be lists, if you want a single plot, need list with one item"""
	os.makedirs(save_dir, exist_ok=True)

	params = {
		"dataset_name": dataset_name,
		"method": method,
		"k": k,
		"sigma": sigma,
		"step_scale": step_scale,
		"n_langevin_steps": n_langevin_steps,
	}

	list_params = [name for name, val in params.items() if isinstance(val, list)]
	if len(list_params) != 2:
		raise ValueError(f"Expected exactly 2 list parameters, got {len(list_params)}: {list_params}")
	
	row_name, col_name = list_params
	print(f"Device:                  {device}")
	print(f"Figure rows will be: 	 {row_name}")
	print(f"Figure columns will be:  {col_name}")

	row_vals = params[row_name]
	col_vals = params[col_name]

	n_rows = len(row_vals)
	n_cols = len(col_vals)

	fig_width = figsize_per_panel[0] * n_cols
	fig_height = figsize_per_panel[0] * n_rows
	fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), squeeze=False)

	overall_title = f"Mixture of Gaussians Score Rescaling"
	fig.suptitle(overall_title, fontsize=14, fontweight='bold')

	x_axis = np.linspace(-x_limit, x_limit, 500)
	bins = np.linspace(-x_limit, x_limit, 200)

	samples = {}
	y_max = 0.0

	if "dataset_name" != row_name:
		model = load_model(f"{ckpt_dir}/{dataset_name}_1.0.pt")
		dataset_config = DATASETS[dataset_name]

	for i, row_val in enumerate(row_vals):

		if "dataset_name" == row_name:
			name = row_val
			model = load_model(f"{ckpt_dir}/{name}_1.0.pt") # dataset name would always be first if it is a param
			dataset_config = DATASETS[name]

		for j, col_val in enumerate(col_vals):

			ax = axes[i, j]

			working_params = params.copy()
			working_params[row_name] = row_val
			working_params[col_name] = col_val

			k_pdf = working_params["k"]

			pdf = compute_mixture_pdf(dataset_config, x_axis, k_pdf)

			print(f"\nRunning sampling with {row_name} = {row_val} and {col_name} = {col_val}")

			x_sampled = sampling(
				model=model,
				dataset_config=dataset_config,
				method=working_params["method"],
				k=working_params["k"],
				sigma=working_params["sigma"],
				step_scale=working_params["step_scale"],
				n_langevin_steps=working_params["n_langevin_steps"]
			)

			samples[(i, j)] = {"params": params, "samples": x_sampled}

			ax.hist(
				x_sampled.cpu().numpy(),
				bins=bins,
				density=True,
				alpha=0.5,
				label=f"Samples ({row_name}={row_val}, {col_name}={col_val})"
			)

			ax.plot(x_axis, pdf, label=f"True k={k_pdf} PDF")
			ax.set_xlim(-x_limit, x_limit)

			if i == 0 and len(col_vals) > 1:
				ax.set_title(f"{col_name} = {col_val}", fontsize=11, fontweight="bold")
			if j == 0 and len(row_vals) > 1:
				ax.set_ylabel(f"{row_name} = {row_val}", fontsize=11, fontweight="bold")

			ax.legend(fontsize=8)
			y_max = max(y_max, ax.get_ylim()[1])

	for row in axes:
		for ax in row:
			ax.set_ylim(0,y_max)

	if filename is None:
		filename = f"{row_name}_{col_name}.png"

	save_path = os.path.join(save_dir, filename)
	
	plt.tight_layout(rect=[0, 0.03, 1, 0.95])
	plt.savefig(save_path, dpi=200)
	plt.show()

	return samples

if __name__ == "__main__":
	dataset_name = "composed"
	methods = ["DDPM", "ULA"]
	k = [0.25, 0.5, 1.0, 2.0, 4.0]
	sigma = 0.5
	step_scale = 2
	n_langevin_steps = 10
	filename = None

	_ = plot_samples_grid(
		dataset_name=dataset_name, 
		method=methods, 
		k=k, 
		sigma=sigma, 
		step_scale=step_scale, 
		n_langevin_steps=n_langevin_steps, 
		x_limit = 10, save_dir="figures", filename=filename
	)