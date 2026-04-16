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
	n_cols = 10

	y_max = 0.0

	fig_width = figsize_per_panel[0] * n_cols
	fig_height = figsize_per_panel[0] * n_rows

	fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), squeeze=False)

	# axes[0, 0].set_title("Replica Swaps", fontsize=11, fontweight="bold")
	# axes[0, 1].set_title("No Replica Swaps", fontsize=11, fontweight="bold")

	overall_title = f"Comparison of Exchanged Chains and DDPM"
	fig.suptitle(overall_title, fontsize=14, fontweight='bold')
	
	x_axis = np.linspace(-x_limit, x_limit, 500)
	bins = np.linspace(-x_limit, x_limit, 200)

	y_max = 0.0

	model = load_model(f"{ckpt_dir}/{dataset_name}_1.0.pt") # dataset name would always be first if it is a param
	dataset_config = DATASETS[dataset_name]
	dataset_shape = dataset_config["dataset_shape"]

	#k_ladder = np.linspace(k, 1.0/k, n_replicas)
	k_ladder = np.array([4.0, 2.0, 1.0, 0.5, 0.25])

	# for our swaps

	x_ladder, a_ladder = sampling(
		model=model,
		dataset_shape=dataset_shape,
		k=k,
		sigma=sigma,
		step_scale=step_scale,
		n_langevin_steps=n_langevin_steps,
		n_replicas=n_replicas,
		k_ladder=k_ladder
	)

	for cols in range(n_cols):

		j = int(N_DIFFUSION_STEPS * cols / (n_cols-1))
		axes[0, cols].set_title(f"{j}", fontsize=11, fontweight="bold")

		for i in range(n_replicas):
			ax = axes[i, cols]
			k_val = k_ladder[i]
			pdf = compute_mixture_pdf(dataset_config, x_axis, k_val)

			ax.hist(x_ladder[j][k_val].cpu().numpy(),
			bins = bins,
			density=True,
			alpha=0.5, 
			label=f"k = {k_val} t = {j}")
		
			ax.plot(x_axis, pdf, label=f"True k={k_val} PDF")
			ax.set_xlim(-x_limit, x_limit)

			ax.legend(fontsize=8)
			y_max = max(y_max, ax.get_ylim()[1])
			axes[i, 0].set_ylabel(f"k = {k_val}", fontsize=11)

	# for i, k_val in enumerate(k_ladder):

	# 	x_ladder_tsr, a_ladder_tsr = sampling(
	# 		model=model,
	# 		dataset_shape=dataset_shape,
	# 		k=k_val,
	# 		sigma=sigma,
	# 		step_scale=step_scale,
	# 		n_langevin_steps=n_langevin_steps,
	# 		n_replicas=1,
	# 		k_ladder=None
	# 	)

	# 	ax = axes[i, 1]
	# 	k_val = k_ladder[i]
	# 	pdf = compute_mixture_pdf(dataset_config, x_axis, k_val)

	# 	ax.hist(x_ladder_tsr[0][k_val].cpu().numpy(),
	# 	  bins = bins,
	# 	  density=True,
	# 	  alpha=0.5, 
	# 	  label=f"k = {k_val}")
	
	# 	ax.plot(x_axis, pdf, label=f"True k={k_val} PDF")
	# 	ax.set_xlim(-x_limit, x_limit)

	# 	ax.legend(fontsize=8)
	# 	y_max = max(y_max, ax.get_ylim()[1])


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


def plot_acceptance_over_time(acceptance_ladder, save_dir="figures", filename="swaps_acceptance_time.png"):

	fig, ax = plt.subplots(figsize=(10, 5))

	for (k_less, k_more), records in acceptance_ladder.items():
		# Group acceptance rates by timestep t
		by_time = defaultdict(list)
		for r in records:
			by_time[r["t"]].append(r["acceptance"].cpu().mean().item())

		# Average over langevin steps at each t
		ts = sorted(by_time.keys())
		ts_cpu = [item.cpu() for item in ts]

		avg_acceptance = [sum(by_time[t]) / len(by_time[t]) for t in ts]

		ax.plot(ts_cpu, avg_acceptance, label=f"k={k_less:.2f} ↔ k={k_more:.2f}", marker='o', markersize=2)

	ax.set_xlabel("Timestep t")
	ax.set_ylabel("Average Acceptance Rate")
	ax.set_title("Parallel Tempering Acceptance Rate Over Time")
	ax.legend()
	ax.grid(True, alpha=0.3)
	plt.tight_layout()

	save_path = os.path.join(save_dir, filename)
	plt.savefig(save_path, dpi=200)

	plt.show()

	return fig

def plot_acceptance_over_position(acceptance_ladder, x_final, timesteps_to_show=[10, 50, 90], save_dir="figures", filename="swaps_acceptance_position.png"):
	x_np = x_final.detach().cpu().numpy().flatten()
	
	bins = np.linspace(x_np.min(), x_np.max(), 40)
	bin_centers = 0.5 * (bins[:-1] + bins[1:])
	bin_width = bins[1] - bins[0]

	y_min = 1.0

	n_pairs = len(acceptance_ladder)
	n_times = len(timesteps_to_show)
	
	fig, axes = plt.subplots(
		n_pairs, n_times,
		figsize=(5 * n_times, 3 * n_pairs),
		sharex=True
	)
	if n_pairs == 1:
		axes = axes[np.newaxis, :]
	if n_times == 1:
		axes = axes[:, np.newaxis]

	for row, ((k_less, k_more), records) in enumerate(acceptance_ladder.items()):
		by_time = defaultdict(list)
		for r in records:
			by_time[r["t"]].append(r["acceptance"].cpu())

		for col, t in enumerate(timesteps_to_show):
			ax = axes[row, col]

			closest_t = min(by_time.keys(), key=lambda t_: abs(t_ - t))
			stacked = torch.stack(by_time[closest_t], dim=0).float()  # (n_langevin, n_particles)
			per_particle_acceptance = stacked.mean(dim=0).numpy().flatten()

			bin_indices = np.digitize(x_np, bins) - 1
			bin_indices = np.clip(bin_indices, 0, len(bin_centers) - 1)

			bin_acceptance = np.full(len(bin_centers), np.nan)
			
			for b in range(len(bin_centers)):
				mask = bin_indices == b
				if mask.sum() > 0:
					bin_acceptance[b] = per_particle_acceptance[mask].mean()
					y_min = min(y_min, per_particle_acceptance[mask].mean())

			valid = ~np.isnan(bin_acceptance)
			ax.bar(bin_centers[valid], bin_acceptance[valid],
				   width=bin_width * 0.9, color="crimson", alpha=0.8)
			
			mean_acc = np.nanmean(bin_acceptance)
			ax.axhline(mean_acc, color='black', linestyle='--', linewidth=1,
					   label=f"mean={mean_acc:.3f}")
			
			ax.set_ylabel("Acceptance rate")
			ax.set_xlabel("x")
			ax.set_title(f"k={k_less:.2f}↔{k_more:.2f},  t≈{closest_t}")
			ax.legend(fontsize=8)

	for row in range(n_pairs):
		for col in range(n_times):
			axes[row, col].set_ylim(y_min * 0.98, 1)

	fig.suptitle("Acceptance Rate by Position", fontsize=14, y=1.01)
	plt.tight_layout()

	save_path = os.path.join(save_dir, filename)
	plt.savefig(save_path, dpi=200)

	plt.show()
	return fig

if __name__ == "__main__":

	dataset_name = "composed"
	k = 7.0
	sigma = 0.5
	step_scale = 1
	n_langevin_steps = 3
	n_replicas = 7

	x_ladder, a_ladder = ladder_ddpm(dataset_name, k, sigma, step_scale, n_langevin_steps, n_replicas, x_limit=6, save_dir="figures", figsize_per_panel=(5,4), filename=None)
	plot_acceptance_over_time(a_ladder)
	plot_acceptance_over_position(a_ladder, x_ladder[0][k])