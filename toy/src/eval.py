import torch
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker

from .model import load_model     
from .dataset import compute_mixture_pdf  
from .config_toy import DEVICE, DATASETS, DATASETS_IMG, CKPT_DIR, N_DIFFUSION_STEPS
from .sample import ddpm_tsr_swapped
from .score_integral import _lam_ladder
from .best_of_n import best_of_n

torch.manual_seed(42)
np.random.seed(42)

device = DEVICE
ckpt_dir = CKPT_DIR

def plot_temperature_triptych(
	dataset_name="composed",
	lam=2.0,
	n_samples=4,
	n_replicas=3,
	replica_swaps=False,
	swap_algorithm= {
		"analytical" : False,
		"swap_towards_lam" : False,
		"even": {90, 70, 50, 30},
		"odd": {80, 60, 40, 20},
		"debug" : False
	},
	x_limit=8,
	n_bins=220,
	best_of_n_runs=1,  # set > 1 to enable best-of-N
):
	"""Create side-by-side visuals for original, flattened, and sharpened sampling."""
	model = load_model(f"{ckpt_dir}/{dataset_name}_1.0.pt", dataset_name)
	n_rows = 1

	if dataset_name in DATASETS:
		dataset_config = DATASETS[dataset_name]
		dataset_shape = dataset_config["dataset_shape"]
		x_axis = np.linspace(-x_limit, x_limit, n_bins)
		bins = np.linspace(-x_limit, x_limit, n_bins)
	elif dataset_name in DATASETS_IMG:
		sample_shape = DATASETS_IMG[dataset_name]["sample_shape"]
		dataset_shape = (n_samples, *sample_shape)
		dataset_config = {"dataset_shape": dataset_shape}
		n_rows = n_samples

	lam_ladder = _lam_ladder(lam, n_replicas)
	titles = [f"lam={lam:.2f}" for lam in lam_ladder]

	# ── Sample: best-of-N or single run ──────────────────────────────────────
	if best_of_n_runs > 1 and dataset_name in DATASETS:
		print(f"Running best-of-{best_of_n_runs}...")
		samples_ladder, best_score, all_scores = best_of_n(
			model=model,
			dataset_config=dataset_config,
			lam=lam,
			lam_ladder=lam_ladder,
			n_samples=best_of_n_runs,
			swap_algorithm=swap_algorithm,
			replica_swaps=replica_swaps,
		)
		print(f"Best score: {best_score:.4f} | All scores: {[f'{s:.4f}' for s in all_scores]}")
	else:
		samples_ladder = ddpm_tsr_swapped(
			model, dataset_shape, lam, lam_ladder,
			replica_swaps=replica_swaps,
			swap_algorithm=swap_algorithm,
		)
	fig, axes = plt.subplots(n_rows, len(lam_ladder), figsize=(len(lam_ladder)*5, 4*n_rows), sharey=True)
	# Normalize axes to always be 2D: (n_rows, 3)
	if n_rows == 1:
		axes = axes[np.newaxis, :]  # (1, 3)
		
	for col, (lam, title) in enumerate(zip(lam_ladder, titles)):

		samples = samples_ladder[lam]

		if n_rows ==1:
			ax = axes[0, col]
			samples_np = samples.detach().cpu().numpy().reshape(-1)
			pdf = compute_mixture_pdf(dataset_name, x_axis, lam=lam)
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

	else:
		parts.append("DDPM TSR")

	if best_of_n_runs > 1:
		parts.append(f"Best of {best_of_n_runs}")

	fig.suptitle(" | ".join(parts), y=1.03)
	plt.tight_layout()

	return fig, axes


device = DEVICE
ckpt_dir = CKPT_DIR
LAM_RANGE = [0.25, 0.5, 1.0, 2.0, 4.0]
DATASETS_PLOT = ["barrier", "composed"]
COLOR_SAMPLES = "#378ADD"
COLOR_PDF     = "#D85A30"

def plot_grid(
    n_replicas=3,
    replica_swaps=False,
    swap_algorithm={"analytical": False, "swap_towards_lam": False,
                    "even": {90,70,50,30}, "odd": {80,60,40,20}, "debug": False},
    x_limit=8,
    n_bins=200,
):
    x_axis = np.linspace(-x_limit, x_limit, 500)
    bins   = np.linspace(-x_limit, x_limit, n_bins)

    fig, axes = plt.subplots(
        2, len(LAM_RANGE),
        figsize=(2.8 * len(LAM_RANGE), 4.8),
        sharey=True,
    )

    for row, dataset_name in enumerate(DATASETS_PLOT):
        model          = load_model(f"{ckpt_dir}/{dataset_name}_1.0.pt", dataset_name)
        dataset_config = DATASETS[dataset_name]
        dataset_shape  = dataset_config["dataset_shape"]

        for col, lam in enumerate(LAM_RANGE):
            ax       = axes[row, col]
            lam_ladder = _lam_ladder(lam, n_replicas)

            samples_ladder = ddpm_tsr_swapped(
                model, dataset_shape, lam, lam_ladder,
                replica_swaps=replica_swaps,
                swap_algorithm=swap_algorithm,
            )

            samples_np = samples_ladder[lam].detach().cpu().numpy().reshape(-1)
            pdf        = compute_mixture_pdf(dataset_name, x_axis, lam=lam)

            ax.hist(samples_np, bins=bins, density=True,
                    color=COLOR_SAMPLES, alpha=0.4, linewidth=0, label="Samples")
            ax.plot(x_axis, pdf,
                    color=COLOR_PDF, linewidth=1.6, linestyle="--", label="Target")

            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", alpha=0.15, linewidth=0.5)
            ax.tick_params(labelsize=8)
            ax.xaxis.set_major_locator(ticker.MaxNLocator(5, integer=True))
            ax.yaxis.set_major_locator(ticker.MaxNLocator(4))

            if row == 0:
                ax.set_title(fr"$\lambda = {lam:.2f}$", fontsize=9, fontweight="normal", pad=4)
            if row == 1:
                ax.set_xlabel("$x$", fontsize=9)
            if col == 0:
                ax.set_ylabel(f"\\textit{{{dataset_name}}}\ndensity", fontsize=9)

    # single legend top-right panel
    axes[0, -1].legend(
        fontsize=8, frameon=False,
        handlelength=1.5, handletextpad=0.5,
    )

    method = "P-TSR" if replica_swaps else "TSR"
    fig.suptitle(method, fontsize=10, fontweight="normal", y=1.01)

    plt.tight_layout(h_pad=0.6, w_pad=0.3)
    return fig, axes