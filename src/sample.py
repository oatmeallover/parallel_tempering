import torch
import numpy as np

from .schedule import betas, alphas, alpha_bars, ts_desc, compute_tsr_schedule
from .config import DEVICE, CKPT_DIR
from .score_integral import compute_score_integral, analytical_energy, compute_correction
from .model import compute_score

torch.manual_seed(42)
np.random.seed(42)

device = DEVICE
ckpt_dir = CKPT_DIR

@torch.no_grad()
def swap_schedule(t, k, k_ladder, x_ladder, swap_algorithm):

	if len(k_ladder) != 3: raise ValueError("Length of k ladder more than 3")
	
	if t in swap_algorithm["even"]:
		pair = (1, 2)
	elif t in swap_algorithm["odd"]:
		pair = (0, 1)
	else:
		return []

	k_t, k_s = k_ladder[pair[1]], k_ladder[pair[0]]
	return [((x_ladder[k_t].clone(), k_t), (x_ladder[k_s].clone(), k_s))]


@torch.no_grad()
def replica_exchange(model, t, k, k_ladder, x_ladder, swap_algorithm):

	pairs = swap_schedule(t, k, k_ladder, x_ladder, swap_algorithm)

	for target, source in pairs:

		(x_t, k_t) = target
		(x_s, k_s) = source

		accept_mask = compute_correction(model, target, source, t, swap_algorithm)

		if swap_algorithm["debug"]: print(f"Time {t} swap btwn source {k_s.item():.2f} and target {k_t.item():.2f} accept {accept_mask.sum().item() / 100_000 :.2f} ")

		x_ladder[k_t] = accept_mask * x_s + (1 - accept_mask) * x_t
		x_ladder[k_s] = accept_mask * x_t + (1 - accept_mask) * x_s
	
	return x_ladder


@torch.no_grad()
def ddpm_tsr_swapped(model, dataset_shape, k, k_ladder=1.0, replica_swaps=False, swap_algorithm=None,):

	x_ladder = {k_val: torch.randn(dataset_shape, device=device) for k_val in k_ladder}
		
	for t in ts_desc: 

		alpha_t = alphas[t]
		beta_t = betas[t]
		sqrt_alpha_t = torch.sqrt(alpha_t)
		sqrt_beta_t = torch.sqrt(beta_t)

		for k_val in k_ladder:
			x = x_ladder[k_val].clone()
			noise = torch.randn(dataset_shape, device=device)
			score_hat = compute_score(model, x, t, k_val)
			x = (x + beta_t * score_hat) / sqrt_alpha_t + sqrt_beta_t * noise
			x_ladder[k_val] = x.clone()

		if replica_swaps:
			x_ladder = replica_exchange(model, t, k, k_ladder, x_ladder, swap_algorithm)

	return x_ladder