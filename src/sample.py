import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy

from .schedule import betas, alphas, alpha_bars, ts_desc, compute_tsr_schedule
from .config import DEVICE, CKPT_DIR
from .score_integral import compute_score_integral
from .model import compute_score

torch.manual_seed(42)
np.random.seed(42)

device = DEVICE
ckpt_dir = CKPT_DIR

@torch.no_grad() 
def unnormalized(z):
	sqrt2 = torch.tensor(2.0).sqrt()
	return (torch.exp(-z**2 / 2)
			+ sqrt2 * torch.exp(-(z + 3)**2)
			+ sqrt2 * torch.exp(-(z - 3)**2))


@torch.no_grad()
def analytical_energy(target, source, swap_algorithm):
	x_t, k_t = target
	x_s, k_s = source

	if swap_algorithm["swap_towards_k"]:
		if swap_algorithm["debug"]: print(f"p_{k_t:.2f} (x_{k_s:.2f}) /\np_{k_t:.2f} (x_{k_t:.2f})")
		return (unnormalized(x_s) / unnormalized(x_t))**k_t
	else:
		if swap_algorithm["debug"]: print(f"p_{k_s:.2f} (x_{k_t:.2f}) /\np_{k_s:.2f} (x_{k_s:.2f})")
		return (unnormalized(x_t) / unnormalized(x_s))**k_s


@torch.no_grad() 
def compute_correction(model, target, source, t, swap_algorithm):
	
	f = compute_score_integral(model, target, source, t, swap_algorithm) # E (x) - E(x hat)  = p(xhat) / p(x)
	a = torch.clamp(torch.exp(f), max = 1.0)  # p (x hat under k) / p(x under k)	

	if swap_algorithm["analytical"]:
		p_ratio = analytical_energy(target, source, swap_algorithm) # p (x hat under k) / p(x under k)
		a_analytical = torch.clamp(p_ratio ,max=1)

		x_t, k_t = target
		x_s, k_s = source
		x_np = x_t.flatten().cpu().numpy()
		for arr, label in [(a_analytical, 'analytical'), (a, 'non_analytical')]:
			means, edges, _ = scipy.stats.binned_statistic(x_np, arr.flatten().cpu().numpy(), bins=100)
			plt.scatter((edges[:-1]+edges[1:])/2, means, label=label)
		plt.scatter([-3,0,3], [1,1,1], label="Distribution modes")
		plt.title(f"Time {t.item()} k target = {k_t} and source {k_s}")
		plt.legend()
		plt.show()

	x_t, k_t = target

	u = torch.rand_like(a)
	accept_mask = (u < a)
	return accept_mask.reshape((-1,) + (1,) * (x_t.dim() - 1))


@torch.no_grad()
def swap_schedule(t, k, k_ladder, x_ladder, swap_algorithm):

	if len(k_ladder) != 3: raise ValueError("Length of k ladder more than 3")
	
	if t in swap_algorithm["even"]:
		pair = (1, 2)
	elif t in swap_algorithm["odd"]:
		pair = (0, 1)
	else:
		return []

	diffs = [abs(k - k_ladder[i]) for i in pair]
	near, far = (0, 1) if diffs[0] < diffs[1] else (1, 0)
	t_idx, s_idx = (pair[near], pair[far]) 

	k_t, k_s = k_ladder[t_idx], k_ladder[s_idx]
	return [((x_ladder[k_t].clone(), k_t), (x_ladder[k_s].clone(), k_s))]

@torch.no_grad()
def replica_exchange(model, t, k, k_ladder, x_ladder, swap_algorithm):

	pairs = swap_schedule(t, k, k_ladder, x_ladder, swap_algorithm)

	for target, source in pairs:

		(x_t, k_t) = target
		(x_s, k_s) = source

		accept_mask = compute_correction(model, target, source, t, swap_algorithm)

		if swap_algorithm["debug"]: print(f"Time {t} swap btwn source {k_s.item():.2f} and target {k_t.item():.2f} accept {accept_mask.sum().item() / 100_000 :.2f} ")
		
		x_ladder[k_t] = torch.where(accept_mask.bool(), x_s, x_t)
		x_ladder[k_s] = torch.where(accept_mask.bool(), x_t, x_s)
	
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