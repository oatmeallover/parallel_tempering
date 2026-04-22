import torch
import numpy as np
import matplotlib.pyplot as plt
import math
import random

from .schedule import betas, alphas, alpha_bars, ts_desc, compute_tsr_schedule
from .config import DEVICE, DATASETS, CKPT_DIR, N_DIFFUSION_STEPS

torch.manual_seed(42)
np.random.seed(42)

device = DEVICE
ckpt_dir = CKPT_DIR

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
def r_curve_func(x, x_hat, s):
	"""Computes curve where r(0) = x and r(1) = x hat"""
	return x + s * (x_hat - x)


@torch.no_grad()
def r_deriv_func(x, x_hat, s):
	"""Computes analytical derivative of r curve"""
	ones = torch.ones_like(s)
	return ones * (x_hat - x)


@torch.no_grad()
def compute_log_transition_ratio(model, x, x_hat, t, step_size, k, sigma, is_ebm):
	"""Computes log [ k(x | x_hat) / k(x_hat | x) ]"""
	score_x = compute_score(model, x, t, k, sigma, is_ebm) 
	score_x_hat = compute_score(model, x_hat, t, k, sigma, is_ebm) 

	forward_diff = x_hat - x - step_size * score_x 
	forward_sq = - 0.5 * forward_diff**2/ (2.0 * step_size)
	
	backward_diff = x - x_hat - step_size * score_x_hat 
	backward_sq = - 0.5 * backward_diff**2/ (2.0 * step_size)
	
	return backward_sq - forward_sq


@torch.no_grad()
def compute_score_integral(model, x, x_hat, t, k, sigma, is_ebm, n_segments=10): # k closer to target
	"""Computes energy of a noise-based model thrfugh integration"""

	s = torch.linspace(0.0, 1.0, n_segments, device=device)

	r = r_curve_func(x, x_hat, s)
	r_deriv = r_deriv_func(x, x_hat, s)

	r_flat = r.reshape(-1,1)

	score = compute_score(model, r_flat, t, k, sigma, is_ebm).reshape(r.shape[0], -1)

	integrand = score * r_deriv 
	f = torch.trapz(integrand, s, dim=1).unsqueeze(-1)

	return f 


@torch.no_grad() # x is k_2 x hat is k_1
def compute_correction(model, x, x_hat, t, k, k_hat, sigma, is_ebm):
	"""Computes acceptance rate for MALA and returns corrected x"""
	f = compute_score_integral(model, x, x_hat, t, k, sigma, is_ebm)
	f_hat = compute_score_integral(model, x, x_hat, t, k_hat, sigma, is_ebm)
	# log_transition_ratio = compute_log_transition_ratio(model, x, x_hat, t, step_size, k, sigma, is_ebm)

	a = torch.clamp(torch.exp(f - f_hat), max=1.0) 
		
	u = torch.rand_like(a)
	accept_mask = (u < a).float()

	x_new = accept_mask * x_hat + (1 - accept_mask) * x
	x_hat_new = accept_mask * x + (1 - accept_mask) * x_hat

	return x_new, x_hat_new, a


def swap_schedule(t, n, n_replicas, k_ladder, iter=12):
	t_start = 12 # so we end in sets of 2

	if len(k_ladder) == 1 or t < t_start or t % iter != 0:
		return None

	mid = n_replicas // 2

	left = list(range(mid-1, -1, -1))            # [0, 1, 2]
	right = list(range(mid+1, n_replicas))  # 4 5 6

	indices = [x for pair in zip(right, left) for x in pair]  # [0, 6, 1, 5, 2, 4]

	index = int(((N_DIFFUSION_STEPS-t)/iter) % (len(indices) + 2))

	if index == 6:
		return [(k_ladder[1], k_ladder[2])]
	if index == 7:
		return [(k_ladder[5], k_ladder[4])]
	swap_idx = indices[index]
	
	return [(k_ladder[swap_idx], k_ladder[mid])]

@torch.no_grad()
def ula_steps(model, t, dataset_shape, x_ladder, k_ladder, a_ladder, sigma, step_scale, n_langevin_steps, beta_t, n_replicas, is_ebm):

	for n in range(n_langevin_steps):

		for k_val in k_ladder:

			noise = torch.randn(dataset_shape, device=device)
			step_size = step_scale * beta_t

			x_k = x_ladder[k_val].clone()
			score_hat = compute_score(model, x_k, t, k_val, sigma, is_ebm)
			x_k = x_k + step_size * score_hat + torch.sqrt(2.0 * step_size) * noise
			x_ladder[k_val] = x_k.clone()
		
	return x_ladder, a_ladder


@torch.no_grad()
def replica_exchange(model, t, n, dataset_shape, x_ladder, k_ladder, a_ladder, step_scale, n_langevin_steps, n_replicas, sigma, beta_t, is_ebm):

	pairs = swap_schedule(t, n, n_replicas, k_ladder)

	if pairs is not None:

		for k_target, k_s in pairs: 
			
			x_s = x_ladder[k_s].clone()
			x_target = x_ladder[k_target].clone()

			x_target_swapped, x_s_swapped, accept_mask = compute_correction(
				model, x_target, x_s, t, k_target, k_s, sigma, is_ebm
			)

			x_ladder[k_s] = x_s_swapped
			x_ladder[k_target] = x_target_swapped

			key = tuple(sorted((k_target, k_s)))
			if key not in a_ladder:
				a_ladder[key] = {}
			a_ladder[key][t] = accept_mask
	
	return x_ladder, a_ladder


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