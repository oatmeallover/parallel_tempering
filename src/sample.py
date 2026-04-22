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
	"""Computes curve where r(0) = x and r(1) = x_hat.
	
	Args:
		x, x_hat: (bs, D)
		s:        (n_segments,)
	Returns:
		r:        (n_segments, bs, D)
	"""
	# s: (n_segments, 1, 1) to broadcast over (bs, D)
	s = s.view(-1, 1, 1)
	return x + s * (x_hat - x)  # (n_segments, bs, D)


@torch.no_grad()
def r_deriv_func(x, x_hat, s):
	"""Computes analytical derivative of r curve.
	
	Args:
		x, x_hat: (bs, D)
		s:        (n_segments,)
	Returns:
		r_deriv:  (n_segments, bs, D)
	"""
	# Derivative is constant w.r.t. s, tile across segment dim
	diff = (x_hat - x).unsqueeze(0)          # (1, bs, D)
	return diff.expand(s.shape[0], -1, -1)   # (n_segments, bs, D)


@torch.no_grad()
def compute_log_transition_ratio(model, x, x_hat, t, step_size, k, sigma):
	"""Computes log [ k(x | x_hat) / k(x_hat | x) ]"""
	score_x = compute_score(model, x, t, k, sigma) 
	score_x_hat = compute_score(model, x_hat, t, k, sigma) 

	forward_diff = x_hat - x - step_size * score_x 
	forward_sq = - 0.5 * forward_diff**2/ (2.0 * step_size)
	
	backward_diff = x - x_hat - step_size * score_x_hat 
	backward_sq = - 0.5 * backward_diff**2/ (2.0 * step_size)
	
	return backward_sq - forward_sq


@torch.no_grad()
def compute_score_integral(model, x, x_hat, t, k, sigma, n_segments=10):
	"""Computes energy of a noise-based model through integration."""
	original_shape = x.shape
	bs = original_shape[0]

	x_flat     = x.reshape(bs, -1)
	x_hat_flat = x_hat.reshape(bs, -1)

	s = torch.linspace(0.0, 1.0, n_segments, device=x.device)

	r       = r_curve_func(x_flat, x_hat_flat, s)       # (n_segments, bs, D)
	r_deriv = r_deriv_func(x_flat, x_hat_flat, s)       # (n_segments, bs, D)

	# r shape torch.Size([10, 4, 784])
	# r deriv shape torch.Size([10, 4, 784])

	n_seg, _, D = r.shape

	r_in = r.reshape(n_seg * bs, *original_shape[1:])

	score = compute_score(model, r_in, t, k, sigma).reshape(n_seg, bs, D)

	integrand = score * r_deriv

	f_flat = torch.trapz(integrand, s, dim=0)            # (bs, D)

	f = f_flat.reshape(original_shape)

	return f

@torch.no_grad() # x is k_2 x hat is k_1
def compute_correction(model, x, x_hat, t, k, k_hat, sigma):
	"""Computes acceptance rate for MALA and returns corrected x"""
	f = compute_score_integral(model, x, x_hat, t, k, sigma)
	f_hat = compute_score_integral(model, x, x_hat, t, k_hat, sigma)
	# log_transition_ratio = compute_log_transition_ratio(model, x, x_hat, t, step_size, k, sigma)

	a = torch.clamp(torch.exp(f - f_hat), max=1.0) 
		
	u = torch.rand_like(a)
	accept_mask = (u < a).float()

	x_new = accept_mask * x_hat + (1 - accept_mask) * x
	x_hat_new = accept_mask * x + (1 - accept_mask) * x_hat

	return x_new, x_hat_new, a

def swap_schedule(t, k_ladder, iter=20):

	if len(k_ladder) == 1 or t < 50:
		return None

	indices = list(range(len(k_ladder) - 1))  # pairs: (0,1), (1,2), (2,3), ...

	if t % iter == 0:
		even_pairs = [(i, i+1) for i in indices if i % 2 == 0]
		return even_pairs
	elif (t - 10) % iter == 0:
		odd_pairs = [(i, i+1) for i in indices if i % 2 == 1]
		return odd_pairs

	return None

@torch.no_grad()
def ula_steps(model, t, dataset_shape, x_ladder, k_ladder, a_ladder, sigma, step_scale, n_langevin_steps, beta_t, n_replicas):

	for n in range(n_langevin_steps):

		for k_val in k_ladder:

			noise = torch.randn(dataset_shape, device=device)
			step_size = step_scale * beta_t

			x_k = x_ladder[k_val].clone()
			score_hat = compute_score(model, x_k, t, k_val, sigma)
			x_k = x_k + step_size * score_hat + torch.sqrt(2.0 * step_size) * noise
			x_ladder[k_val] = x_k.clone()
		
	return x_ladder, a_ladder


@torch.no_grad()
def replica_exchange(model, t, x_ladder, k_ladder, sigma):

	pairs = swap_schedule(t, k_ladder)

	if pairs is not None:

		for k_targ_ind, k_s_ind in pairs: 

			k_target = k_ladder[k_targ_ind]
			k_s = k_ladder[k_s_ind]

			x_s = x_ladder[k_s].clone()
			x_target = x_ladder[k_target].clone()

			x_target_swapped, x_s_swapped, accept_mask = compute_correction(
				model, x_target, x_s, t, k_target, k_s, sigma
			)

			print(f"Time {t} swap {accept_mask.mean().item()}")

			x_ladder[k_s] = x_s_swapped
			x_ladder[k_target] = x_target_swapped

	return x_ladder


def ddpm_tsr_swapped(model, dataset_shape, ks, sigma=1.0, replica_swaps=False):

	x_ladder = {k_val: torch.randn(dataset_shape, device=device) for k_val in ks}
		
	for t in ts_desc: 

		alpha_t = alphas[t]
		beta_t = betas[t]
		sqrt_alpha_t = torch.sqrt(alpha_t)
		sqrt_beta_t = torch.sqrt(beta_t)

		for k_val in ks:
			x = x_ladder[k_val].clone()
			noise = torch.randn(dataset_shape, device=device)
			score_hat = compute_score(model, x, t, k_val, sigma)
			x = (x + beta_t * score_hat) / sqrt_alpha_t + sqrt_beta_t * noise
			x_ladder[k_val] = x.clone()

		if replica_swaps:
			x_ladder = replica_exchange(model, t, x_ladder, ks, sigma)

	return x_ladder
