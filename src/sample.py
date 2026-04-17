import torch
import numpy as np
import matplotlib.pyplot as plt
import math
import random

from .schedule import betas, alphas, alpha_bars, ts_desc, compute_tsr_schedule, swap_schedule_even, swap_schedule_odd
from .config import DEVICE, DATASETS, CKPT_DIR, N_DIFFUSION_STEPS

torch.manual_seed(42)
np.random.seed(42)

device = DEVICE
ckpt_dir = CKPT_DIR

# ---------- sampling utilities (compute_score, compute_tsr, etc.) ----------
@torch.no_grad()
def compute_score(model, x, t, k, sigma, is_ebm=False):
	"""Computes score = - epsilon * temp / √(1 - α_bar)"""

	x_shape = x.shape
	ones = torch.ones((x_shape[0], 1), device=device)
	eps_hat = model(x, t * ones)   
	a_bar = alpha_bars[t]
	temp_t = compute_tsr_schedule(k, sigma, t)

	if is_ebm == False:
		score_hat= - eps_hat * temp_t / torch.sqrt(1.0 - a_bar)

	else:
		print("Taking grad")
		score_hat = - temp_t * torch.autograd.grad(
			outputs=eps_hat,
			inputs=x,
			grad_outputs=torch.ones_like(eps_hat),
			create_graph=False,       # keep graph if you need higher-order grads
			retain_graph=False        # if you will use output again
		)[0]

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
def compute_correction(model, x, x_hat, t, step_size, k, k_hat, sigma, is_ebm):
	"""Computes acceptance rate for MALA and returns corrected x"""
	f = compute_score_integral(model, x, x_hat, t, 1.0, sigma, is_ebm)

	print(f"f avg: {f.mean().item():.4f}")
	print(f"f std: {f.std().item():.4f}")

	#log_transition_ratio = compute_log_transition_ratio(model, x, x_hat, t, step_size, k, sigma, is_ebm)

	temp = compute_tsr_schedule(k, sigma, t)
	temp_hat = compute_tsr_schedule(k_hat, sigma, t)

	a = torch.clamp(torch.exp(f * (temp - temp_hat) ), max=1.0) 
	
	print(f"Time {t}: Swap {k} and {k_hat} tsr diff {(temp - temp_hat).mean().item()} mean acceptance {a.mean().item()}")
	
	u = torch.rand_like(a)
	accept_mask = (u < a).float()

	x_new = accept_mask * x_hat + (1 - accept_mask) * x
	x_hat_new = accept_mask * x + (1 - accept_mask) * x_hat

	return x_new, x_hat_new, a

@torch.no_grad() # when to incorporate swaps
def swap_probability(t, decay=0.1):
	return math.exp(-decay * t)


# ---------- main sampling function (sampling) ----------
@torch.no_grad()
def sampling(model, dataset_shape, k=1.0, sigma=1.0, step_scale=1, n_langevin_steps=0, n_replicas=1, swap_iter = 25, k_ladder=None, is_ebm = False, debug=True):
	"""Sampling algorithm for DDPM, ULA, and MALA"""

	x_initial = torch.randn(dataset_shape, device=device)

	if k_ladder is None: k_ladder = np.linspace(k, 1/k, n_replicas)
	even_pairs = [(k_ladder[i], k_ladder[i+1]) for i in range(0, len(k_ladder)-1, 2)]
	odd_pairs  = [(k_ladder[i], k_ladder[i+1]) for i in range(1, len(k_ladder)-1, 2)]

	if debug==True: 
		
		print(f"Running with {N_DIFFUSION_STEPS} steps for {k_ladder}")

		time_prints = "  t        : "
		std_prints = {k_val:  f"  Chain {k_val:>2.1f}: " for k_val in k_ladder}
		swap_prints = {k_val:  f"  Swap   ^ : " for k_val in k_ladder[0:]}
		for k_val in k_ladder:
			swap_prints.setdefault(k_val, "")
	
	ts_desc_init = torch.arange(N_DIFFUSION_STEPS, -1, -1, device=device)
	x_ladder = {t.item(): {k_val.item(): x_initial.clone()  for k_val in k_ladder} for t in ts_desc_init}

	a_ladder = {(k_ladder[i], k_ladder[i+1]): [] for i in range(len(k_ladder) - 1)}
	
	for t in ts_desc: # goes from 11 to 0

		if debug==True: time_prints += f"{t:>5}  | "

		alpha_t = alphas[t]
		beta_t = betas[t]
		sqrt_alpha_t = torch.sqrt(alpha_t)
		sqrt_beta_t = torch.sqrt(beta_t)
		step_size = beta_t * torch.tensor(step_scale, device=device)
		noise = torch.randn(dataset_shape, device=device)

		for k_val in k_ladder:

			x_t_k = x_ladder[t.item()][k_val].clone()
			score_hat = compute_score(model, x_t_k, t, k_val, sigma, is_ebm)
			x_t_k = (x_t_k + beta_t * score_hat) / sqrt_alpha_t + sqrt_beta_t * noise

			if debug==True: std_prints[k_val] += f"{x_t_k.std().item():5.2f} -> "

			# for n in range(n_langevin_steps):
			# 	score_hat = compute_score(model, x_t_k, t, k_val, sigma, is_ebm)
			# 	noise = torch.randn_like(x_t_k)
			# 	x_t_k = x_t_k + step_size * score_hat + torch.sqrt(2.0 * step_size) * noise

			x_ladder[t.item()][k_val] = x_t_k.clone()
		
		if debug==True: updated = set()

		if t != 0 and (t.item() in swap_schedule_even or t.item() in swap_schedule_odd):
			pairs = even_pairs if (t.item() in swap_schedule_even) else odd_pairs

			for k_2, k_1 in pairs:  # k_2 is more tempered (hotter)
				x_1_temp = x_ladder[t.item()][k_1]
				x_2_temp = x_ladder[t.item()][k_2]

				x_2_temp, x_1_temp, accept_mask = compute_correction(
					model, x_2_temp, x_1_temp, t, step_size, k_2, k_1, sigma, is_ebm
				)

				a_ladder[(k_2, k_1)].append({"t": t, "acceptance": accept_mask})

				x_ladder[t.item()][k_1] = x_1_temp
				x_ladder[t.item()][k_2] = x_2_temp

				swap_prints[k_1] += f"    {accept_mask.mean().item():5.2f}"
				updated.add(k_1)

		if t!= ts_desc[-1]:
			for k_val in k_ladder:
				x_ladder[(t-1).item()][k_val] = x_ladder[t.item()][k_val].clone()
				
		if debug == True:
			for k_val in k_ladder:
				if k_val not in updated:
					swap_prints[k_val] += " " * 9

			if t % 25 ==0:
				print(time_prints)
				for k_val in k_ladder:
					#if k_val != k_ladder[0]: print(swap_prints[k_val])
					print(std_prints[k_val])
				print("\n")

				time_prints = "  t        : "
				std_prints = {k_val:  f"  Chain {k_val:>2.1f}: " for k_val in k_ladder}
				swap_prints = {k_val:  f"  Swap   ^ : " for k_val in k_ladder[0:]}
				for k_val in k_ladder:
					swap_prints.setdefault(k_val, "")

	return x_ladder, a_ladder

