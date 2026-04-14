import torch
import numpy as np
import matplotlib.pyplot as plt

from .schedule import betas, alphas, alpha_bars, ts_desc, compute_tsr_schedule, cosine_beta_schedule
from .config import DEVICE, DATASETS, CKPT_DIR

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
def compute_score_integral(model, x, x_hat, t, k, sigma, is_ebm, n_segments=5): # k closer to target
	"""Computes energy of a noise-based model thrfugh integration"""

	s = torch.linspace(0.0, 1.0, n_segments, device=device)

	r = r_curve_func(x, x_hat, s)
	r_deriv = r_deriv_func(x, x_hat, s)

	r_flat = r.reshape(-1,1)

	score = compute_score(model, r_flat, t, k, sigma, is_ebm).reshape(r.shape[0], -1)

	integrand = score * r_deriv 
	f = torch.trapz(integrand, s, dim=1).unsqueeze(-1)

	return f 


@torch.no_grad()
def compute_correction(model, x, x_hat, t, step_size, k, sigma, is_ebm):
	"""Computes acceptance rate for MALA and returns corrected x"""
	f = compute_score_integral(model, x, x_hat, t, k, sigma, is_ebm)
	log_transition_ratio = compute_log_transition_ratio(model, x, x_hat, t, step_size, k, sigma, is_ebm)

	a = torch.clamp(torch.exp(f + log_transition_ratio), max=1.0) # add a_bar back
	
	u = torch.rand_like(a)
	accept_mask = (u < a).float()

	x_new = accept_mask * x_hat + (1 - accept_mask) * x
	x_hat_new = accept_mask * x + (1 - accept_mask) * x_hat

	return x_new, x_hat_new, a


# ---------- main sampling function (sampling) ----------
@torch.no_grad()
def sampling(model, dataset_shape, k=1.0, sigma=1.0, step_scale=1, n_langevin_steps=0, n_replicas=1, k_ladder=None, is_ebm = False, debug=True):
	"""Sampling algorithm for DDPM, ULA, and MALA"""

	x_initial = torch.randn(dataset_shape, device=device)

	if k_ladder is None: k_ladder = np.linspace(k, 1.0, n_replicas)
	x_ladder = {k_val: x_initial.clone() for k_val in k_ladder}
	a_ladder = {(k_ladder[i], k_ladder[i+1]): [] for i in range(len(k_ladder) - 1)}
	median_k = sorted(k_ladder)[len(k_ladder) // 2]

	if debug==True: 
		
		print(f"We will be running {n_replicas} replicas within k values {k_ladder}")
		# n_diffusion_steps = 12
		# betas = cosine_beta_schedule(n_diffusion_steps).to(device)
		# alphas = 1.0 - betas
		# ts_desc = torch.arange(n_diffusion_steps - 1, -1, -1, device=device)

		time_prints = "  t        : "
		std_prints = {k_val:  f"  Chain {k_val:>2.1f}: " for k_val in k_ladder}
		swap_prints = {k_val:  f"  Swap   ^ : " for k_val in k_ladder[0:]}
		for k_val in k_ladder:
			swap_prints.setdefault(k_val, "")
			   
	for t in ts_desc:

		time_prints += f"{t:>5}  | "

		alpha_t = alphas[t]
		beta_t = betas[t]
		sqrt_alpha_t = torch.sqrt(alpha_t)
		sqrt_beta_t = torch.sqrt(beta_t)
		step_size = beta_t * torch.tensor(step_scale, device=device)

		for k_val in k_ladder:

			score_hat = compute_score(model, x_ladder[k_val], t, k_val, sigma, is_ebm)
			noise = torch.randn(dataset_shape, device=device)
			x_ladder[k_val] = (x_ladder[k_val] + beta_t * score_hat) / sqrt_alpha_t + sqrt_beta_t * noise

			std_prints[k_val] += f"{x_ladder[k_val].std().item():5.2f} -> "

			for n in range(n_langevin_steps):
				score_hat = compute_score(model, x_ladder[k_val], t, k_val, sigma, is_ebm)
				noise = torch.randn_like(x_ladder[k_val])
				x_ladder[k_val] = x_ladder[k_val] + step_size * score_hat + torch.sqrt(2.0 * step_size) * noise
			
		updated = set()

		start = t % 2  # 0 if n even, 1 if n odd

		for i in range(start, len(k_ladder) - 1, 2):

			k_1 = k_ladder[i+1]
			k_2 = k_ladder[i] # more tempered

			k_target = min(k_1, k_2, key=lambda k: abs(k - median_k))

			x_1_temp = x_ladder[k_1]
			x_2_temp = x_ladder[k_2]
			
			x_2_temp, x_1_temp, accept_mask = compute_correction(model, x_2_temp, x_1_temp, t, step_size, k_target, sigma, is_ebm)

			a_ladder[(k_2, k_1)].append({
				"t": t,
				"acceptance": accept_mask
			})

			x_ladder[k_1] = x_1_temp
			x_ladder[k_2] = x_2_temp

			swap_prints[k_1] += f"    {accept_mask.mean().item():5.2f}"
			updated.add(k_1)

		for k_val in k_ladder:
			if k_val not in updated:
				swap_prints[k_val] += " " * 9

		if t % 25 ==0 and debug==True:
			print(time_prints)
			for k_val in k_ladder:
				if k_val != k_ladder[0]: print(swap_prints[k_val])
				print(std_prints[k_val])
			print("\n")

			time_prints = "  t        : "
			std_prints = {k_val:  f"  Chain {k_val:>2.1f}: " for k_val in k_ladder}
			swap_prints = {k_val:  f"  Swap   ^ : " for k_val in k_ladder[0:]}
			for k_val in k_ladder:
				swap_prints.setdefault(k_val, "")

	return x_ladder, a_ladder

