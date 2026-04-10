import torch
import numpy as np
import matplotlib.pyplot as plt

from .schedule import betas, alphas, alpha_bars, ts_desc, compute_tsr_schedule
from .config import DEVICE, DATASETS, CKPT_DIR

torch.manual_seed(42)
np.random.seed(42)

device = DEVICE
ckpt_dir = CKPT_DIR

# ---------- sampling utilities (compute_score, compute_tsr, etc.) ----------
@torch.no_grad()
def compute_score(model, x, t, k, sigma, dataset_config):
	"""Computes score = - epsilon * temp / √(1 - α_bar)"""

	ones = torch.ones_like(x, device=device)
	eps_hat = model(x, t * ones)   

	a_bar = alpha_bars[t]

	temp_t = compute_tsr_schedule(k, sigma, t)

	return - eps_hat * temp_t / torch.sqrt(1.0 - a_bar)
	

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
def compute_log_transition_ratio(model, x, x_hat, t, step_size, k, sigma, dataset_config):
	"""Computes log [ k(x | x_hat) / k(x_hat | x) ]"""
	score_x = compute_score(model, x, t, k, sigma, dataset_config) 
	score_x_hat = compute_score(model, x_hat, t, sigma, dataset_config) 

	forward_diff = x_hat - x - step_size * score_x 
	forward_sq = - 0.5 * forward_diff**2/ (2.0 * step_size)
	
	backward_diff = x - x_hat - step_size * score_x_hat 
	backward_sq = - 0.5 * backward_diff**2/ (2.0 * step_size)
	
	return backward_sq - forward_sq


@torch.no_grad()
def compute_score_integral(model, x, x_hat, t, k, sigma, dataset_config, n_segments=5): # k closer to target
	"""Computes energy of a noise-based model thrfugh integration"""
	a_bar = alpha_bars[t]

	s = torch.linspace(0.0, 1.0, n_segments, device=device)

	r = r_curve_func(x, x_hat, s)
	r_deriv = r_deriv_func(x, x_hat, s)

	r_flat = r.reshape(-1,1) # B*S, 1
	t_batch = t * torch.ones_like(r_flat)

	eps_hat = model(r_flat, t_batch).reshape(r.shape[0], -1)

	integrand = k * eps_hat * r_deriv 

	temp_t = compute_tsr_schedule(k, sigma, t)

	f = torch.trapz(integrand, s, dim=1).unsqueeze(-1)
	f = - f * temp_t / torch.sqrt(1.0 - a_bar) 

	return f 


@torch.no_grad()
def compute_correction(model, x, x_hat, t, step_size, k, sigma, dataset_config):
	"""Computes acceptance rate for MALA and returns corrected x"""
	f = compute_score_integral(model, x, x_hat, t, k, sigma, dataset_config)
	log_transition_ratio = compute_log_transition_ratio(model, x, x_hat, t, step_size, k, sigma, dataset_config)

	a = torch.clamp(torch.exp(f + log_transition_ratio), max=1.0) # add a_bar back
	
	u = torch.rand_like(a)
	accept_mask = (u < a).float()
	x = accept_mask * x_hat + (1 - accept_mask) * x

	return x, accept_mask


# ---------- main sampling function (sampling) ----------
@torch.no_grad()
def sampling(model, dataset_config, method, k=1.0, sigma=1.0, step_scale=5, n_langevin_steps=3, debug=True, log_freq=10):
	"""Sampling algorithm for DDPM, ULA, and MALA"""

	dataset_shape = dataset_config["dataset_shape"]
	x = torch.randn(dataset_shape, device=device)

	ladder_length = 3

	k_ladder = np.linspace(1.0, k, ladder_length)

	for t in ts_desc:

		if t % log_freq == 0 and debug==True:
			print(f"\ntime {t}: x has standard deviation {x.std().item():.2f}")

		alpha_t = alphas[t]
		beta_t = betas[t]
		sqrt_alpha_t = torch.sqrt(alpha_t)
		sqrt_beta_t = torch.sqrt(beta_t)

		score_hat = compute_score(model, x, t, k, sigma, dataset_config)

		noise = torch.randn_like(x, device=device)
		x = (x + beta_t * score_hat) / sqrt_alpha_t + sqrt_beta_t * noise

		if method in ["ULA", "MALA"]:

			step_size = beta_t * torch.tensor(step_scale,  device=device)
			
			x_ladder = {k_val: x.clone() for k_val in k_ladder}

			for n in range(n_langevin_steps):

				if t % log_freq == 0 and debug == True:
					print(f"\nLangevin step {n}")

				for k_val in k_ladder:

					score_hat = compute_score(model, x_ladder[k_val], t, k_val, sigma, dataset_config)

					noise = torch.randn_like(x_ladder[k_val])
					x_hat = x_ladder[k_val] + step_size * score_hat + torch.sqrt(2.0 * step_size) * noise	

					# if method == "MALA":
					# 	x_hat, accept_mask = compute_correction(model, x, x_hat, t, step_size, k, sigma, dataset_config)

					# 	if t % log_freq == 0 and debug==True:
					# 		print(f"time {t} acceptance {accept_mask.mean().item()}")

					x_ladder[k_val] = x_hat

				for i in range(ladder_length-1): # odd then even

					k_more = k_ladder[i+1]
					k_less = k_ladder[i]

					x_more_temp = x_ladder[k_more]
					x_less_temp = x_ladder[k_less]
					
					x_less_temp_corrected, accept_mask = compute_correction(model, x_less_temp, x_more_temp, t, step_size, k_more, sigma, dataset_config)
					x_more_temp_corrected = accept_mask * x_less_temp + (1 - accept_mask) * x_more_temp

					x_ladder[k_more] = x_more_temp_corrected
					x_ladder[k_less] = x_less_temp_corrected

					if t % log_freq == 0 and debug == True:
						print(f"time {t} acceptance {accept_mask.mean().item()}")

			x = x_ladder[k]

	return x

