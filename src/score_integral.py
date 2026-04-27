import torch
from .model import compute_score
from .schedule import compute_tsr_schedule
import matplotlib.pyplot as plt
import scipy
import numpy as np


@torch.no_grad()
def _k_ladder(tsr_k, n_replicas):
    k = float(tsr_k)
    half = int(n_replicas // 2)
    bottom = np.linspace(1.0/k, 1.0, half + 1)  # [1/k, ..., 1.0]
    top = np.linspace(1.0, k, half + 1)          # [1.0, ..., k]
    return np.concat([bottom, top[1:]])  # drop duplicate 1.0 from top


@torch.no_grad()
def r_curve_func(x, x_hat, s):
	s = s.view(-1, 1, 1)
	return x + s * (x_hat - x)  


@torch.no_grad()
def r_deriv_func(x, x_hat, s): 
	diff = (x_hat - x).unsqueeze(0)         
	return diff.expand(s.shape[0], -1, -1)   


@torch.no_grad()
def compute_log_transition_ratio(model, x, x_hat, t, step_size, k):
	"""Computes log [ k(x | x_hat) / k(x_hat | x) ]"""
	score_x = compute_score(model, x, t, k) 
	score_x_hat = compute_score(model, x_hat, t, k) 

	forward_diff = x_hat - x - step_size * score_x 
	forward_sq = - 0.5 * forward_diff**2/ (2.0 * step_size)
	
	backward_diff = x - x_hat - step_size * score_x_hat 
	backward_sq = - 0.5 * backward_diff**2/ (2.0 * step_size)
	
	return backward_sq - forward_sq


@torch.no_grad()
def compute_score_integral(model, target, source, t, swap_algorithm, second_energy=False, n_segments=8):

	x_t, k_t = target
	x_s, k_s = source

	x = x_s
	x_hat = x_t 

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

	score = compute_score(model, r_in, t, 1.0).reshape(n_seg, bs, D)

	integrand = score * r_deriv

	f_flat = torch.trapz(integrand, s, dim=0)            # (bs, D)

	f = - f_flat.reshape(original_shape) # p(x_s) / p(x_t)

	if swap_algorithm["p_ratio"] == "s":
		temp_s = compute_tsr_schedule(k_s, t)
		return f * temp_s
	elif swap_algorithm["p_ratio"] == "t":
		temp_t = compute_tsr_schedule(k_t, t)
		return - f * temp_t
	elif swap_algorithm["p_ratio"] == "p":
		temp_t = compute_tsr_schedule(k_t, t)
		temp_s = compute_tsr_schedule(k_s, t)
		return f * (temp_s - temp_t) 
	

@torch.no_grad() 
def mixture_pdf(x, k):
    means = torch.tensor([-3.0, 0.0, 3.0], device=x.device)
    stds  = torch.tensor([ 0.5, 1.0, 0.5], device=x.device) / torch.sqrt(torch.tensor(k))
    weight = 1/3

    components = torch.exp(-0.5 * ((x - means) / stds)**2) / (stds * (2 * torch.pi)**0.5)
    return weight * components.sum(dim=-1)


@torch.no_grad()
def analytical_energy(target, source, swap_algorithm, second_energy=False):
	x_t, k_t = target
	x_s, k_s = source

	if swap_algorithm["p_ratio"] == "s":
		if swap_algorithm["debug"]: print(f"p_{k_s:.2f} (x_{k_t:.2f}) \n---------------\np_{k_s:.2f} (x_{k_s:.2f})")
		return (mixture_pdf(x_t, k_s) / mixture_pdf(x_s, k_s))
	elif swap_algorithm["p_ratio"] == "t":
		if swap_algorithm["debug"]: print(f"p_{k_t:.2f} (x_{k_s:.2f}) \n---------------\np_{k_t:.2f} (x_{k_t:.2f})")
		return (mixture_pdf(x_s, k_t) / mixture_pdf(x_t, k_t))
	elif swap_algorithm["p_ratio"] == "p":
		if swap_algorithm["debug"]: print(f"p_{k_t:.2f} (x_{k_s:.2f}) p_{k_s:.2f} (x_{k_t:.2f}) \n------------------------------\np_{k_t:.2f} (x_{k_t:.2f}) p_{k_s:.2f} (x_{k_s:.2f})")
		return (mixture_pdf(x_t, k_s) * mixture_pdf(x_s, k_t)) / ( mixture_pdf(x_s, k_s) * mixture_pdf(x_t, k_t) )
		

@torch.no_grad() 
def compute_correction(model, target, source, t, swap_algorithm):
	
	x_t, k_t = target
	x_s, k_s = source

	f = compute_score_integral(model, target, source, t, swap_algorithm) 
	a = torch.clamp(torch.exp(f), max = 1.0)

	if swap_algorithm["debug"]:

		x_np = x_s.flatten().cpu().numpy()
		means, edges, _ = scipy.stats.binned_statistic(x_np, a.flatten().cpu().numpy(), bins=100)
		plt.scatter((edges[:-1]+edges[1:])/2, means, label='Acceptances')
		plt.scatter([-3,0,3], [1,1,1], label="Distribution modes")
		plt.title(f"Time {t.item()} k target = {k_t} and source {k_s}")
		plt.legend()
		plt.show()

	u = torch.rand_like(a)
	accept_mask = (u < a).float()
	return accept_mask