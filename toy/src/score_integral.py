import torch
from .model import compute_score
from .schedule import compute_tsr_schedule
import matplotlib.pyplot as plt
import scipy
import numpy as np
from .schedule import betas, alphas, alpha_bars, ts_desc, compute_tsr_schedule

@torch.no_grad()
def _lam_ladder(tsr_lam, n_replicas):
	scale = 0.9
	return np.array([tsr_lam, tsr_lam / scale , tsr_lam /(scale**2)])


@torch.no_grad()
def r_curve_func(x, x_hat, s):
	s = s.view(-1, 1, 1)
	return x + s * (x_hat - x)  


@torch.no_grad()
def r_deriv_func(x, x_hat, s): 
	diff = (x_hat - x).unsqueeze(0)         
	return diff.expand(s.shape[0], -1, -1)   

@torch.no_grad()
def compute_score_integral(model, target, source, t, swap_algorithm, second_energy=False, n_segments=8):

	x_t, lam_t = target
	x_s, lam_s = source

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

	f = - f_flat.reshape(original_shape) # p(x_t) / p(x_s)

	return f 
	

@torch.no_grad() 
def mixture_pdf(x, lam):
	means = torch.tensor([-3.0, 0.0, 3.0], device=x.device)
	stds  = torch.tensor([ 0.5, 1.0, 0.5], device=x.device) / torch.sqrt(torch.tensor(lam))
	weight = 1/3

	components = torch.exp(-0.5 * ((x - means) / stds)**2) / (stds * (2 * torch.pi)**0.5)
	return weight * components.sum(dim=-1)


@torch.no_grad()
def analytical_energy(target, source, swap_algorithm, second_energy=False):
	x_t, lam_t = target
	x_s, lam_s = source

	if swap_algorithm["p_ratio"] == "s":
		if swap_algorithm["debug"]: print(f"p_{lam_s:.2f} (x_{lam_t:.2f}) \n---------------\np_{lam_s:.2f} (x_{lam_s:.2f})")
		return (mixture_pdf(x_t, lam_s) / mixture_pdf(x_s, lam_s))
	elif swap_algorithm["p_ratio"] == "t":
		if swap_algorithm["debug"]: print(f"p_{lam_t:.2f} (x_{lam_s:.2f}) \n---------------\np_{lam_t:.2f} (x_{lam_t:.2f})")
		return (mixture_pdf(x_s, lam_t) / mixture_pdf(x_t, lam_t))
	elif swap_algorithm["p_ratio"] == "p":
		if swap_algorithm["debug"]: print(f"p_{lam_t:.2f} (x_{lam_s:.2f}) p_{lam_s:.2f} (x_{lam_t:.2f}) \n------------------------------\np_{lam_t:.2f} (x_{lam_t:.2f}) p_{lam_s:.2f} (x_{lam_s:.2f})")
		return (mixture_pdf(x_t, lam_s) * mixture_pdf(x_s, lam_t)) / ( mixture_pdf(x_s, lam_s) * mixture_pdf(x_t, lam_t) )
		

@torch.no_grad() 
def compute_correction(model, target, source, t, swap_algorithm):
	
	x_t, lam_t = target
	x_s, lam_s = source

	f = compute_score_integral(model, target, source, t, swap_algorithm) 
	temp_t = compute_tsr_schedule(lam_t, t)
	temp_s = compute_tsr_schedule(lam_s, t)

	diff = (1/temp_t)
	print(f" diff {diff.mean().item()}")
 
	f_diff = torch.clamp(f*diff, max=0)

	a = torch.exp(f_diff) # clamp in log space, safer

	x_np = x_t.flatten().cpu().numpy()
	means, edges, _ = scipy.stats.binned_statistic(x_np, a.flatten().cpu().numpy(), bins=100)
	plt.scatter((edges[:-1]+edges[1:])/2, means)
	plt.title(f"Time {t.item()} targ = {lam_t} and sour= {lam_s}")
	plt.scatter([-3,0,3], [1,1,1])
	plt.legend()
	plt.show()

	print(f"p ratio.         {a.mean().item()} std {a.std().item()}")

	u = torch.rand_like(a)
	accept_mask = (u < a).float()
	return accept_mask