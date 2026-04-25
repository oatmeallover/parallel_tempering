import torch
from .model import compute_score

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
def compute_score_integral(model, target, source, t, swap_algorithm, n_segments=8):
	"""Computes E (x) - E(x hat) = p(x hat) / p(x)"""

	if swap_algorithm["swap_towards_k"]: # p (source) / p (target)
		x, k = target
		x_hat, k_hat= source
	else: # p (target) / p (source)
		x, k = source
		x_hat, k_hat= target

	original_shape = x.shape
	bs = original_shape[0]

	x_flat     = x.reshape(bs, -1)
	x_hat_flat = x_hat.reshape(bs, -1)

	s = torch.linspace(0.0, 1.0, n_segments, device=x.device)

	r       = r_curve_func(x_flat, x_hat_flat, s)       # (n_segments, bs, D)
	r_deriv = r_deriv_func(x_flat, x_hat_flat, s)       # (n_segments, bs, D)

	n_seg, _, D = r.shape
	r_in = r.reshape(n_seg * bs, *original_shape[1:])

	score = compute_score(model, r_in, t, k).reshape(n_seg, bs, D)
	integrand = score * r_deriv
	f_flat = torch.trapezoid(integrand, s, dim=0)            # (bs, D)
	f = f_flat.reshape(original_shape)

	return f
