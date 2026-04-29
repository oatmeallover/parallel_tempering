import torch 
import sys
@torch.no_grad()
def compute_tsr_constant(lam, sigma: torch.Tensor, tsr_sigma: float):
	sigma = sigma.float()
	
	# Handle both float and tensor inputs for k
	if not isinstance(lam, torch.Tensor):
		lam = torch.tensor([lam], device=sigma.device, dtype=torch.float32)
	else:
		lam = lam.float()
	
	eta_t = ((1.0 - sigma) ** 2) / (sigma**2 + 1e-12)
	base = eta_t * (tsr_sigma**2)
	tsr = (base + 1.0) / (base * lam.unsqueeze(-1) + 1.0)  # (n_replicas, sigma_dim)
	
	# Replace nan with per-replica lam value
	tsr = torch.where(torch.isnan(tsr), lam.unsqueeze(-1), tsr)
	tsr = torch.min(tsr, lam.unsqueeze(-1))  # clamp max per-replica
	
	return tsr.squeeze().to(sigma.dtype)  # squeeze so scalar k returns scalar-like tensor


@torch.no_grad()
def _lam_ladder(tsr_lam, n_replicas, device, dtype, scale = 0.1):
	return torch.tensor([tsr_lam, tsr_lam+scale, tsr_lam-scale], device=device, dtype=dtype)


@torch.no_grad()
def _replica_view(latents, n_replicas):
	batch_size = latents.shape[0] // int(n_replicas)
	return latents.view(int(n_replicas), batch_size, *latents.shape[1:]), batch_size


@torch.no_grad()
def compute_score(
	model,
	x,
	t,
	alpha_bar_i,
	prompt_embeds,
	pooled_prompt_embeds,
	joint_attention_kwargs=None,
):
	batch_size = x.shape[0]
	timestep = t.expand(batch_size)
	pe = prompt_embeds.repeat_interleave(max(1, batch_size // prompt_embeds.shape[0]), dim=0)
	ppe = pooled_prompt_embeds.repeat_interleave(max(1, batch_size // pooled_prompt_embeds.shape[0]), dim=0)
	eps_hat = model(
		hidden_states=x,
		timestep=timestep,
		encoder_hidden_states=pe,
		pooled_projections=ppe,
		joint_attention_kwargs=joint_attention_kwargs,
		return_dict=False,
	)[0]
	return -eps_hat / torch.sqrt(1.0 - alpha_bar_i) 


@torch.no_grad()
def swap_schedule(latents, i, t, tsr_lam, tsr_sigma, sigma, swap_algorithm, lam_ladder):
	n_replicas = int(swap_algorithm["n_replicas"])

	if i in swap_algorithm["even_indices"]:
		start = 0
	elif i in swap_algorithm["odd_indices"]:
		start = 1
	else:
		return []
	x, _ = _replica_view(latents, n_replicas)
	temp_ladder = compute_tsr_constant(lam_ladder, sigma, tsr_sigma) 
	
	return [((x[0], temp_ladder[0], lam_ladder[0], 0),
		     (x[j], temp_ladder[j], lam_ladder[j], j))
		for i, j in ((i, i + 1) for i in range(start, n_replicas - 1, 2))
	]


@torch.no_grad()
def _segment_path(x, x_hat, n_segments):
	s = torch.linspace(0.0, 1.0, n_segments, device=x.device)
	sv = s.view(-1, 1, 1)
	diff = (x_hat - x).unsqueeze(0)
	return s, x + sv * diff, diff.expand(s.shape[0], -1, -1)


@torch.no_grad()
def compute_score_integral(
	model,
	target,
	source,
	t,
	swap_algorithm,
	alpha_bar_i,
	prompt_embeds,
	pooled_prompt_embeds,
	joint_attention_kwargs=None,
	n_segments=3,
):
	x_t, temp_t, lam_t, i_t = target
	x_s, temp_s, lam_s, i_s = source

	orig_shape = x_s.shape
	bs = orig_shape[0]
	x_flat = x_s.reshape(bs, -1)
	x_hat_flat = x_t.reshape(bs, -1)
	s, r, r_deriv = _segment_path(x_flat, x_hat_flat, n_segments)
	n_seg, _, d = r.shape
	r_in = r.reshape(n_seg * bs, *orig_shape[1:]).to(next(model.parameters()).dtype)
	score = compute_score(
		model,
		r_in,
		t,
		alpha_bar_i,
		prompt_embeds,
		pooled_prompt_embeds,
		joint_attention_kwargs=joint_attention_kwargs,
	).reshape(n_seg, bs, d)
	f = -torch.trapezoid(score * r_deriv, s, dim=0).reshape(orig_shape)
	return f * (temp_s - temp_t)


@torch.no_grad()
def compute_correction(
	model,
	target,
	source,
	t,
	swap_algorithm,
	alpha_bar_i,
	prompt_embeds,
	pooled_prompt_embeds,
	joint_attention_kwargs=None,
):
	f = compute_score_integral(
		model,
		target,
		source,
		t,
		swap_algorithm,
		alpha_bar_i,
		prompt_embeds,
		pooled_prompt_embeds,
		joint_attention_kwargs=joint_attention_kwargs,
	)
	a = torch.clamp(torch.exp(f), max=1.0)
	return (torch.rand_like(a) < a).float()


@torch.no_grad()
def exchanged_replicas(
	model,
	latents,
	i,
	t,
	tsr_lam,
	tsr_sigma,
	sigma,
	swap_algorithm,
	alpha_bar_i,
	prompt_embeds,
	pooled_prompt_embeds,
	lam_ladder,
	joint_attention_kwargs=None,
):

	n_replicas = int(swap_algorithm["n_replicas"])
	x, _ = _replica_view(latents, n_replicas)

	for target, source in swap_schedule(latents, i, t, tsr_lam, tsr_sigma, sigma, swap_algorithm, lam_ladder):

		x_t, temp_t, lam_t, i_t = target
		x_s, temp_s, lam_s, i_s = source

		accept = compute_correction(
			model,
			target,
			source,
			t,
			swap_algorithm,
			alpha_bar_i,
			prompt_embeds,
			pooled_prompt_embeds,
			joint_attention_kwargs=joint_attention_kwargs,
		)

		if swap_algorithm["debug"]:
			rate = accept.mean().item()
			print(f"Time {t:.2f} swap btwn source {float(lam_s):.2f} and target {float(lam_t):.2f} accept {rate:.3f} std {x.std().item():.3f}")
				
		mask = accept.bool()
		x[i_t] = torch.where(mask, x_s, x_t)
		x[i_s] = torch.where(mask, x_t, x_s)

	return x.reshape_as(latents)