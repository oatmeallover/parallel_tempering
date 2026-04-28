import torch 
import sys
@torch.no_grad()
def compute_tsr_constant(k, sigma: torch.Tensor, tsr_sigma: float):
    sigma = sigma.float()
    
    # Handle both float and tensor inputs for k
    if not isinstance(k, torch.Tensor):
        k = torch.tensor([k], device=sigma.device, dtype=torch.float32)
    else:
        k = k.float()
    
    eta_t = ((1.0 - sigma) ** 2) / (sigma**2 + 1e-12)
    base = eta_t * (tsr_sigma**2)
    tsr = (base + 1.0) / (base / k.unsqueeze(-1) + 1.0)  # (n_replicas, sigma_dim)
    
    # Replace nan with per-replica k value
    tsr = torch.where(torch.isnan(tsr), k.unsqueeze(-1), tsr)
    tsr = torch.min(tsr, k.unsqueeze(-1))  # clamp max per-replica
    
    return tsr.squeeze().to(sigma.dtype)  # squeeze so scalar k returns scalar-like tensor


@torch.no_grad()
def _k_ladder(tsr_k, n_replicas, device, dtype):
    k = float(tsr_k)
    half = int(n_replicas // 2)
    bottom = torch.linspace(1.0/k, 1.0, half + 1, device=device, dtype=dtype)
    top = torch.linspace(1.0, k, half + 1, device=device, dtype=dtype)
    ladder = torch.cat([bottom, top[1:]])
    return ladder.flip(0)  # [k, ..., 1.0, ..., 1/k]


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
def swap_schedule(latents, i, t, tsr_k, tsr_sigma, sigma, swap_algorithm):
	n_replicas = int(swap_algorithm["n_replicas"])

	if swap_algorithm["debug"]:
		print(f"t is {t}")

	if i in swap_algorithm["even_indices"]:
		start = 0
	elif i in swap_algorithm["odd_indices"]:
		start = 1
	else:
		return []
	x, _ = _replica_view(latents, n_replicas)
	k_ladder = _k_ladder(tsr_k, n_replicas, latents.device, latents.dtype)
	temp_ladder = compute_tsr_constant(k_ladder, sigma, tsr_sigma) 
	
	return [((x[i].clone(), temp_ladder[i], k_ladder[i], i),
		  (x[j].clone(), temp_ladder[j], k_ladder[j], j))
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
	n_segments=8,
):
	x_t, temp_t, k_t, i_t = target
	x_s, temp_s, k_s, i_s = source

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
	
	p_ratio = swap_algorithm["p_ratio"]

	if p_ratio == "s":
		return f * temp_s
	if p_ratio == "t":
		return -f * temp_t
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
	tsr_k,
	tsr_sigma,
	sigma,
	swap_algorithm,
	alpha_bar_i,
	prompt_embeds,
	pooled_prompt_embeds,
	joint_attention_kwargs=None,
):

	n_replicas = int(swap_algorithm["n_replicas"])
	x, _ = _replica_view(latents, n_replicas)

	for target, source in swap_schedule(latents, i, t, tsr_k, tsr_sigma, sigma, swap_algorithm):

		x_t, temp_t, k_t, i_t = target
		x_s, temp_s, k_s, i_s = source

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
			print(f"Time {t} swap btwn source {float(k_s):.2f} and target {float(k_t):.2f} accept {rate:.3f} std {x.std().item():.3f}")
				
		mask = accept.bool()
		x[i_t] = torch.where(mask, x_s, x_t)
		x[i_s] = torch.where(mask, x_t, x_s)

	return x.reshape_as(latents)