import torch
import numpy as np
from .sample import ddpm_tsr_swapped

@torch.no_grad()
def mixture_log_likelihood(x, means, stds, weights=None):
    """Score each sample under the Gaussian mixture — higher is better."""
    means = torch.tensor(means, device=x.device, dtype=x.dtype)
    stds = torch.tensor(stds, device=x.device, dtype=x.dtype)
    if weights is None:
        weights = torch.ones(len(means), device=x.device) / len(means)
    else:
        weights = torch.tensor(weights, device=x.device, dtype=x.dtype)

    # x: [N, 1], means: [K] → broadcast to [N, K]
    log_probs = -0.5 * ((x - means) / stds) ** 2 - torch.log(stds) - 0.5 * np.log(2 * np.pi)
    log_mixture = torch.logsumexp(log_probs + torch.log(weights), dim=-1)  # [N]
    return log_mixture

@torch.no_grad()
def best_of_n(model, dataset_config, k, k_ladder, n_samples=8, swap_algorithm=None, replica_swaps=False):
    means = dataset_config["means"]
    stds  = dataset_config["stds"]
    shape = dataset_config["dataset_shape"]

    best_x     = None
    best_score = -float("inf")
    all_scores = []

    for n in range(n_samples):
        torch.manual_seed(n)
        np.random.seed(n)

        x_ladder = ddpm_tsr_swapped(
            model=model,
            dataset_shape=shape,
            k=k,
            k_ladder=k_ladder,
            replica_swaps=replica_swaps,
            swap_algorithm=swap_algorithm,
        )

        # Use k=1.0 replica as candidate
        x_candidate = x_ladder[k]  # [N, 1]

        ll = mixture_log_likelihood(x_candidate, means, stds)
        mean_ll = ll.mean().item()
        all_scores.append(mean_ll)

        if mean_ll > best_score:
            best_score = mean_ll
            best_x = {k_val: x_ladder[k_val].clone() for k_val in k_ladder}

        print(f"  Run {n+1}/{n_samples}  mean_ll={mean_ll:.4f}  best_so_far={best_score:.4f}")

    print(f"\nBest run score: {best_score:.4f}  (across {n_samples} runs)")
    return best_x, best_score, all_scores