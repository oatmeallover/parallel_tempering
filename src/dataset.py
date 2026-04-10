import torch 
import numpy as np
from scipy.stats import norm

torch.manual_seed(42)
np.random.seed(42)

@torch.no_grad()
def generate_gaussian_mixture(n_samples, means, stds, device='cpu'):
	"""Generates mixture of gaussians according to inputted means and standard deviations"""
	means = torch.as_tensor(means, dtype=torch.float32)
	n_gaussians = len(means)
	
	if isinstance(stds, (int, float)):
		stds = torch.full((n_gaussians,), float(stds))
	else:
		stds = torch.as_tensor(stds, dtype=torch.float32)
		assert len(stds) == n_gaussians, f"stds length {len(stds)} != n_gaussians {n_gaussians}"
		
	component_ids = np.random.choice(n_gaussians, size=n_samples)
	samples = torch.zeros(n_samples, 1, device=device)
	
	for i in range(n_gaussians):
		mask = component_ids == i
		samples[mask] = torch.normal(
			mean=float(means[i]),
			std=float(stds[i]),
			size=(mask.sum(), 1)
		).to(device)
	
	return samples


@torch.no_grad()
def compute_mixture_pdf(dataset_config, x_axis, k=1.0):
	"""Computes analtytical pdf of training dataset from dataset config file, used for plotting"""
	means = np.array(dataset_config['means'])
	stds = dataset_config['stds']
		
	if isinstance(stds, (int, float)):
		stds = np.full(len(means), stds)
	else:
		stds = np.array(stds)
		
	stds = stds / np.sqrt(k)
	
	n_gaussians = len(means)
	pdf = np.zeros_like(x_axis)
	
	for mu, sigma in zip(means, stds):
		pdf += norm.pdf(x_axis, loc=mu, scale=sigma)
	
	pdf /= n_gaussians  
	
	return pdf 
