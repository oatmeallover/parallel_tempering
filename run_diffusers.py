
import sys
sys.path.insert(0, "/n/netscratch/kempner_undergrads/Everyone/zwu/parallel_toy/diffusers/src")
from diffusers.replica_exchange.acceptance import _k_ladder

import torch
from diffusers import StableDiffusion3Pipeline
import numpy as np


pipe = StableDiffusion3Pipeline.from_pretrained(
	"stabilityai/stable-diffusion-3-medium-diffusers",
    torch_dtype=torch.float16,
	cache_dir="/n/netscratch/kempner_undergrads/Everyone/zwu/parallel_toy/model_checkpoints"
)
pipe = pipe.to("cuda")

swap_algorithm={
		"n_replicas": 3, 
		"p_ratio": "p",
		"even_indices": [7, 12, 16, 19, 21],   # t ≈ 870, 763, 648, 536
		"odd_indices":  [8, 13, 17, 20, 22],
		"debug": True
	}

tsr_sigma = 3.0
replica_exchange = True

# with swaps 1.2

tsr_k = 1.2
generator = torch.Generator(device="cuda").manual_seed(42)

all_images = pipe(
	"basket of flowers and a basket with Easter eggs",
	negative_prompt="",
	num_inference_steps=30,
	guidance_scale=5.0,
	tsr_k=tsr_k,
	tsr_sigma=tsr_sigma,
	replica_exchange=replica_exchange,
	swap_algorithm=swap_algorithm,
	generator=generator,
).images

for idx in range(len(all_images)):
	image = all_images[idx]

	arr = np.array(image)
	print(f"Image {idx}: min={arr.min()}, max={arr.max()}, mean={arr.mean():.2f}")
	
	if replica_exchange:
		k_ladder = _k_ladder(tsr_k, swap_algorithm["n_replicas"], device=tsr_k.device, dtype=tsr_k.dtype)
		k_val = k_ladder[idx]
		string = f"pt_tsr_{tsr_k}_val_{k_val}"
	else:
		string = f"tsr_{tsr_k}"

	output_path = f"/n/netscratch/kempner_undergrads/Everyone/zwu/parallel_toy/figures/basket_{string}.png"
	image.save(output_path)
	print(f"Saved to {output_path}")

# With swaps 0.8
tsr_k = 0.8
generator = torch.Generator(device="cuda").manual_seed(42)

all_images = pipe(
	"basket of flowers and a basket with Easter eggs",
	negative_prompt="",
	num_inference_steps=30,
	guidance_scale=5.0,
	tsr_k=tsr_k,
	tsr_sigma=tsr_sigma,
	replica_exchange=replica_exchange,
	swap_algorithm=swap_algorithm,
	generator=generator,
).images

for idx in range(len(all_images)):
	image = all_images[idx]

	arr = np.array(image)
	print(f"Image {idx}: min={arr.min()}, max={arr.max()}, mean={arr.mean():.2f}")
	
	if replica_exchange:
		k_ladder = np.linspace(1.0/float(tsr_k), float(tsr_k), int(swap_algorithm["n_replicas"]))
		k_val = k_ladder[idx]
		string = f"pt_tsr_{tsr_k}_val_{k_val:3f}"
	else:
		string = f"tsr_{tsr_k:3f}"

	output_path = f"/n/netscratch/kempner_undergrads/Everyone/zwu/parallel_toy/figures/basket_{string}.png"
	image.save(output_path)
	print(f"Saved to {output_path}")
