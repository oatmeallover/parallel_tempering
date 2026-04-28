import torch
import gc

gc.collect()
torch.cuda.empty_cache()

import sys
sys.path.insert(0, "/n/netscratch/kempner_undergrads/Everyone/zwu/parallel_toy/diffusers/src")
from diffusers.replica_exchange.acceptance import _k_ladder

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
		"even_indices": [0, 7, 12, 16, 19, 21],   # t ≈ 870, 763, 648, 536
		"odd_indices":  [1, 8, 13, 17, 20, 22],
		"debug": True
	}

tsr_sigma = 3.0
labels = ["chocolates", "badminton", "princess", "buildings", "canoe"]
prompts = ["a box of chocolates", "Boy playing badminton with his grandfather", "The Princess and the Frog Read-Along W/CD [With Paperback Book]", "London from the Sky Garden Photographic Print", "Canoe on Elk Lake"]


tsr_k_vals = [1.4, 0.6]
replica_exchanges = [True, False]

for i, prompt in enumerate(prompts):

	for tsr_k in tsr_k_vals:
		for replica_exchange in replica_exchanges:

			generator = torch.Generator(device="cuda").manual_seed(42)

			all_images = pipe(
				prompt,
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
					k_ladder = _k_ladder(torch.tensor(tsr_k), swap_algorithm["n_replicas"], device="cpu", dtype=torch.float32)				
					k_val = k_ladder[idx]
					string = f"pt_tsr_{tsr_k}_val_{k_val:.2f}"
				else:
					string = f"tsr_{tsr_k}"

				output_path = f"/n/netscratch/kempner_undergrads/Everyone/zwu/parallel_toy/figures/{labels[i]}_{string}.png"
				image.save(output_path)
				print(f"Saved to {output_path}")