import torch
from diffusers import StableDiffusion3Pipeline
import numpy as np

pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    torch_dtype=torch.float16,
    cache_dir="/n/netscratch/kempner_undergrads/Everyone/zwu/parallel_toy/model_checkpoints"
)
pipe = pipe.to("cuda")


tsr_k = 0.75
tsr_sigma = 1.0
replica_exchange = True
swap_algorithm={
        "n_replicas": 3,
        "p_ratio": "p",
        "even_indices" : [7, 14, 19, 21],  # even steps
		"odd_indices" : [8, 15, 20, 22],
		"debug": True
    }

generator = torch.Generator(device="cuda").manual_seed(42)

all_images = pipe(
    "hyperrealism chiaroscuro cinematic oil on canvas matte painting of professional golden hour flambient real estate photo scifi traditional japanese onsen shinto temple zen garden",
    negative_prompt="",
    num_inference_steps=28,
    guidance_scale=7.0,
    tsr_k=tsr_k,
    tsr_sigma=tsr_sigma,
    replica_exchange=replica_exchange,
    swap_algorithm=swap_algorithm,
    generator=generator,
).images

for idx in range(len(all_images)):

	image = all_images[idx]
	if replica_exchange:
		k_ladder = np.linspace(float(tsr_k), 1.0, int(swap_algorithm["n_replicas"]))
		k_val = k_ladder[idx]
		string = f"pt_tsr_{tsr_k}_val_{k_val}"
	else:
		string = f"tsr_{tsr_k}"

	output_path = f"/n/netscratch/kempner_undergrads/Everyone/zwu/parallel_toy/figures/japanese_{string}.png"
	image.save(output_path)
	print(f"Saved to {output_path}")