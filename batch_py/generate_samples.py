import torch
import pandas as pd
from tqdm import tqdm
from config import LAM_VALUES, TSR_DIR, PT_TSR_DIR, PROMPTS_FILE, MODEL_CACHE, SEED, TSR_SIGMA, SWAP_ALGORITHM, N_INF_STEPS, GUIDANCE_SCALE
import gc

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--replica_exchange", action="store_true", default=False)
parser.add_argument("--lam_values", type=float, nargs="+", default=None)
parser.add_argument("--index_until", type=int, default=None)

args = parser.parse_args()

REPLICA_EXCHANGE = args.replica_exchange
LAM_VALUES = args.lam_values if args.lam_values is not None else LAM_VALUES
INDEX_UNTIL = args.index_until 

# ── Load prompts once ─────────────────────────────────────────────────────────
df = pd.read_csv(PROMPTS_FILE)
prompts = df["text"].tolist()[:INDEX_UNTIL]
print(f"Loaded {len(prompts)} prompts")

gc.collect()
torch.cuda.empty_cache()

from diffusers import StableDiffusion3Pipeline

# ── Load model once ───────────────────────────────────────────────────────────
pipe = StableDiffusion3Pipeline.from_pretrained(
	"stabilityai/stable-diffusion-3-medium-diffusers",
	torch_dtype=torch.float16,
	cache_dir=MODEL_CACHE,
)
pipe = pipe.to("cuda")
pipe.set_progress_bar_config(disable=True)


if REPLICA_EXCHANGE:
	BASE_OUTPUT_DIR = PT_TSR_DIR
else:
	BASE_OUTPUT_DIR = TSR_DIR

# ── Sweep ─────────────────────────────────────────────────────────────────────
for tsr_lam in LAM_VALUES:

	k_str = f"lam{tsr_lam:.3f}".replace(".", "p")   # e.g. "k0p950" — safe for filenames
	output_dir = BASE_OUTPUT_DIR / k_str
	output_dir.mkdir(parents=True, exist_ok=True)

	for idx, prompt in enumerate(tqdm(prompts, desc=f"l={tsr_lam}")):

		if (output_dir / f"{idx:05d}.png").exists():
			continue

		generator = torch.Generator(device="cuda").manual_seed(SEED + idx)

		images = pipe(
			prompt,
			negative_prompt="",
			num_inference_steps=N_INF_STEPS,
			guidance_scale=GUIDANCE_SCALE,
			tsr_lam=tsr_lam,
			tsr_sigma=TSR_SIGMA,
			replica_exchange=REPLICA_EXCHANGE,
			swap_algorithm=SWAP_ALGORITHM,
			generator=generator,
		).images

		out_path = output_dir / f"{idx:05d}.png"
		images[0].save(out_path, icc_profile=None)
		
		del images
		torch.cuda.empty_cache()

print("\n All k values complete.")