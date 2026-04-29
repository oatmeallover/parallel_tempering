import torch
import numpy as np
import pandas as pd
from diffusers import StableDiffusion3Pipeline
from pathlib import Path
import json
from tqdm import tqdm

from config import INDEX_UNTIL, K_VALUES, REAL_DIR, TSR_DIR, PT_TSR_DIR, PROMPTS_FILE, MODEL_CACHE, SEED, TSR_SIGMA, SWAP_ALGORITHM, N_INF_STEPS, GUIDANCE_SCALE

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--replica_exchange", action="store_true", default=False)
parser.add_argument("--k_values", type=float, nargs="+", default=None)
parser.add_argument("--index_until", type=int, default=None)

args = parser.parse_args()

REPLICA_EXCHANGE = args.replica_exchange
K_VALUES = args.k_values if args.k_values is not None else K_VALUES
INDEX_UNTIL = args.index_until if args.index_until is not None else INDEX_UNTIL

if REPLICA_EXCHANGE:
	BASE_OUTPUT_DIR = PT_TSR_DIR
else:
	BASE_OUTPUT_DIR = TSR_DIR

# ── Load prompts once ─────────────────────────────────────────────────────────
df = pd.read_csv(PROMPTS_FILE, dtype={"original_idx": str})
prompts = df["text"].tolist()[:INDEX_UNTIL]
print(f"Loaded {len(prompts)} prompts")

# ── Load model once ───────────────────────────────────────────────────────────
pipe = StableDiffusion3Pipeline.from_pretrained(
	"stabilityai/stable-diffusion-3-medium-diffusers",
	torch_dtype=torch.float16,
	cache_dir=MODEL_CACHE,
)
pipe = pipe.to("cuda")
pipe.transformer = torch.compile(pipe.transformer, mode="reduce-overhead") 
pipe.set_progress_bar_config(disable=True)

# ── Sweep ─────────────────────────────────────────────────────────────────────
for tsr_k in K_VALUES:
	k_str = f"k{tsr_k:.3f}".replace(".", "p")   # e.g. "k0p950" — safe for filenames
	output_dir = BASE_OUTPUT_DIR / k_str
	checkpoint_file = output_dir / "completed.json"
	output_dir.mkdir(parents=True, exist_ok=True)

	existing = list(output_dir.glob("*.png"))
	completed = set(range(len(existing)))
	if existing:
		print(f"\n[k={tsr_k}] Resuming — {len(completed)}/{len(prompts)} done")
	else:
		completed = set()
		print(f"\n[k={tsr_k}] Starting fresh")

	for idx, prompt in enumerate(tqdm(prompts, desc=f"k={tsr_k}")):
		if idx in completed:
			continue

		generator = torch.Generator(device="cuda").manual_seed(SEED + idx)

		try:
			images = pipe(
				prompt,
				negative_prompt="",
				num_inference_steps=N_INF_STEPS,
				guidance_scale=GUIDANCE_SCALE,
				tsr_k=tsr_k,
				tsr_sigma=TSR_SIGMA,
				replica_exchange=REPLICA_EXCHANGE,
				swap_algorithm=SWAP_ALGORITHM,
				generator=generator,
			).images

			out_path = output_dir / f"{df.iloc[idx]['original_idx']}.png"
			images[0].save(out_path, icc_profile=None)
			
			completed.add(idx)
			if idx % 50 == 0:
				checkpoint_file.write_text(json.dumps(list(completed)))

		except Exception as e:
			print(f"[ERROR] k={tsr_k} idx={idx}: {e}")
			continue

	checkpoint_file.write_text(json.dumps(list(completed)))
	print(f"[k={tsr_k}] Done — {len(completed)} images saved to {output_dir}")

print("\n All k values complete.")