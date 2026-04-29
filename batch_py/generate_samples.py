import torch
import pandas as pd
import json
from tqdm import tqdm
from config import INDEX_UNTIL, LAM_VALUES, REAL_DIR, TSR_DIR, PT_TSR_DIR, PROMPTS_FILE, MODEL_CACHE, SEED, TSR_SIGMA, SWAP_ALGORITHM, N_INF_STEPS, GUIDANCE_SCALE
import gc
from fid import (
    build_prompt_map, build_clip_model, build_feat_model,
    compute_real_stats, compute_fid_score, compute_clip, sanity_check
)
import matplotlib.pyplot as plt
from pathlib import Path

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--replica_exchange", action="store_true", default=False)
parser.add_argument("--lam_values", type=float, nargs="+", default=None)
parser.add_argument("--index_until", type=int, default=None)

args = parser.parse_args()

REPLICA_EXCHANGE = args.replica_exchange
LAM_VALUES = args.lam_values if args.lam_values is not None else LAM_VALUES
INDEX_UNTIL = args.index_until if args.index_until is not None else INDEX_UNTIL

LAM_FULL = LAM_VALUES + [1.0 + (1.0-l) for l in LAM_VALUES]

SWAP_ALGORITHM = {
	"n_replicas": 4,
	"p_ratio": "p",
	"even_indices": [2,  8, 12,  16],   # t ≈ 870, 763, 648, 536
	"odd_indices":  [4,  10, 14,  18],
	"debug": False,
}

# ── Load prompts once ─────────────────────────────────────────────────────────
df = pd.read_csv(PROMPTS_FILE, dtype={"original_idx": str})
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
	LAM_ALL = LAM_VALUES
else:
	BASE_OUTPUT_DIR = TSR_DIR
	LAM_ALL = LAM_FULL

# ── Sweep ─────────────────────────────────────────────────────────────────────
for tsr_lam in LAM_ALL:
	k_str = f"lam{tsr_lam:.3f}".replace(".", "p")   # e.g. "k0p950" — safe for filenames
	output_dir = BASE_OUTPUT_DIR / k_str
	checkpoint_file = output_dir / "completed.json"
	output_dir.mkdir(parents=True, exist_ok=True)

	if REPLICA_EXCHANGE:
		k_str_flipped = f"lam{2.0-tsr_lam:.3f}".replace(".", "p")   # e.g. "k0p950" — safe for filenames
		output_dir_flipped = BASE_OUTPUT_DIR / k_str_flipped
		checkpoint_file_flipped = output_dir_flipped / "completed.json"
		output_dir_flipped.mkdir(parents=True, exist_ok=True)

	existing = list(output_dir.glob("*.png"))
	completed = set(range(len(existing)))
	if existing:
		print(f"\n[k={tsr_lam}] Resuming — {len(completed)}/{len(prompts)} done")
	else:
		completed = set()
		print(f"\n[k={tsr_lam}] Starting fresh")

	for idx, prompt in enumerate(tqdm(prompts, desc=f"k={tsr_lam}")):
		if idx in completed:
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

		out_path = output_dir / f"{df.iloc[idx]['original_idx']}.png"
		images[1].save(out_path, icc_profile=None)

		if REPLICA_EXCHANGE:
			out_path = output_dir_flipped / f"{df.iloc[idx]['original_idx']}.png"
			images[2].save(out_path, icc_profile=None)
		
		del images
		torch.cuda.empty_cache()

		completed.add(idx)
		if idx % 50 == 0:
			checkpoint_file.write_text(json.dumps(list(completed)))


	checkpoint_file.write_text(json.dumps(list(completed)))
	print(f"[k={tsr_lam}] Done — {len(completed)} images saved to {output_dir}")

print("\n All k values complete.")