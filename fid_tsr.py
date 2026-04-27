import torch
import numpy as np
import pandas as pd
from diffusers import StableDiffusion3Pipeline
from pathlib import Path
import json
from tqdm import tqdm

# ── Config ──────────────────────────────────────────────────────────────────
BASE_OUTPUT_DIR = Path("/n/netscratch/kempner_undergrads/Everyone/zwu/parallel_toy/figures/laion_5k_generated")
PROMPTS_FILE = "laion_5k_prompts.csv"
MODEL_CACHE = "/n/netscratch/kempner_undergrads/Everyone/zwu/parallel_toy/model_checkpoints"
SEED = 42

# ── K sweep ──────────────────────────────────────────────────────────────────
K_VALUES = [1.05, 1.0, 0.99, 0.98, 0.95, 0.93, 0.9]

# ── Shared pipeline config ───────────────────────────────────────────────────
TSR_SIGMA = 3.0
REPLICA_EXCHANGE = False
SWAP_ALGORITHM = {
    "n_replicas": 3,
    "p_ratio": "p",
    "even_indices": [9, 14, 19, 21],  # even steps
	"odd_indices" : [10, 15, 20, 22],
    "debug": False,
}

# ── Load prompts once ─────────────────────────────────────────────────────────
df = pd.read_csv(PROMPTS_FILE)
prompts = df["TEXT"].tolist()[:5000]
print(f"Loaded {len(prompts)} prompts")

# ── Load model once ───────────────────────────────────────────────────────────
pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    torch_dtype=torch.float16,
    cache_dir=MODEL_CACHE,
)
pipe = pipe.to("cuda")
pipe.set_progress_bar_config(disable=True)

# ── Sweep ─────────────────────────────────────────────────────────────────────
for tsr_k in K_VALUES:
    k_str = f"k{tsr_k:.3f}".replace(".", "p")   # e.g. "k0p950" — safe for filenames
    output_dir = BASE_OUTPUT_DIR / k_str
    checkpoint_file = output_dir / "completed.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resume support per k value
    if checkpoint_file.exists():
        completed = set(json.loads(checkpoint_file.read_text()))
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
                num_inference_steps=30,
                guidance_scale=5.0,
                tsr_k=tsr_k,
                tsr_sigma=TSR_SIGMA,
                replica_exchange=REPLICA_EXCHANGE,
                swap_algorithm=SWAP_ALGORITHM,
                generator=generator,
            ).images

            out_path = output_dir / f"{idx:05d}.png"
            images[0].save(out_path)

            completed.add(idx)
            if idx % 50 == 0:
                checkpoint_file.write_text(json.dumps(list(completed)))

        except Exception as e:
            print(f"[ERROR] k={tsr_k} idx={idx}: {e}")
            continue

    checkpoint_file.write_text(json.dumps(list(completed)))
    print(f"[k={tsr_k}] Done — {len(completed)} images saved to {output_dir}")

print("\n All k values complete.")