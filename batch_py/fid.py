from PIL import Image, PngImagePlugin, ImageFile
from pathlib import Path
from config import INDEX_UNTIL, LAM_VALUES, REAL_DIR, TSR_DIR, PT_TSR_DIR, PROMPTS_FILE
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor
import cleanfid.fid as fid
from cleanfid.fid import get_files_features, frechet_distance

ImageFile.LOAD_TRUNCATED_IMAGES = True


# ── Setup ─────────────────────────────────────────────────────────────────────

def build_prompt_map(prompts_file=PROMPTS_FILE, index_until=INDEX_UNTIL):
    df = pd.read_csv(prompts_file, dtype={"original_idx": str})
    df = df.iloc[:index_until]
    return {int(row["original_idx"]): row["text"] for _, row in df.iterrows()}


def build_clip_model(device):
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.eval()
    return clip_model, clip_processor


def build_feat_model(device):
    return fid.build_feature_extractor("clean", device=torch.device(device))


# ── Feature extraction ────────────────────────────────────────────────────────

def get_features_subset(folder, model, device, n=INDEX_UNTIL):
    files = sorted(Path(folder).glob("*.png"))[:n]
    return get_files_features(
        [str(f) for f in files],
        model,
        device=device,
    )


# ── FID ───────────────────────────────────────────────────────────────────────

def compute_real_stats(feat_model, device, real_dir=REAL_DIR, n=INDEX_UNTIL):
    print("Computing real image features...")
    real_feats = get_features_subset(real_dir, feat_model, device=torch.device(device), n=n)
    mu_real = np.mean(real_feats, axis=0)
    sigma_real = np.cov(real_feats, rowvar=False)
    print(f"Real features computed from {len(real_feats)} images.\n")
    return mu_real, sigma_real


def compute_fid_score(gen_dir, feat_model, mu_real, sigma_real, device, n=INDEX_UNTIL):
    gen_feats = get_features_subset(gen_dir, feat_model, device=torch.device(device), n=n)
    mu_gen = np.mean(gen_feats, axis=0)
    sigma_gen = np.cov(gen_feats, rowvar=False)
    return frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)


# ── CLIP ──────────────────────────────────────────────────────────────────────

def compute_clip(gen_dir, tsr_lam, prompt_map, clip_model, clip_processor, device, index_until=INDEX_UNTIL):
    scores = []
    for img_path in tqdm(sorted(gen_dir.glob("*.png"))[:index_until], desc=f"CLIP lam={tsr_lam}"):
        idx = int(img_path.stem)
        if idx not in prompt_map:
            continue
        prompt = prompt_map[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"  Warning, skipping {img_path.name}: {e}")
            continue
        inputs = clip_processor(
            text=[prompt], images=image,
            return_tensors="pt", padding=True,
            truncation=True, max_length=75
        ).to(device)
        with torch.no_grad():
            score = clip_model(**inputs).logits_per_image[0, 0].item()
        scores.append(score)
    return np.mean(scores)


# ── Sanity check ──────────────────────────────────────────────────────────────

def sanity_check(gen_dir, prompt_map, idx=1000):
    img_path = gen_dir / f"{idx:05d}.png"
    prompt = prompt_map.get(idx, "NOT FOUND")
    print(f"idx={idx}  prompt: {prompt}")
    return Image.open(img_path).convert("RGB")


# ── Main: compute all metrics for a directory ─────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_dir", type=str, required=True)
    parser.add_argument("--tsr_lam", type=float, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    gen_dir = Path(args.gen_dir)
    device = args.device

    prompt_map = build_prompt_map()
    clip_model, clip_processor = build_clip_model(device)
    feat_model = build_feat_model(device)
    mu_real, sigma_real = compute_real_stats(feat_model, device)

    fid_score = compute_fid_score(gen_dir, feat_model, mu_real, sigma_real, device)
    clip_score = compute_clip(gen_dir, args.tsr_lam, prompt_map, clip_model, clip_processor, device)

    print(f"\nFID:  {fid_score:.4f}")
    print(f"CLIP: {clip_score:.4f}")