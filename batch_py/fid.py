from PIL import Image, PngImagePlugin, ImageFile
from pathlib import Path
from config import LAM_VALUES, REAL_DIR, TSR_DIR, PT_TSR_DIR, PROMPTS_FILE
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

def build_prompt_map(prompts_file=PROMPTS_FILE, index_until=None):
	df = pd.read_csv(prompts_file)
	if index_until: df = df.iloc[:index_until]
	return {i: row["text"] for i, (_, row) in enumerate(df.iterrows())}

def build_clip_model(device):
	clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
	clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
	clip_model.eval()
	return clip_model, clip_processor


def build_feat_model(device):
	return fid.build_feature_extractor("clean", device=torch.device(device))


# ── Feature extraction ────────────────────────────────────────────────────────

def get_features_subset(folder, model, device, n=None, target_indices = None):
	files = sorted(Path(folder).glob("*.png"))[:n]
	return get_files_features(
		[str(f) for f in files],
		model,
		device=device,
	)

# ── FID ───────────────────────────────────────────────────────────────────────

def compute_real_stats(feat_model, device, real_dir=REAL_DIR, n=None, target_indices=None):
	print("Computing real image features...")
	real_feats = get_features_subset(real_dir, feat_model, device=torch.device(device), n=n, target_indices=target_indices)
	mu_real = np.mean(real_feats, axis=0)
	sigma_real = np.cov(real_feats, rowvar=False)
	print(f"Real features computed from {len(real_feats)} images.\n")
	return mu_real, sigma_real


def compute_fid_score(gen_dir, feat_model, mu_real, sigma_real, device, n=None, target_indices=None):
	gen_feats = get_features_subset(gen_dir, feat_model, device=torch.device(device), n=n, target_indices=target_indices)
	mu_gen = np.mean(gen_feats, axis=0)
	sigma_gen = np.cov(gen_feats, rowvar=False)
	return frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)


# ── CLIP ──────────────────────────────────────────────────────────────────────

def compute_clip(gen_dir, tsr_lam, prompt_map, clip_model, clip_processor, device, index_until=None):
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


# ── Sweep ─────────────────────────────────────────────────────────────────────

def compute_sweep(
	lam_values,
	replica_exchanges,
	device,
	target_indices=None,
	index_until=None,
	tsr_dir = None,
	pt_sr_dir = None,
):

	if pt_sr_dir is not None: PT_TSR_DIR= pt_sr_dir
	if tsr_dir is not None: 
		TSR_DIR= tsr_dir	

	prompt_map                 = build_prompt_map(index_until=index_until)
	clip_model, clip_processor = build_clip_model(device)
	feat_model                 = build_feat_model(device)
	mu_real, sigma_real        = compute_real_stats(feat_model, device, n=index_until)
	
	tsr_results = {alg: {} for alg in replica_exchanges}

	tsr_dirs = []
	for alg in replica_exchanges:
		if alg == False:
			tsr_dirs.append(TSR_DIR)
		elif alg == True:
			tsr_dirs.append(PT_TSR_DIR)

	for tsr_lam in lam_values:
		lam_str = f"lam{tsr_lam:.3f}".replace(".", "p")
		print(f"\n── {lam_str} ──")

		for alg_idx, tsr_samples_dir in enumerate(tsr_dirs):
			alg = replica_exchanges[alg_idx]
			samples_dir = (TSR_DIR if tsr_lam == 1.0 else tsr_samples_dir) / lam_str

			png_files = sorted(samples_dir.glob("*.png"))[:index_until]
			if not png_files:
				print(f"[{alg}]  lam={tsr_lam:.3f}  SKIPPED (no PNGs in {samples_dir})")
				continue

			fid_val  = compute_fid_score(samples_dir, feat_model, mu_real, sigma_real, device, target_indices=target_indices, n=index_until)
			clip_val = compute_clip(samples_dir, tsr_lam, prompt_map, clip_model, clip_processor, device, index_until=index_until)
			tsr_results[alg][tsr_lam] = (fid_val, clip_val)
			print(f"[{alg}]  lam={tsr_lam:.3f}  FID={fid_val:.4f}  CLIP={clip_val:.4f}")

	fig, ax = plt.subplots(figsize=(8, 6))

	for alg in replica_exchanges:
		tsr_lam_vals    = sorted(tsr_results[alg].keys(), reverse=True)
		tsr_clip_vals = [tsr_results[alg][lam][1] for lam in tsr_lam_vals]
		tsr_fid_vals  = [tsr_results[alg][lam][0] for lam in tsr_lam_vals]

		ax.plot(tsr_clip_vals, tsr_fid_vals, marker="o", linewidth=2, label=f"{alg}, CFG=7.5, σ=3.0")
		for lam in tsr_lam_vals:
			f, c = tsr_results[alg][lam]
			ax.annotate(f"lam={lam}", (c, f), textcoords="offset points", xytext=(6, 0), fontsize=8, color="goldenrod")

	ax.set_xlabel("CLIP", fontsize=12)
	ax.set_ylabel("FID", fontsize=12)
	ax.set_title("FID vs CLIP comparison", fontsize=14)
	ax.legend(fontsize=9)
	ax.grid(True, alpha=0.3)
	plt.tight_layout()
	plt.savefig("fid_vs_clip.png", dpi=150)
	plt.show()

	return tsr_results