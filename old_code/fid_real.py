import requests
from pathlib import Path
import pandas as pd
from PIL import Image
from io import BytesIO

real_dir = Path("/n/netscratch/kempner_undergrads/Everyone/zwu/parallel_toy/figures/laion_5k_real")
real_dir.mkdir(exist_ok=True)

df = pd.read_csv("laion_5k_prompts.csv")
failed = 0

for idx, row in df.iterrows():
    img_path = real_dir / f"{idx:05d}.png"
    if img_path.exists():
        continue  # resume if interrupted
    try:
        r = requests.get(row["url"], timeout=5)
        r.raise_for_status()
        # Convert to PNG via PIL to avoid saving corrupt/non-image bytes
        img = Image.open(BytesIO(r.content)).convert("RGB")
        img.save(img_path)
    except Exception as e:
        failed += 1

print(f"Done. {len(df) - failed}/{len(df)} images saved, {failed} failed.")