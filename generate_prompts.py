import pandas as pd
import requests
from datasets import load_dataset
from tqdm import tqdm
import os
import time

# Load existing CSV
csv_path = "data_files/laion_5k_prompts.csv"
existing_df = pd.read_csv(csv_path)
existing_urls = set(existing_df['url'].tolist())
print(f"Existing entries: {len(existing_df)}")

TARGET = 5000
needed = TARGET - len(existing_df)
print(f"Need to collect: {needed} more entries")

# Load LAION dataset (streaming to avoid downloading everything)
dataset = load_dataset("laion/aesthetics_v2_4.5", split="train", streaming=True)

def check_url(url, timeout=5):
    try:
        response = requests.head(url, timeout=timeout, allow_redirects=True)
        if response.status_code == 200:
            return True
        # Some servers don't support HEAD, try GET with stream
        if response.status_code in [405, 403]:
            response = requests.get(url, timeout=timeout, stream=True)
            return response.status_code == 200
        return False
    except Exception:
        return False

new_rows = []
checked = 0
collected = 0

with tqdm(total=needed, desc="Collecting valid URLs") as pbar:
    for idx, sample in enumerate(dataset):
        url = sample.get('URL', '')
        text = sample.get('TEXT', '')
        
        # Skip if already in CSV
        if url in existing_urls:
            continue
        
        checked += 1
        
        if check_url(url):
            new_rows.append({
                'url': url,
                'text': text,
                'original_idx': idx
            })
            existing_urls.add(url)
            collected += 1
            pbar.update(1)
        
        # Save in batches of 100
        if len(new_rows) % 100 == 0 and len(new_rows) > 0:
            batch_df = pd.DataFrame(new_rows)
            batch_df.to_csv(csv_path, mode='a', header=False, index=False)
            new_rows = []
            print(f"\nSaved batch | Checked: {checked} | Collected: {collected}")
        
        if collected >= needed:
            break
        
        # Small delay to avoid rate limiting
        time.sleep(0.05)

# Save any remaining rows
if new_rows:
    batch_df = pd.DataFrame(new_rows)
    batch_df.to_csv(csv_path, mode='a', header=False, index=False)

# Verify final count
final_df = pd.read_csv(csv_path)
print(f"\nDone! Final CSV has {len(final_df)} entries")