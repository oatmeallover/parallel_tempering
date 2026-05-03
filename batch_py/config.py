from pathlib import Path


LAM_VALUES    = [1.2, 1.15, 1.1, 1.01, 1.05, 0.98]

REAL_DIR    = "/n/netscratch/kempner_undergrads/Everyone/zwu/parallel_toy/images/laion_5k_real"
TSR_DIR     = Path("/n/netscratch/kempner_undergrads/Everyone/zwu/parallel_toy/images/tsr_samples")
PT_TSR_DIR  = Path("/n/netscratch/kempner_undergrads/Everyone/zwu/parallel_toy/images/pt_samples")
PROMPTS_FILE = Path("/n/netscratch/kempner_undergrads/Everyone/zwu/parallel_toy/data_files/laion_5k_prompts.csv")

MODEL_CACHE = "/n/netscratch/kempner_undergrads/Everyone/zwu/parallel_toy/model_checkpoints"
SEED = 42

TSR_SIGMA = 3.0
SWAP_ALGORITHM = {
	"n_replicas": 3,
	"p_ratio": "p",
	"odd_indices":  [2, 6, 10, 14, 18, 21],
	"even_indices": [4, 8, 12, 16, 20, 23],
	"debug": True,
}

N_INF_STEPS = 30

GUIDANCE_SCALE = 7.6