from pathlib import Path


INDEX_UNTIL = 610

K_VALUES    = [0.93, 0.98, 0.95, 1.05, 0.9, 1.00]

REAL_DIR    = "/n/netscratch/kempner_undergrads/Everyone/zwu/parallel_toy/images/laion_5k_real"
TSR_DIR     = Path("/n/netscratch/kempner_undergrads/Everyone/zwu/parallel_toy/images/laion_5k_generated")
PT_TSR_DIR  = Path("/n/netscratch/kempner_undergrads/Everyone/zwu/parallel_toy/images/laion_5k_generated_pt")
PROMPTS_FILE = Path("/n/netscratch/kempner_undergrads/Everyone/zwu/parallel_toy/data_files/laion_5k_prompts.csv")

MODEL_CACHE = "/n/netscratch/kempner_undergrads/Everyone/zwu/parallel_toy/model_checkpoints"
SEED = 42

TSR_SIGMA = 3.0
SWAP_ALGORITHM = {
	"n_replicas": 3,
	"p_ratio": "p",
	"even_indices": [0, 7, 12, 16, 19, 21],   # t ≈ 870, 763, 648, 536
	"odd_indices":  [1, 8, 13, 17, 20, 22],
	"debug": False,
}

N_INF_STEPS = 30

GUIDANCE_SCALE = 7.5