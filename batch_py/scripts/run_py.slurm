#!/bin/bash
#SBATCH --job-name=tsr_610
#SBATCH --account=kempner_undergrads
#SBATCH --partition=kempner
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --time=0-01:00:00
#SBATCH --mem=24G
#SBATCH --output=output_files/pt_tsr_sample_generation.out
#SBATCH --error=output_files/pt_tsr_sample_generation.err
#SBATCH --mail-type=END
#SBATCH --mail-user=zoewu@g.harvard.edu

module purge
module load python # we are in scripts
source activate /n/netscratch/kempner_undergrads/Everyone/zwu/environments/diffusers_env

cd ..
cd ..
srun python generate_prompts.py
