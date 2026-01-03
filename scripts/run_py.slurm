#!/bin/bash
#SBATCH --job-name=parllel_toy
#SBATCH --account=kempner_undergrads
#SBATCH --partition=kempner
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=0-12:00
#SBATCH --mem=256G
#SBATCH --output=output_files/output_parallel_toy.out
#SBATCH --error=output_files/output_parallel_toy.err
#SBATCH --mail-type=END
#SBATCH --mail-user=zoewu@g.harvard.edu

module purge
module load python
source activate parallel_toy

srun python parallel_toy_cfg.py
