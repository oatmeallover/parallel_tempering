#!/bin/bash
#SBATCH --job-name=parallel_toy
#SBATCH --account=kempner_undergrads
#SBATCH --partition=kempner
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=0-10:00:00
#SBATCH --mem=64G
#SBATCH --output=output_files/output_parallel_toy.out
#SBATCH --error=output_files/output_parallel_toy.err
#SBATCH --mail-type=END
#SBATCH --mail-user=zoewu@g.harvard.edu

module purge
module load python # we are in scripts
source activate /n/netscratch/kempner_undergrads/Everyone/zwu/conda_envs/parallel_toy

cd ..
srun python -m src.eval
