#!/bin/bash
#SBATCH --job-name=parallel_toy
#SBATCH --account=kempner_undergrads
#SBATCH --partition=kempner_requeue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --time=0-00:05:00
#SBATCH --mem=2G
#SBATCH --output=output_files/output_parallel_trial.out
#SBATCH --error=output_files/output_parallel_trial.err
#SBATCH --mail-type=END
#SBATCH --mail-user=zoewu@g.harvard.edu

module purge
module load python # we are in scripts
source activate /n/netscratch/kempner_undergrads/Everyone/zwu/environments/diffusers_env

cd ..
srun python -m run_diffusers
