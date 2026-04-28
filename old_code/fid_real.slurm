#!/bin/bash
#SBATCH --job-name=fid_real
#SBATCH --account=kempner_undergrads
#SBATCH --partition=kempner
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --time=0-01:30:00
#SBATCH --mem=16G
#SBATCH --output=output_files/output_parallel_real.out
#SBATCH --error=output_files/output_parallel_real.err
#SBATCH --mail-type=END
#SBATCH --mail-user=zoewu@g.harvard.edu

module purge
module load python # we are in scripts
source activate /n/netscratch/kempner_undergrads/Everyone/zwu/environments/diffusers_env

cd ..
srun python -m fid_real