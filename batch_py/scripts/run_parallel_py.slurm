#!/bin/bash
#SBATCH --job-name=pt_ind_swaps
#SBATCH --account=kempner_undergrads
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=0-01:30:00
#SBATCH --mem=32G
#SBATCH --output=output_files/half_hr_pt_ptsr.out
#SBATCH --error=output_files/half_hr_pt_ptsr.err
#SBATCH --mail-type=END
#SBATCH --mail-user=zoewu@g.harvard.edu

module purge
module load python
source activate /n/netscratch/kempner_undergrads/Everyone/zwu/environments/diffusers_env

cd ..
srun python generate_samples.py --index_until 1000
