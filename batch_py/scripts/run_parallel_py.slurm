#!/bin/bash
#SBATCH --job-name=pt_tsr_100_new
#SBATCH --account=kempner_undergrads
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=0-12:00:00
#SBATCH --mem=32G
#SBATCH --output=output_files/12_hrs_pt_ptsr.out
#SBATCH --error=output_files/12_hrs_pt_ptsr.err
#SBATCH --mail-type=END
#SBATCH --mail-user=zoewu@g.harvard.edu

module purge
module load python
source activate /n/netscratch/kempner_undergrads/Everyone/zwu/environments/diffusers_env

cd ..
srun python generate_samples.py --index_until 5000
