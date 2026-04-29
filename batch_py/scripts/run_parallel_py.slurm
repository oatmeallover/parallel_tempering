#!/bin/bash
#SBATCH --job-name=pt_tsr_100_new
#SBATCH --account=kempner_undergrads
#SBATCH --partition=kempner
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=0-01:00:00
#SBATCH --mem=24G
#SBATCH --output=output_files/output_pt_tsr_new.out
#SBATCH --error=output_files/output_pt_tsr_new.err
#SBATCH --mail-type=END
#SBATCH --mail-user=zoewu@g.harvard.edu

module purge
module load python
source activate /n/netscratch/kempner_undergrads/Everyone/zwu/environments/diffusers_env

cd ..
srun python generate_samples.py --replica_exchange --k_values 0.90 0.93 0.95 0.98 1.05 --index_until 50
srun python generate_samples.py --k_values 0.90 0.93 0.95 0.98 1.05 1.00 --index_until 50
srun python generate_samples.py --replica_exchange --k_values 0.90 0.93 0.95 0.98 1.05 --index_until 100
srun python generate_samples.py --k_values 0.90 0.93 0.95 0.98 1.05 1.00 --index_until 100


# srun python generate_samples.py --replica_exchange --k_values 0.90 0.93 0.95 0.98 1.05 --index_until 1000
# srun python generate_samples.py --replica_exchange --k_values 0.90 0.93 0.95 0.98 1.05  --index_until 1500
# srun python generate_samples.py --replica_exchange --k_values 0.85 --index_until 1500
