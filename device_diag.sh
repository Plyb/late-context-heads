# NOTE: BYU ORC only — uses SBATCH/SLURM job submission. Not needed for general use.
#!/bin/bash --login
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=p100:2
#SBATCH --mem-per-cpu=60G
#SBATCH --output=slurm_logs/%j.out

nvidia-smi -L
srun uv run python tests/device_diag.py
