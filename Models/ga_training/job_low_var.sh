#!/bin/bash -l
#SBATCH --job-name=tc-train-low-var
#SBATCH --account=nano
#SBATCH --partition=etna_gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:V100:1
#SBATCH --time=9:00:00
#SBATCH --output=train-%j.out

module load ml/pytorch

cd "$SLURM_SUBMIT_DIR"
mkdir -p runs
python -u -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -u -c "from genetic_algorithm import train; train(generations=2000, P=50, K=250, M=1000, m_chunk=1000, devices=1, save_path='runs/run_low_var.npz', checkpoint_every=50, var_weight=0.05)"
