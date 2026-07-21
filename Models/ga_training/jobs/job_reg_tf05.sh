#!/bin/bash -l
#SBATCH --job-name=tc-train-tf05-vw0
#SBATCH --account=nano
#SBATCH --partition=etna_gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:V100:1
#SBATCH --time=18:00:00
#SBATCH --output=logs/train-tf05-vw0-%j.out

module load ml/pytorch

# GA-train the native 4x8 thermodynamic computer FROM SCRATCH (random init, no
# GD), var_weight = 0 (pure MSE fitness).  Identical to job_reg.sh except the
# observation clock is tf = 0.5 (default was 1.0).  Checkpoints every 50 gens.

cd "$SLURM_SUBMIT_DIR"
mkdir -p runs
python -u -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -u -c "from genetic_algorithm import train; train(generations=3000, P=50, K=250, M=1000, m_chunk=1000, devices=1, save_path='runs/run_reg_tf05.npz', checkpoint_every=50, tf=0.5)"
