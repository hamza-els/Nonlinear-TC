#!/bin/bash -l
#SBATCH --job-name=tc-seed-sweep
#SBATCH --account=nano
#SBATCH --partition=etna_gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:V100:1
#SBATCH --time=12:00:00
#SBATCH --output=seed-sweep-%j.out

module load ml/pytorch

# 100 seeds each of the tanh and thermo teacher activations (200 trainings) at
# the current champion config (tf, K, beta, exact guides).  Records per-seed
# accuracy, noise, teacher fit, activation magnitudes, saturation, and
# trackability to runs/seed_sweep_results.npz for downstream trend analysis
# (does thermo give a tighter / safer distribution?  what predicts a good
# student?).  Checkpoints every 5 seeds so a walltime kill preserves progress.
#
# NOTE: run() saves a per-model weight file to runs/weights/ for every
# training, so this run will drop ~200 small .pt files there.

# works whether submitted from gd_training (sbatch jobs/job_...) or from
# inside jobs/ (sbatch job_...): cd up until experiments/ is visible
cd "$SLURM_SUBMIT_DIR"
if [ ! -d experiments ]; then cd ..; fi
mkdir -p runs logs
python -u -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -u experiments/experiment_seed_sweep.py 100
