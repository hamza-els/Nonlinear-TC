#!/bin/bash -l
#SBATCH --job-name=tc-kill-patterns
#SBATCH --account=nano
#SBATCH --partition=etna_gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:V100:1
#SBATCH --time=6:00:00
#SBATCH --output=kill-patterns-%j.out

module load ml/pytorch

# Best kill-timing pattern for the cascaded staged 4x8 computer: 3 window
# shapes (equal / long_tight_long / tight_longer_long) x 2 clocks (0.6, 0.8)
# x 4 seeds, each vs a standard baseline at the same tf.  Requires digital_net
# at WIDTH=8, DEPTH=4 (thermo default).  Checkpoints per seed to
# runs/kill_patterns_results.npz; regenerate the figure locally with:
#     python experiments/experiment_kill_patterns.py plot

# works whether submitted from gd_training (sbatch jobs/job_...) or from
# inside jobs/ (sbatch job_...): cd up until experiments/ is visible
cd "$SLURM_SUBMIT_DIR"
if [ ! -d experiments ]; then cd ..; fi
mkdir -p runs logs
python -u -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -u experiments/experiment_kill_patterns.py
