#!/bin/bash -l
#SBATCH --job-name=tc-freq-tf-grid
#SBATCH --account=nano
#SBATCH --partition=etna_gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:V100:1
#SBATCH --time=24:00:00
#SBATCH --output=freq-tf-grid-%j.out

module load ml/pytorch

# 5 x 5 grid: target cos(f pi z) for f in {1,2,3,4,5} (rows) x trained clock
# tf in {0.4, 0.8, 1.2, 1.6, 2.0} (columns), 5 seeds per cell = 125 computers.
# Config is printed at the top of the log; results checkpoint to
# runs/freq_tf_grid_results.npz after every (frequency, seed), so a walltime
# kill preserves completed work.  Per-model weights land in runs/weights/.
# The figure is optional on the node -- regenerate locally with:
#     python experiments/experiment_freq_tf_grid.py plot

# works whether submitted from gd_training (sbatch jobs/job_...) or from
# inside jobs/ (sbatch job_...): cd up until experiments/ is visible
cd "$SLURM_SUBMIT_DIR"
if [ ! -d experiments ]; then cd ..; fi
mkdir -p runs logs
python -u -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -u experiments/experiment_freq_tf_grid.py
