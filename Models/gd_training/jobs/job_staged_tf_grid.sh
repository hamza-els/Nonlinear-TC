#!/bin/bash -l
#SBATCH --job-name=tc-staged-tf-grid
#SBATCH --account=nano
#SBATCH --partition=etna_gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:V100:1
#SBATCH --time=24:00:00
#SBATCH --output=staged-tf-grid-%j.out

module load ml/pytorch

# phi-vs-observation-time GRID for even-kill staged 4x8 computers: one panel per
# trained clock tf in {0.5, 0.55, ..., 3.0} (51 clocks), 10 seeds each -> 510
# staged students, each with a phi sweep.  Every student is a cascaded 4x8 with
# EVENLY spaced kills (layers die at 0.25/0.5/0.75 tf, sample at tf).  This is a
# LONG run (high-tf students integrate up to ~3000 steps); results checkpoint to
# runs/staged_tf_grid_results.npz after every seed, so a timeout still leaves a
# usable partial grid.  Regenerate the figure locally with:
#     python experiments/experiment_staged_tf_grid.py plot

# works whether submitted from gd_training (sbatch jobs/job_...) or from
# inside jobs/ (sbatch job_...): cd up until experiments/ is visible
cd "$SLURM_SUBMIT_DIR"
if [ ! -d experiments ]; then cd ..; fi
mkdir -p runs logs
# select the deep 4x8 teacher for this job (digital_net reads these at import)
export TC_WIDTH=8 TC_DEPTH=4
python -u -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -u experiments/experiment_staged_tf_grid.py
