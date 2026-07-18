#!/bin/bash -l
#SBATCH --job-name=tc-tf-sweep-4x8
#SBATCH --account=nano
#SBATCH --partition=etna_gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:V100:1
#SBATCH --time=4:00:00
#SBATCH --output=tf-sweep-4x8-%j.out

module load ml/pytorch

# Optimal observation time for the deep 4x8 architecture: tf in {0.2..1.0},
# 5 seeds each (45 students; teacher trained once per seed).  Standard
# non-staged computer.  Requires digital_net at WIDTH=8, DEPTH=4.  Results
# checkpoint to runs/tf_sweep_4x8_results.npz after every seed; regenerate the
# figure locally with: python experiments/experiment_tf_sweep_4x8.py plot

# works whether submitted from gd_training (sbatch jobs/job_...) or from
# inside jobs/ (sbatch job_...): cd up until experiments/ is visible
cd "$SLURM_SUBMIT_DIR"
if [ ! -d experiments ]; then cd ..; fi
mkdir -p runs logs
# select the deep 4x8 teacher for this job (digital_net reads these at import);
# leaves the checked-in default of 2x32 untouched for every other job.
export TC_WIDTH=8 TC_DEPTH=4
python -u -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -u experiments/experiment_tf_sweep_4x8.py
