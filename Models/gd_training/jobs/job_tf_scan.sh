#!/bin/bash -l
#SBATCH --job-name=tc-tf-scan
#SBATCH --account=nano
#SBATCH --partition=etna_gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:V100:1
#SBATCH --time=24:00:00
#SBATCH --output=tf-scan-%j.out

module load ml/pytorch

# Clock-time scan: 31 trained tf values (0.05, 0.1 .. 3.0) x 5 seeds.
# Config is printed at the top of the log; results checkpoint to
# runs/tf_scan_results.npz after every seed.  The figure is optional on the
# node -- regenerate locally with:  python experiments/experiment_tf_scan.py plot

# works whether submitted from gd_training (sbatch jobs/job_...) or from
# inside jobs/ (sbatch job_...): cd up until experiments/ is visible
cd "$SLURM_SUBMIT_DIR"
if [ ! -d experiments ]; then cd ..; fi
mkdir -p runs logs
python -u -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -u experiments/experiment_tf_scan.py
