#!/bin/bash -l
#SBATCH --job-name=tc-finetune-thermo-vw001
#SBATCH --account=nano
#SBATCH --partition=etna_gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:V100:1
#SBATCH --time=24:00:00
#SBATCH --output=finetune-thermo-vw001-%j.out

module load ml/pytorch

# GA fine-tune of the 2x32 THERMO CHAMPION (best of the 100-seed sweep:
# RMSE 0.0093, true bias 0.0064, Var 0.0137), with a LIGHT variance penalty
# (fitness = bias + 0.01 * Var[y]).  Warm start is loaded DIRECTLY from the
# champion weights file (portable) rather than a seed -- seeds do not
# reproduce across platform/torch version.
#
# Args: seed=42 (GA RNG only), generations=200, var_weight=0.01,
#       activation=thermo, warm=runs/weights/CHAMPION_thermo_2x32.pt.
# REQUIRES: runs/weights/CHAMPION_thermo_2x32.pt present on the node (sync it
# from local, or copy the cluster's 20260717-155526_run_seed42 file to that
# name).  The final figure is skipped (no teacher when warm-starting).
# Protocol (P, K, M_FIT, CRN, ...) prints at the top of the log and is encoded
# in the output filename tag.
# Outputs: runs/run_ga_finetune_thermo_seed4_g200_vw0.01_M1000_crn.npz + log.

# works whether submitted from gd_training (sbatch jobs/job_...) or from
# inside jobs/ (sbatch job_...): cd up until experiments/ is visible
cd "$SLURM_SUBMIT_DIR"
if [ ! -d experiments ]; then cd ..; fi
mkdir -p runs logs
python -u -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -u experiments/experiment_ga_finetune.py 42 200 0.01 thermo runs/weights/CHAMPION_thermo_2x32.pt
