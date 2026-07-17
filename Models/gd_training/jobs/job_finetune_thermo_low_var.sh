#!/bin/bash -l
#SBATCH --job-name=tc-finetune-thermo-lowvar
#SBATCH --account=nano
#SBATCH --partition=etna_gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:V100:1
#SBATCH --time=24:00:00
#SBATCH --output=finetune-thermo-lowvar-%j.out

module load ml/pytorch

# GA fine-tune warm-started from the THERMO-ACTIVATION GD champion, with the
# LOW-VAR objective (fitness = bias + 0.05 * Var[y]), matching the low_var
# condition used for the tanh fine-tune and the ga_training runs.
#
# Teacher activation: sigma(x) = 0.5617*(x^2/(x^2+13.53))*cbrt(x) + tanh(x).
# Seed 4 is the best of 10 thermo seeds (GD RMSE 0.0104, Var[y] 0.011).
#
# Args: seed=4, generations=200, var_weight=0.05, activation=thermo.
# Reference points from the tanh pipeline (same settings):
#   regular (vw=0)     0.0157 -> 0.0077, Var 0.022 -> 0.013
#   low-var (vw=0.05)  0.0157 -> 0.0094, Var 0.022 -> 0.010
# The thermo champion already starts at Var 0.011 -- i.e. at the level the
# tanh low-var run needed 200 generations to reach -- so this run tests how
# much further the variance can be pushed from an already-quiet start.
# Protocol (P, K, M_FIT, CRN, ...) is printed at the top of the log and
# encoded in the output filename tag.
# Outputs: runs/run_ga_finetune_thermo_seed4_g200_vw0.05_M1000_crn.npz + log.

# works whether submitted from gd_training (sbatch jobs/job_...) or from
# inside jobs/ (sbatch job_...): cd up until experiments/ is visible
cd "$SLURM_SUBMIT_DIR"
if [ ! -d experiments ]; then cd ..; fi
mkdir -p runs logs
python -u -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -u experiments/experiment_ga_finetune.py 4 200 0.05 thermo
