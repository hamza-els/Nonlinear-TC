#!/bin/bash -l
#SBATCH --job-name=tc-finetune-thermo-vw1
#SBATCH --account=nano
#SBATCH --partition=etna_gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:V100:1
#SBATCH --time=24:00:00
#SBATCH --output=finetune-thermo-vw1-%j.out

module load ml/pytorch

# GA fine-tune from the THERMO-ACTIVATION GD champion (seed 4), with a HEAVY
# variance penalty (fitness = bias + 1.0 * Var[y]).  Anchors the high end of
# the var_weight sweep (vw=0, 0.01, 0.05, 1.0): here the Var term dominates the
# fitness (Var ~ 0.005-0.01 vs bias^2 ~ 2e-5), so the GA should drive Var[y] as
# low as the dynamics allow, trading substantial bias for it -- the quietest
# single-shot computer of the set, useful for probing the variance floor.
#
# Teacher activation: sigma(x) = 0.5617*(x^2/(x^2+13.53))*cbrt(x) + tanh(x).
# Reference (thermo, same settings, high-M true bias / Var[y]):
#   vw=0     bias 0.0041, Var 0.0065
#   vw=0.05  bias 0.0050, Var 0.0048
#
# Args: seed=4, generations=200, var_weight=1.0, activation=thermo.
# Protocol (P, K, M_FIT, CRN, ...) prints at the top of the log and is encoded
# in the output filename tag.
# Outputs: runs/run_ga_finetune_thermo_seed4_g200_vw1_M1000_crn.npz + log.

# works whether submitted from gd_training (sbatch jobs/job_...) or from
# inside jobs/ (sbatch job_...): cd up until experiments/ is visible
cd "$SLURM_SUBMIT_DIR"
if [ ! -d experiments ]; then cd ..; fi
mkdir -p runs logs
python -u -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -u experiments/experiment_ga_finetune.py 4 200 1.0 thermo
