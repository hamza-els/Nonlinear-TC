#!/bin/bash -l
#SBATCH --job-name=tc-finetune-thermo
#SBATCH --account=nano
#SBATCH --partition=etna_gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:V100:1
#SBATCH --time=24:00:00
#SBATCH --output=finetune-thermo-%j.out

module load ml/pytorch

# GA fine-tune warm-started from the THERMO-ACTIVATION GD champion.
# The teacher uses the fitted thermodynamic-neuron activation
#   sigma(x) = 0.5617*(x^2/(x^2+13.53))*cbrt(x) + tanh(x)
# instead of tanh.  Seed 4 is the best of 10 thermo seeds (GD RMSE 0.0104,
# vs 0.0129 for the best of 10 tanh seeds).
#
# Args: seed=4, generations=200, var_weight=0, activation=thermo.
# Reference point: the tanh pipeline refined 0.0157 -> 0.0077 with these same
# settings, so the question is whether thermo's better start refines further.
# Protocol (P, K, M_FIT, CRN, ...) is printed at the top of the log and
# encoded in the output filename tag.
# Outputs: runs/run_ga_finetune_thermo_seed4_g200_vw0_M1000_crn.npz + log.

# works whether submitted from gd_training (sbatch jobs/job_...) or from
# inside jobs/ (sbatch job_...): cd up until experiments/ is visible
cd "$SLURM_SUBMIT_DIR"
if [ ! -d experiments ]; then cd ..; fi
mkdir -p runs logs
python -u -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -u experiments/experiment_ga_finetune.py 4 200 0 thermo
