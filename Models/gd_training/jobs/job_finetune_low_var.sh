#!/bin/bash -l
#SBATCH --job-name=tc-finetune-low-var
#SBATCH --account=nano
#SBATCH --partition=etna_gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:V100:1
#SBATCH --time=24:00:00
#SBATCH --output=finetune-lowvar-%j.out

module load ml/pytorch

# GA fine-tune of the GD-trained thermodynamic computer -- LOW-VAR objective
# (fitness = bias + 0.05 * Var[y]), matching the ga_training low_var runs.
#
# Submit from Models/gd_training:   sbatch jobs/job_finetune_low_var.sh
# Args: seed=0, generations=200, var_weight=0.05.  Protocol (P, K, M_FIT,
# CRN, ...) comes from the script's constants and is printed at the top of
# the log and encoded in the output filename tag.
# NOTE: at the GD starting point 0.05*Var (~1e-3) dominates bias (~2e-4), so
# expect this run to spend most of its effort quieting the machine, possibly
# trading some bias away -- same trade as the cluster low_var GA runs.

# works whether submitted from gd_training (sbatch jobs/job_...) or from
# inside jobs/ (sbatch job_...): cd up until experiments/ is visible
cd "$SLURM_SUBMIT_DIR"
if [ ! -d experiments ]; then cd ..; fi
mkdir -p runs logs
python -u -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -u experiments/experiment_ga_finetune.py 0 200 0.05
