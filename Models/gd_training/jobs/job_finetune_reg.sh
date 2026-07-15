#!/bin/bash -l
#SBATCH --job-name=tc-finetune
#SBATCH --account=nano
#SBATCH --partition=etna_gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:V100:1
#SBATCH --time=24:00:00
#SBATCH --output=finetune-reg-%j.out

module load ml/pytorch

# GA fine-tune of the GD-trained thermodynamic computer -- REGULAR objective
# (fitness = bias only, var_weight = 0).
#
# Submit from Models/gd_training:   sbatch jobs/job_finetune_reg.sh
# Args: seed=0, generations=200, var_weight=0.  Protocol (P, K, M_FIT, CRN,
# ...) comes from the script's constants and is printed at the top of the log
# and encoded in the output filename tag.

cd "$SLURM_SUBMIT_DIR"
mkdir -p runs logs
python -u -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -u experiments/experiment_ga_finetune.py 0 200 0
