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
# Args: seed=0, generations=200, var_weight=0.  Protocol per the script's
# constants: P=50, N_ELITE=5, K=256, M_FIT=1000, CRN=False (original-GA-style
# independent noise), tf/beta inherited from thermo_student.py (0.40 / 10).
# Outputs: runs/run_ga_finetune_seed0_g200_vw0_M1000.npz + figure + this log.
# NOTE: measured ~200 s/gen at P=32/K=128 locally; at P=50/K=256 expect
# roughly 3x that per generation -- trim generations if walltime is capped.

cd "$SLURM_SUBMIT_DIR"
mkdir -p runs logs
python -u -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -u experiments/experiment_ga_finetune.py 0 200 0
