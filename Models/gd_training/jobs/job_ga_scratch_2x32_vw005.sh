#!/bin/bash -l
#SBATCH --job-name=tc-ga-scratch-vw005
#SBATCH --account=nano
#SBATCH --partition=etna_gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:V100:1
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=ga-scratch-vw005-%j.out

module load ml/pytorch

# GA-train a 2x32 thermo computer FROM SCRATCH (fresh GD warm start via
# run(seed=4), NOT the saved champion), 6000 generations, LIGHT variance penalty
# (fitness = bias + 0.05 * Var[y]).  Same starting seed as the vw=0 job, so the
# pair isolates the effect of the variance term.  digital_net stays at its
# default 2x32 (no TC_WIDTH/DEPTH).
#
# RUNTIME: at K=100/M_FIT=500 a generation is ~10 s, so 6000 gens ~ 17 h and
# should finish inside this 24 h wall.  As insurance the experiment checkpoints
# the GA population every 100 gens to
#   runs/ckpt_ga_finetune_thermo_seed4_g6000_vw0.05_M500_crn.pt
# and RESUMES from it automatically; if the job is ever pre-empted or over-runs,
# just resubmit the SAME job to continue:
#   sbatch jobs/job_ga_scratch_2x32_vw005.sh
# Each checkpoint also refreshes runs/run_ga_finetune_thermo_seed4_g6000_vw0.05_M500_crn.npz
# with the best-so-far model, so a killed run is never wasted.  The checkpoint
# is deleted on clean completion.
#
# Args: seed=4, generations=6000, var_weight=0.05, activation=thermo (no warm file).

# works whether submitted from gd_training (sbatch jobs/job_...) or from
# inside jobs/ (sbatch job_...): cd up until experiments/ is visible
cd "$SLURM_SUBMIT_DIR"
if [ ! -d experiments ]; then cd ..; fi
mkdir -p runs logs
python -u -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -u experiments/experiment_ga_finetune.py 4 6000 0.05 thermo
