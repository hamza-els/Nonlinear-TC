#!/bin/bash -l
#SBATCH --job-name=tc-ga-scratch-4x8-vw0
#SBATCH --account=nano
#SBATCH --partition=etna_gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:V100:1
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=ga-scratch-4x8-vw0-%j.out

module load ml/pytorch

# PURE GA FROM SCRATCH (random init, no teacher, no GD) on the gd-architecture
# thermodynamic computer: 32 hidden + 8 output, all-to-all coupled, 6-poly
# input, native 2x+4x^3 neuron, tf = 0.5.  var_weight = 0 (pure MSE fitness),
# 3000 generations.  Selects the 4x8 (N=40) shape via TC_WIDTH/TC_DEPTH.
#
# RUNTIME: est ~25 s/gen -> 3000 gens ~ 20 h (should fit this 24 h wall).  The
# GA population is checkpointed every 100 gens to
#   runs/ckpt_ga_scratch_4x8_vw0_tf0.5_seed0_g3000.pt
# and resumes automatically on resubmission if pre-empted / over-run.  Each
# checkpoint also refreshes runs/run_ga_scratch_4x8_vw0_tf0.5_seed0_g3000.npz
# with the best-so-far computer; the checkpoint is deleted on clean completion.
#
# Args: gens=3000, var_weight=0, seed=0, tf=0.5.

cd "$SLURM_SUBMIT_DIR"
if [ ! -d experiments ]; then cd ..; fi
mkdir -p runs logs
export TC_WIDTH=8 TC_DEPTH=4
python -u -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -u experiments/experiment_ga_scratch.py 3000 0 0 0.5
