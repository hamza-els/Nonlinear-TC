#!/bin/bash -l
#SBATCH --job-name=tc-ga-mnist-resume
#SBATCH --account=nano
#SBATCH --partition=etna_gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:V100:1
#SBATCH --time=6:00:00
#SBATCH --requeue
#SBATCH --output=ga-mnist-resume-%j.out

module load ml/pytorch

# Resume the big-MUT MNIST GA fine-tune (tf=0.4, var_weight=0, MUT=0.1,
# N_ELITE=10) from the gen-150 checkpoint and run 1000 MORE generations
# (-> gen 1150 total).  The local run reached gen 150 (CE 0.203, test acc
# 94.80%, +0.36% over the OM-GD baseline) before the local GPU slept; this
# continues it to see where big-MUT search plateaus.
#
# HOW RESUME WORKS: the checkpoint filename encodes the total gen count, so the
# gen-150 checkpoint has been cloned to the g1150 tag
#   runs/ckpt_ga_mnist_tf0.4_vw0_seed0_g1150.pt   (gen=150 inside)
# and running with GENS=1150 makes the experiment resume at gen 150 and loop to
# 1150.  Fitness is on a FIXED 10k TRAIN-set batch; the TEST set is only used for
# the final accuracy report (no leakage).  It re-runs the ~3 min warm-start
# (deterministic, seed 0) before loading the checkpoint -- harmless.
# Checkpoints every 25 gens, so a wall-limit hit resumes on resubmission.
#
# RUNTIME: ~9.8 s/gen -> 1000 gens ~ 2.7 h (+ ~3 min warm start).
#
# REQUIRES on the node:
#   - runs/ckpt_ga_mnist_tf0.4_vw0_seed0_g1150.pt  (the cloned checkpoint --
#     sync it: it is under runs/ (gitignored), so `git add -f` it or scp it)
#   - MNIST in data/ (pre-download on the login node if needed).
#
# Args: generations=1150 (resumes at 150 -> 1150), var_weight=0, tf=0.4, seed=0.

cd "$SLURM_SUBMIT_DIR"
if [ ! -f experiment_ga_mnist.py ]; then cd Models/mnist_training; fi
mkdir -p runs
CKPT=runs/ckpt_ga_mnist_tf0.4_vw0_seed0_g1150.pt
test -f "$CKPT" && echo "checkpoint present ($(python -c "import torch;print(torch.load('$CKPT',map_location='cpu')['gen'])") gens done)" \
                || echo "WARNING: $CKPT missing -- will start FRESH from gen 0"
python -u -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -u experiment_ga_mnist.py 1150 0 0.4 0
