#!/bin/bash -l
#SBATCH --job-name=tc-ga-mnist-full
#SBATCH --account=nano
#SBATCH --partition=etna_gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:V100:1
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=ga-mnist-full-%j.out

module load ml/pytorch

# GA fine-tune of the per-tf-refit MNIST thermodynamic computer (tf=0.4,
# var_weight=0), the THOROUGH configuration:
#   - P = 100 candidate computers per generation (MUT=0.1, 10 elites)
#   - fitness = each computer's cross-entropy over the ENTIRE 60,000-digit
#     TRAIN set (no subsampling) -- the true full-train error, identical every
#     generation.  TEST set is untouched (final accuracy report only).
# Self-contained: refits the activation at (beta=1, tf=0.4), retrains the
# teacher, OM-GD trains the student (warm start), then runs the GA.
#
# NOTE: this is a FRESH run -- P=100 is incompatible with the earlier P=50
# gen-150 checkpoint, so it does not (cannot) resume that.  It checkpoints every
# 25 gens to runs/ckpt_ga_mnist_tf0.4_vw0_seed0_g500.pt and resumes ITSELF on
# resubmission.
#
# RUNTIME: ~12x the old cost (P 50->100 is 2x, K 10k->60k is 6x).  The old run
# was ~9.8 s/gen on a fast local GPU, so estimate ~2 min/gen there and likely
# more on a V100 -> 500 gens is many hours and will span multiple 24h windows;
# just resubmit to continue.  Reduce the generation count (last arg) if you want
# a shorter first pass.
#
# REQUIRES: MNIST in data/ (pre-download on the login node if compute nodes have
# no internet -- see job_tf_scan_mnist.sh).
#
# Args: generations=500, var_weight=0, tf=0.4, seed=0.

cd "$SLURM_SUBMIT_DIR"
if [ ! -f experiment_ga_mnist.py ]; then cd Models/mnist_training; fi
mkdir -p runs
python -u -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -u experiment_ga_mnist.py 500 0 0.4 0
