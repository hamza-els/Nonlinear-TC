#!/bin/bash -l
#SBATCH --job-name=tc-ga-mnist-tf04-vw1
#SBATCH --account=nano
#SBATCH --partition=etna_gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:V100:1
#SBATCH --time=4:00:00
#SBATCH --requeue
#SBATCH --output=ga-mnist-tf04-vw1-%j.out

module load ml/pytorch

# GA fine-tune of the PER-TF-REFIT MNIST thermodynamic computer at tf = 0.4,
# with var_weight = 1.
#   fitness = CE(<x_out(tf)>, label) + 1 * mean Var[x_out]
# Self-contained: refits the thermo activation to the real neuron at
# (beta=1, tf=0.4), retrains the teacher with it, OM-GD trains the student
# (the warm start), then runs 1000 GA generations on top.
#
# GA: P=50, 5 elites, M_FIT=20 reset samples, K_FIT=500 fresh digits/gen (CRN).
# RUNTIME: ~7.1 s/gen -> 1000 gens ~ 2.0 h, plus ~3 min warm-start setup.
# Checkpoints every 25 gens to runs/ckpt_ga_mnist_tf0.4_vw1_seed0_g1000.pt
# and resumes on resubmission.
# Outputs: runs/ga_mnist_tf0.4_vw1_seed0_g1000.npz (champion params +
# history + GD-vs-GA test accuracy).
#
# REQUIRES: MNIST in data/ (pre-download on the login node if compute nodes
# have no internet -- see job_tf_scan_mnist.sh).
#
# Args: generations=1000, var_weight=1, tf=0.4, seed=0.

cd "$SLURM_SUBMIT_DIR"
if [ ! -f experiment_ga_mnist.py ]; then cd Models/mnist_training; fi
mkdir -p runs
python -u -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -u experiment_ga_mnist.py 1000 1 0.4 0
