#!/bin/bash -l
#SBATCH --job-name=tc-tf-scan-mnist
#SBATCH --account=nano
#SBATCH --partition=etna_gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:V100:1
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=tf-scan-mnist-%j.out

module load ml/pytorch

# tf scan for the MNIST thermodynamic computer: top-1 test accuracy vs
# observation time, training a fresh student AT each tf, 10 seeds each.
# Locked config: THERMO-activation teacher (teacher_thermo.pt, 97.06%) +
# PROPORTIONAL guide k=100 (best of the teacher x guide comparison, 93.8% @tf=0.2).
# tfs = {0.05..1.5} (13 values) x 10 seeds = 130 students, 15 epochs each.
#
# RUNTIME: ~53 s/student at tf=0.2, ~323 s at tf=1.5 -> ~4.5 h total (fits this
# 12 h wall).  Checkpoints after every (tf, seed) to
# runs/tf_scan_mnist_results.npz and RESUMES (skips finished cells) on
# resubmission, so a pre-emption/over-run loses nothing.
#
# REQUIRES on the node:
#   - runs/teacher_thermo.pt  (the locked thermo teacher; sync/commit it -- it
#     is under runs/ which is gitignored, so `git add -f` it or scp it over)
#   - MNIST data: teacher_net.load_mnist downloads via torchvision to data/.
#     If compute nodes have no internet, pre-download on the login node first:
#       python -c "from torchvision import datasets; \
#         datasets.MNIST('data',train=True,download=True); \
#         datasets.MNIST('data',train=False,download=True)"
#
# Regenerate the figure locally after pulling the npz:
#   python experiment_tf_scan_mnist.py plot

cd "$SLURM_SUBMIT_DIR"
if [ ! -f experiment_tf_scan_mnist.py ]; then cd Models/mnist_training; fi
mkdir -p runs
python -u -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -u experiment_tf_scan_mnist.py
