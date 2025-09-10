#!/bin/bash
#SBATCH -A m3246         # project name
#SBATCH -J production    # job name
#SBATCH -C cpu           # cpu or gpu partition
#SBATCH -q shared        # queue type
#SBATCH -t 24:00:00       # wall time in HH:MM:SS
#SBATCH -N 1             # number of nodes requested
#SBATCH -n 1             # tasks per node
#SBATCH -c 32            # num cpus
#SBATCH -o ./batch_outputs/%x_%j.out     # %x - job name, %j - job id     


source setup.sh

law index

srun law run SubmitSignalsAllMass --version eventgen_production --process qq --n-events 500000 --cluster-mode slurm --workers 144
