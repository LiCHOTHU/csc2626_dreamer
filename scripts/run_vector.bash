#!/bin/bash
#SBATCH --job-name=dreamer
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=rtx6000
#SBATCH --gres=gpu:1
#SBATCH --qos=normal
#SBATCH --open-mode=append

echo `date`: "Job $SLURM_JOB_ID is allocated resource"

SRC_ROOT=/h/guqiao/src/csc2626_dreamer/

# Link folder for checkpointing
ln -sfn /checkpoint/${USER}/${SLURM_JOB_ID} $SRC_ROOT/checkpoint/${SLURM_JOB_ID}

# an empty file preventing the checkpoint folder from being purged
touch /checkpoint/${USER}/${SLURM_JOB_ID}/DELAYPURGE

# Setup environment
source /h/guqiao/anaconda3/bin/activate
conda activate gad

# Run training
$* --ckpt_folder $SRC_ROOT/checkpoint/${SLURM_JOB_ID} --preemptive

echo `date`: "Job $SLURM_JOB_ID finished running, exit code: $?"

date=$(date '+%Y-%m-%d')
archive=$HOME/finished_jobs/$date/$SLURM_JOB_ID
mkdir -p $archive

cp ./$SLURM_JOB_ID.out $archive/job.out
cp ./$SLURM_JOB_ID.err $archive/job.err
