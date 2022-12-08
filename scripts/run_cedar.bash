#!/bin/bash
#SBATCH --account=def-florian7_gpu 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=dreamer
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --gpus-per-node=v100l:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8

echo `date`: "Job $SLURM_JOB_ID is allocated resource"

ENV_ROOT=/home/guqiao/projects/def-florian7/guqiao/envs/dreamer
SRC_ROOT=/home/guqiao/src/csc2626_dreamer

echo 'set up the environment'
module load python/3.9 scipy-stack
source $ENV_ROOT/bin/activate

cd $SRC_ROOT

$* --result_root $SLURM_TMPDIR/models/

echo `date`: "Job $SLURM_JOB_ID finished running, exit code: $?"