#!/bin/bash

#SBATCH --ntasks=100
#SBATCH --time=23:59:59
#SBATCH --output=/scratch/alpine/adde1214/UFET-Probabilities/logs/recount_all.%j.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=adde1214@colorado.edu

module purge

echo "loading anaconda"
module load anaconda

echo "loading cuda"
module load cuda/12.1.1

echo "activating long_chain"
conda activate long_chain

echo "changing directory"
cd /scratch/alpine/adde1214/UFET-Probabilities

echo "running script"
# python parallel_counter.py /scratch/alpine/adde1214/UFET-Probabilities/data/input/split_4.csv --split train --num_proc 128 


python parallel_counter.py /scratch/alpine/adde1214/UFET-Probabilities/data/input/split_2.csv --split train --num_proc 64 

# python scorer.py /scratch/alpine/adde1214/UFET-Probabilities/data/output/llama_baseline/1B/test_predictions.json