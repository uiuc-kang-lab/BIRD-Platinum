#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=opensearch_sql_exec
#SBATCH --mail-user=yxx404@illinois.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --partition=ddkang-high
#SBATCH --ntasks-per-node=16
#SBATCH --time=7-00:00:00
#SBATCH --mem=256G

module load anaconda3
source activate opensearch
sh run/run_main.sh