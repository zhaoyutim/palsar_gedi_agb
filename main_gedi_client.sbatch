#!/usr/bin/env bash
#SBATCH --output /geoinfo_vol1/zhao2/run_logs/%J_slurm.out
#SBATCH --error  /geoinfo_vol1/zhao2/run_logs/%J_slurm.err
#SBATCH --mem  32GB
#SBATCH --cpus-per-task 1

PYTHONUNBUFFERED=1; python main_gedi_client.py -region $1
