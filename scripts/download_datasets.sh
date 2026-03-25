#!/bin/bash
#SBATCH --job-name=oeb-download
#SBATCH --partition=shared
#SBATCH --account=csd403
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/expanse/projects/nemar/eeg_finetuning/pierre/oeb_logs/download_%j.out
#SBATCH --error=/expanse/projects/nemar/eeg_finetuning/pierre/oeb_logs/download_%j.err

cd /home/pguetschel/OpenEEG-Bench-clean
.venv/bin/python scripts/download_datasets.py
