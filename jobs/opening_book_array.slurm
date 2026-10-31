#!/usr/bin/env bash
# Build the opening book on a SLURM cluster: one array task per block of
# positions, one CPU each.
#
# Submit from the repository root (create logs/ first):
#   sbatch jobs/opening_book_array.slurm
# then, once every task has finished:
#   python code/opening_book_generator.py collect
#
# For a list of positions instead of all 57 (e.g. the sample positions), set POSITIONS
# to anything --positions accepts; task i then takes the i-th block of
# POSITIONS_PER_TASK positions from the list:
#   sbatch --array=0-13 --export=ALL,POSITIONS=@config/sample_positions.txt,POSITIONS_PER_TASK=1 \
#       jobs/opening_book_array.slurm
#
# Settings come from config/opening_book.toml. In Chess960 the book cost
# ~270 core-hours for 960 positions (~17 minutes per position), so blocks of
# 4 positions take about 70 minutes. With POSITIONS_PER_TASK=4 the 57
# positions need array indices 0-14; print the range for another spec with
#   python code/task_positions.py <POSITIONS> --array <POSITIONS_PER_TASK>
# Finished positions are skipped, so resubmitting the same job resumes it.

#SBATCH --job-name=kvb-opening-book
#SBATCH --array=0-14
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=03:00:00
#SBATCH --output=logs/opening_book_%A_%a.out

set -euo pipefail

POSITIONS="${POSITIONS:-all}"
POSITIONS_PER_TASK="${POSITIONS_PER_TASK:-4}"

# Any modules the engine binary needs at run time, e.g. MODULES="GCC/10.2.0".
MODULES="${MODULES:-}"
set +u
if [[ -n "${MODULES}" ]]; then
    module load ${MODULES}
fi
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-kvb}"
set -u

TASK_POSITIONS=$(python code/task_positions.py "${POSITIONS}" "${SLURM_ARRAY_TASK_ID}" "${POSITIONS_PER_TASK}")
if [[ -z "${TASK_POSITIONS}" ]]; then
    echo "Task ${SLURM_ARRAY_TASK_ID} has no positions; exiting."
    exit 0
fi

echo "opening book positions=${TASK_POSITIONS} host=$(hostname)"
python code/opening_book_generator.py build --positions "${TASK_POSITIONS}"
