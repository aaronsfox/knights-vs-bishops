#!/usr/bin/env bash
# evaluate_sp (the screening pass) on a SLURM cluster: one array task per
# position by default, one CPU each.
#
# For the CPU engines (Stockfish, Caissa, the Stockfish 11 builds); Lc0 uses
# jobs/evaluate_sp_lc0_gpu.slurm. Submit from the repository root, e.g.
#   sbatch --export=ALL,ENGINE=stockfish jobs/evaluate_sp_array.slurm
#   sbatch --export=ALL,ENGINE=stockfish,MULTIPV=5 jobs/evaluate_sp_array.slurm
#
# With POSITIONS_PER_TASK=1 the 57 positions need array indices 0-56.
# Stockfish takes ~40 minutes per position (budgets up to 1e9 at ~0.5M
# nodes/s). For another spec or block size, print the --array range with
#   python code/task_positions.py <POSITIONS> --array <POSITIONS_PER_TASK>
# Finished positions are skipped, so resubmitting the same job resumes it.

#SBATCH --job-name=kvb-evaluate-sp
#SBATCH --array=0-56
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --output=logs/evaluate_sp_%x_%A_%a.out

set -euo pipefail

ENGINE="${ENGINE:?set ENGINE, e.g. --export=ALL,ENGINE=stockfish}"
MULTIPV="${MULTIPV:-1}"
POSITIONS="${POSITIONS:-all}"
POSITIONS_PER_TASK="${POSITIONS_PER_TASK:-1}"

# Run-time environment: any modules the engine binaries need (e.g. the GCC
# they were built with, if they report GLIBCXX errors), and the project's
# Python environment. Set with --export=ALL,MODULES="GCC/10.2.0".
MODULES="${MODULES:-}"
# Module and conda scripts can reference unset variables, so relax `set -u`.
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

echo "engine=${ENGINE} multipv=${MULTIPV} positions=${TASK_POSITIONS} host=$(hostname)"
python code/evaluate_sp.py run --engine "${ENGINE}" --multipv "${MULTIPV}" \
    --positions "${TASK_POSITIONS}"
