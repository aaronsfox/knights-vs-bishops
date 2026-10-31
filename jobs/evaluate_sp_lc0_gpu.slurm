#!/usr/bin/env bash
# evaluate_sp for Lc0 on a GPU node: one array task per block of positions,
# one GPU each, several Lc0 processes sharing that GPU.
#
# Submit from the repository root (create logs/ first):
#   sbatch jobs/evaluate_sp_lc0_gpu.slurm
#
# One single-threaded Lc0 search keeps an A100 mostly idle (~5k nodes/s), so
# each task runs PROCS_PER_GPU independent Lc0 processes on its GPU, each on
# its own share of the task's positions. Every search is still a fresh,
# single-threaded, node-limited search, so results don't depend on how many
# processes share the GPU; only the recorded times do. Sharing a GPU may
# add little throughput, so check PROCS_PER_GPU on the pilot.
#
# Defaults: POSITIONS_PER_TASK=20 with PROCS_PER_GPU=4, so array 0-2 covers
# the 57 positions, 5 per process. The time limit is a starting guess (a few
# minutes per position on an A100); set it from the pilot. For another spec or block size, print the
# --array range with
#   python code/task_positions.py <POSITIONS> --array <POSITIONS_PER_TASK>
# Finished positions are skipped, so resubmitting the same job resumes it.

#SBATCH --job-name=kvb-evaluate-sp-lc0
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --array=0-2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/evaluate_sp_%x_%A_%a.out

set -euo pipefail

ENGINE="${ENGINE:-lc0}"
POSITIONS="${POSITIONS:-all}"
POSITIONS_PER_TASK="${POSITIONS_PER_TASK:-20}"
PROCS_PER_GPU="${PROCS_PER_GPU:-4}"

# Run-time environment: the CUDA libraries Lc0 was built against (libcublas
# is otherwise missing on the compute nodes), and the project's Python
# environment. Override with --export=ALL,MODULES="...".
MODULES="${MODULES:-CUDA/12.2.2}"
set +u
module load ${MODULES}
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-kvb}"
set -u

TASK_POSITIONS=$(python code/task_positions.py "${POSITIONS}" "${SLURM_ARRAY_TASK_ID}" "${POSITIONS_PER_TASK}")
if [[ -z "${TASK_POSITIONS}" ]]; then
    echo "Task ${SLURM_ARRAY_TASK_ID} has no positions; exiting."
    exit 0
fi
IFS=',' read -r -a IDS <<< "${TASK_POSITIONS}"

echo "engine=${ENGINE} positions=${TASK_POSITIONS} processes=${PROCS_PER_GPU} host=$(hostname)"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true

# Deal the positions out in turn (process 0 gets the 1st, (P+1)th, ...), so
# each process gets a similar mix. Each process logs to its own file.
LOG_BASE="logs/evaluate_sp_${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
pids=()
for (( p = 0; p < PROCS_PER_GPU; p++ )); do
    mine=()
    for (( i = p; i < ${#IDS[@]}; i += PROCS_PER_GPU )); do
        mine+=("${IDS[$i]}")
    done
    (( ${#mine[@]} == 0 )) && continue
    positions=$(IFS=','; echo "${mine[*]}")
    echo "process ${p}: positions ${positions} (log ${LOG_BASE}_p${p}.log)"
    python code/evaluate_sp.py run --engine "${ENGINE}" --positions "${positions}" \
        > "${LOG_BASE}_p${p}.log" 2>&1 &
    pids+=($!)
done

status=0
for pid in "${pids[@]}"; do
    wait "${pid}" || status=1
done
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader || true
if (( status != 0 )); then
    echo "at least one Lc0 process failed; see ${LOG_BASE}_p*.log" >&2
fi
exit "${status}"
