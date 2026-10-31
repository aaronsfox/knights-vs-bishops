#!/usr/bin/env bash
# Engine games on a SLURM cluster: one array task per block of positions,
# each task running several games at once (one game per physical core).
#
# Build fastchess once on a login node first:
#   bash tools/fastchess/build_linux.sh
# Submit from the repository root (create logs/ first), choosing the player:
#
#   Stage 2, self-play on all 57 positions (15 tasks of 4 positions):
#     sbatch --export=ALL,PLAYER=stockfish jobs/selfplay_array.slurm
#     sbatch --export=ALL,PLAYER=caissa    jobs/selfplay_array.slurm
#   Strength ladder, a rung on the key cells (29 positions, 8 tasks):
#     sbatch --array=0-7 --export=ALL,PLAYER=stockfish_n10k,POSITIONS=key jobs/selfplay_array.slurm
#   Topping up the key cells by 1,000 games each (another batch), so each
#   position has 1,500 and each comparison (a position and its twin) 3,000:
#     sbatch --array=0-7 --time=03:00:00 \
#         --export=ALL,PLAYER=stockfish,POSITIONS=key,BATCH=b2,GAMES_PER_LINE=20 jobs/selfplay_array.slurm
#   Stage 3, a Stockfish 11 variant against the base build: 3,000 games from
#   the standard position (50 lines x 60), one task of ~1.5 hours on 16 cores:
#     sbatch --array=0 --time=03:00:00 \
#         --export=ALL,PLAYER=sf11_b050,OPPONENT=sf11,POSITIONS=start,GAMES_PER_LINE=60 \
#         jobs/selfplay_array.slurm
#
# then, once every task has finished:
#   python code/selfplay.py collect
#
# The --array range is the number of blocks minus one; print it with
#   python code/task_positions.py <POSITIONS> --array <POSITIONS_PER_TASK>
#
# Timing: in the Chess960 pilot (16 cores per task, 10+0.1), a position's
# 500 games took 750-1000 s for Stockfish and 880-1020 s for Caissa, so a
# block of 4 positions takes about an hour; the 2-hour limit leaves a margin
# and short limits help backfill. The log prints progress every 5 minutes.
# Finished positions are skipped, so resubmitting the same job resumes it.
#
# Timing matters for time-control games, so each game gets a whole physical
# core (--hint=nomultithread) and fastchess runs as many games at once as
# the task has CPUs. Don't run other work inside these allocations.

#SBATCH --job-name=kvb-selfplay
#SBATCH --array=0-14
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/selfplay_%A_%a.out

set -euo pipefail

PLAYER="${PLAYER:?set PLAYER, e.g. --export=ALL,PLAYER=stockfish}"
POSITIONS="${POSITIONS:-all}"
POSITIONS_PER_TASK="${POSITIONS_PER_TASK:-4}"
CONCURRENCY="${CONCURRENCY:-${SLURM_CPUS_PER_TASK:-1}}"
EXTRA=()
[[ -n "${OPPONENT:-}" ]] && EXTRA+=(--opponent "${OPPONENT}")
[[ -n "${BATCH:-}" ]] && EXTRA+=(--batch "${BATCH}")
[[ -n "${GAMES_PER_LINE:-}" ]] && EXTRA+=(--games-per-line "${GAMES_PER_LINE}")

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

# fastchess's scratch files (openings, in-progress PGN) go to node-local
# storage when the cluster provides it.
WORK_DIR="${TMPDIR:-/tmp}/kvb-selfplay-${SLURM_JOB_ID}"
mkdir -p "${WORK_DIR}"
trap 'rm -rf "${WORK_DIR}"' EXIT

echo "player=${PLAYER} opponent=${OPPONENT:-} positions=${TASK_POSITIONS} batch=${BATCH:-} concurrency=${CONCURRENCY} host=$(hostname)"
python code/selfplay.py run --player "${PLAYER}" --positions "${TASK_POSITIONS}" \
    --concurrency "${CONCURRENCY}" --work-dir "${WORK_DIR}" ${EXTRA[@]+"${EXTRA[@]}"}
