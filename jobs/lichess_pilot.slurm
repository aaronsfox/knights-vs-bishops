#!/usr/bin/env bash
# Human-play pilot: stream one month of Lichess games (high-rated classical
# by default), extract the bishop/knight events and write the exploratory
# summary, all on a compute node (needs internet there). The month is
# streamed and filtered as it arrives; only kept games are written.
#
# Submit from the repository root (create logs/ first):
#   sbatch jobs/lichess_pilot.slurm
#   sbatch --export=ALL,MONTH=2026-09,MIN_RATING=2200 jobs/lichess_pilot.slurm
# The fetch is skipped when the month is already parsed, so this can also
# follow jobs/lichess_fetch.slurm (which streams many months in parallel).
#
# Afterwards, commit the summary, tables, figures and the small sample PGN
# (results/lichess/analysis/, results/lichess/events/, results/figures/,
# results/lichess/lichess_*_pilot.*); the parsed games (results/lichess/raw/)
# hold the full movetext and stay local.

#SBATCH --job-name=kvb-lichess-pilot
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/lichess_pilot_%j.out

set -euo pipefail

MONTH="${MONTH:-2026-10}"
CATEGORIES="${CATEGORIES:-classical}"
MIN_RATING="${MIN_RATING:-2000}"
TAG="${TAG:-pilot}"

set +u
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-kvb}"
set -u

echo "month=${MONTH} categories=${CATEGORIES} min_rating=${MIN_RATING} tag=${TAG} host=$(hostname)"
python code/lichess_games.py fetch --months "${MONTH}" --categories "${CATEGORIES}" \
    --min-mean-rating "${MIN_RATING}" --tag "${TAG}"
python code/lichess_games.py summarise --tag "${TAG}"
python code/lichess_games.py sample --tag "${TAG}" --n 2000
python code/human_events.py run --tag "${TAG}" --workers "${SLURM_CPUS_PER_TASK:-1}"
python code/explore_human_pilot.py --tag "${TAG}"
