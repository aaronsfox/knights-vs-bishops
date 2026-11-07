#!/usr/bin/env bash
# Human-play pilot on a SLURM cluster: parse one month of Lichess standard
# games already downloaded to data/lichess/, keep high-rated classical games,
# extract the bishop/knight events and write the exploratory summary.
#
# Download on the login node first (compute nodes usually have no internet;
# a month is tens of GB, so put data/ on scratch or project storage if home
# is small):
#   mkdir -p data/lichess logs
#   wget -c -P data/lichess https://database.lichess.org/standard/lichess_db_standard_rated_2026-10.pgn.zst
# then, from the repository root:
#   sbatch jobs/lichess_pilot.slurm
#   sbatch --export=ALL,MONTH=2026-09,MIN_RATING=2200 jobs/lichess_pilot.slurm
#
# Reading a month means decompressing and scanning every game's headers
# (roughly 100 million games), so allow a few hours; the event extraction
# and summary on the kept games take minutes. Finished steps are skipped on
# resubmission.
#
# Afterwards, commit the summary, tables, figures and the small sample PGN
# (results/lichess/analysis/, results/lichess/events/, results/figures/,
# results/lichess/lichess_*_pilot.*); the parsed games (results/lichess/raw/)
# hold the full movetext and stay local.

#SBATCH --job-name=kvb-lichess-pilot
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=logs/lichess_pilot_%j.out

set -euo pipefail

MONTH="${MONTH:-2026-10}"
CATEGORIES="${CATEGORIES:-classical}"
MIN_RATING="${MIN_RATING:-2000}"
TAG="${TAG:-pilot}"
PGN_DIR="${PGN_DIR:-data/lichess}"

set +u
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-kvb}"
set -u

echo "month=${MONTH} categories=${CATEGORIES} min_rating=${MIN_RATING} tag=${TAG} host=$(hostname)"
python code/lichess_games.py fetch --months "${MONTH}" --categories "${CATEGORIES}" \
    --min-mean-rating "${MIN_RATING}" --tag "${TAG}" --pgn-dir "${PGN_DIR}"
python code/lichess_games.py summarise --tag "${TAG}"
python code/lichess_games.py sample --tag "${TAG}" --n 2000
python code/human_events.py run --tag "${TAG}" --workers "${SLURM_CPUS_PER_TASK:-1}"
python code/explore_human_pilot.py --tag "${TAG}"
