#!/usr/bin/env bash
# Human-play pilot, part 2: extract the bishop/knight events from the games
# parsed by lichess_games.py and write the exploratory summary.
#
# Part 1 streams the month and needs internet, so run it on the login node
# first, as in the Chess960 project (nothing is downloaded to disk: the
# export is decompressed and filtered as it streams, and only the kept games
# are written). It reads every game's headers in the month (~100 million),
# so use screen/tmux or nohup:
#   python code/lichess_games.py fetch --months 2026-10 --categories classical \
#       --min-mean-rating 2000 --tag pilot
# (If a compute node has internet access, this job can stream it instead: it
# runs the same fetch first and skips it when the month is already parsed.)
#
# Then, from the repository root (create logs/ first):
#   sbatch jobs/lichess_pilot.slurm
#   sbatch --export=ALL,MONTH=2026-09,MIN_RATING=2200 jobs/lichess_pilot.slurm
#
# Afterwards, commit the summary, tables, figures and the small sample PGN
# (results/lichess/analysis/, results/lichess/events/, results/figures/,
# results/lichess/lichess_*_pilot.*); the parsed games (results/lichess/raw/)
# hold the full movetext and stay local.

#SBATCH --job-name=kvb-lichess-pilot
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
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
