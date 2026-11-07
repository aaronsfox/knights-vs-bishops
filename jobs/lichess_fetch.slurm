#!/usr/bin/env bash
# Stream Lichess monthly exports on compute nodes: one array task per month,
# so months stream in parallel across nodes. Each task reads its month from
# database.lichess.org, decompresses and filters it as it arrives, and writes
# only the kept games (results/lichess/raw/lichess_YYYY-MM[_TAG].parquet).
# Nothing large is written to disk. Needs internet on the compute nodes.
#
# Submit from the repository root (create logs/ first). MONTHS is a month or
# a range; the --array range is the number of months minus one:
#   sbatch --array=0 --export=ALL,MONTHS=2026-10,CATEGORIES=classical,MIN_RATING=2000,TAG=pilot \
#       jobs/lichess_fetch.slurm
#   sbatch --array=0-5 --export=ALL,MONTHS=2026-05:2026-10 jobs/lichess_fetch.slurm
# Chain the event extraction and summary after it:
#   jid=$(sbatch --parsable --array=0 --export=ALL,MONTHS=2026-10,CATEGORIES=classical,MIN_RATING=2000,TAG=pilot jobs/lichess_fetch.slurm)
#   sbatch --dependency=afterok:${jid} jobs/lichess_pilot.slurm
#
# Timing: parsing runs at roughly 300 MB of PGN text a second on one core
# (measured locally), so a ~100-million-game month is about 15-30 minutes of
# parsing; the download speed from Lichess usually sets the pace. One core is
# enough; WORKERS > 1 parses blocks of games in parallel, which only helps
# when many games are kept (e.g. all blitz). A month already parsed is
# skipped, and a task that fails (e.g. a dropped connection) can simply be
# resubmitted.

#SBATCH --job-name=kvb-lichess-fetch
#SBATCH --array=0
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/lichess_fetch_%A_%a.out

set -euo pipefail

MONTHS="${MONTHS:?set MONTHS, e.g. --export=ALL,MONTHS=2026-10}"
CATEGORIES="${CATEGORIES:-blitz,rapid,classical}"
MIN_RATING="${MIN_RATING:-0}"
SAMPLE_FRAC="${SAMPLE_FRAC:-1}"
TAG="${TAG:-}"
WORKERS="${WORKERS:-1}"

set +u
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-kvb}"
set -u

MONTH=$(python -c "import sys; sys.path.insert(0, 'code'); from lichess_games import month_range; \
m = month_range('${MONTHS}'); i = ${SLURM_ARRAY_TASK_ID:-0}; print(m[i] if i < len(m) else '')")
if [[ -z "${MONTH}" ]]; then
    echo "Task ${SLURM_ARRAY_TASK_ID:-0} has no month; exiting."
    exit 0
fi

echo "month=${MONTH} categories=${CATEGORIES} min_rating=${MIN_RATING} sample_frac=${SAMPLE_FRAC} tag=${TAG} host=$(hostname)"
python code/lichess_games.py fetch --months "${MONTH}" --categories "${CATEGORIES}" \
    --min-mean-rating "${MIN_RATING}" --sample-frac "${SAMPLE_FRAC}" \
    ${TAG:+--tag "${TAG}"} --workers "${WORKERS}"
