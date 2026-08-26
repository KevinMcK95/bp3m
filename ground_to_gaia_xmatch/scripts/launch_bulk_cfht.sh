#!/bin/bash
# Launch N detached bulk workers over the CFHT/UNIONS archive.
#
#   bash launch_bulk_cfht.sh [n_workers]      # default 4
#
# Notes on the launch mechanism, learned the hard way:
#   * `conda run` CAPTURES stdout by default and only emits it on exit, so a
#     long-running worker's log stays empty.  --no-capture-output is required.
#   * PYTHONPATH rather than `cd`: a `cd` here was silently killing the script.
#   * setsid + </dev/null fully detaches, so the workers outlive the shell (and
#     any agent-tool timeout) and keep running until the archive is done.
#
# Partitioning is deterministic (index %% n_workers), so workers never collide.
# Resume is automatic via .complete sentinels — safe to re-run this script at
# any time to restart dead workers.
set -u

REPO=/bootes_raid6/users/kmckinnon/claude/bp3m
FIELD_ROOT=/home/jupyter-kmckinnon/data_bootes/bp3m/CFHT/UNIONS
LOGDIR="$FIELD_ROOT/bulk_logs"
NW=${1:-4}

mkdir -p "$LOGDIR"

for i in $(seq 0 $((NW - 1))); do
    LOG="$LOGDIR/worker_$(printf '%03d' "$i").log"
    PYTHONPATH="$REPO" setsid nohup conda run --no-capture-output -n bp3m-test \
        python -u -m ground_to_gaia_xmatch.scripts.run_bulk \
        --instrument cfht --field-root "$FIELD_ROOT" \
        --n-workers "$NW" --worker-id "$i" --no-plots \
        >> "$LOG" 2>&1 < /dev/null &
    disown
    echo "worker $i -> $LOG"
done
