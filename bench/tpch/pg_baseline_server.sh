#!/usr/bin/env bash
# The Postgres server for the TPC-H baseline: its own data folder, port 5434.
# Usage: bash bench/tpch/pg_baseline_server.sh {init|start|stop|status}
# PG_BIN and PG_PORT can be set to override the defaults; the data folder is always this repo's.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PG_BIN="${PG_BIN:-/opt/homebrew/opt/postgresql@17/bin}"
PG_PORT="${PG_PORT:-5434}"
PG_DATA="$REPO_ROOT/bench/results/tpch_postgres/pgdata"
PG_LOG="$(dirname "$PG_DATA")/server.log"

usage() {
    echo "Usage: bash bench/tpch/pg_baseline_server.sh {init|start|stop|status}" >&2
    exit 1
}

[[ $# -eq 1 ]] || usage

# 5433 is the vector-search server; this script never touches it.
if [[ "$PG_PORT" == "5433" ]]; then
    echo "Error: port 5433 is the vector-search server. The TPC-H baseline runs on 5434." >&2
    exit 1
fi

# Every port the folder's server is set to or was started on, one per line:
# the conf files' last port line, the -p in postmaster.opts, line 4 of postmaster.pid.
folder_ports() {
    local file
    for file in "$PG_DATA/postgresql.conf" "$PG_DATA/postgresql.auto.conf"; do
        if [[ -f "$file" ]]; then
            sed -nE "s/^[[:space:]]*port([[:space:]]*=[[:space:]]*|[[:space:]]+)'?([0-9]+).*/\2/p" "$file" | tail -n 1
        fi
    done
    if [[ -f "$PG_DATA/postmaster.opts" ]]; then
        grep -oE '"-p" "[0-9]+"|"-c" "port=[0-9]+"|"--port=[0-9]+"' "$PG_DATA/postmaster.opts" | grep -oE '[0-9]+' || true
    fi
    if [[ -f "$PG_DATA/postmaster.pid" ]]; then
        sed -n 4p "$PG_DATA/postmaster.pid"
    fi
}

# start, stop and status act on whatever server owns the folder, so check its ports too.
if [[ "$1" != "init" ]]; then
    ports="$(folder_ports)"
    if grep -qxE '[[:space:]]*5433[[:space:]]*' <<< "$ports"; then
        echo "Error: the server of $PG_DATA is set to or runs on port 5433, the vector-search server. This script won't touch it." >&2
        exit 1
    fi
fi

case "$1" in
    init)
        if [[ -e "$PG_DATA" ]]; then
            echo "Error: $PG_DATA already exists. Remove it first to start over." >&2
            exit 1
        fi
        # Build the settings first, so a failure leaves no half-made folder.
        conf="$("$REPO_ROOT/.venv/bin/python" "$REPO_ROOT/bench/tpch/run_postgres_benchmarks.py" --print-conf --port "$PG_PORT")"
        mkdir -p "$(dirname "$PG_DATA")"
        "$PG_BIN/initdb" -D "$PG_DATA" -E UTF8 --no-locale
        echo "$conf" >> "$PG_DATA/postgresql.conf"
        echo "Initialized $PG_DATA for port $PG_PORT. Start it with: bash bench/tpch/pg_baseline_server.sh start"
        ;;
    start)
        "$PG_BIN/pg_ctl" -D "$PG_DATA" -l "$PG_LOG" -w start
        ;;
    stop)
        "$PG_BIN/pg_ctl" -D "$PG_DATA" -l "$PG_LOG" -w stop
        ;;
    status)
        "$PG_BIN/pg_ctl" -D "$PG_DATA" status
        ;;
    *)
        usage
        ;;
esac
