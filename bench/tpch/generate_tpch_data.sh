#!/bin/bash
# Generates TPC-H data with dbgen: generate_tpch_data.sh [SF ...], default 0.01 0.1 1.0.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
TPCH_DATA_DIR="$PROJECT_DIR/bench/tpch_data"

if [ $# -eq 0 ]; then
    set -- 0.01 0.1 1.0
fi

# Check every scale factor before cloning or building anything
for SF in "$@"; do
    case "$SF" in
        0.01|0.1|1.0|10) ;;
        *) echo "unknown scale factor '$SF' (choose from 0.01 0.1 1.0 10)" >&2; exit 1 ;;
    esac
done


DBGEN_DIR="${DBGEN_DIR:-/tmp/tpch-dbgen}"
if [ ! -d "$DBGEN_DIR" ]; then
    git clone https://github.com/electrum/tpch-dbgen.git "$DBGEN_DIR"
fi

cd "$DBGEN_DIR"


# perl -pi edits in place the same way on macOS and Linux
perl -pi -e 's/^CC *=.*/CC = cc/' makefile
perl -pi -e 's/^DATABASE *=.*/DATABASE = NONE/' makefile
perl -pi -e 's/^MACHINE *=.*/MACHINE = LINUX/' makefile
perl -pi -e 's/^WORKLOAD *=.*/WORKLOAD = TPCH/' makefile
perl -pi -e 's/#include <malloc.h>/#include <stdlib.h>/' bm_utils.c
# dbgen reads st_mode after a failed stat; zeroed, a new file gets fopen's normal mode
perl -pi -e 's/struct stat fstats;/struct stat fstats = {0};/' bm_utils.c
grep -q 'struct stat fstats = {0};' bm_utils.c || { echo "could not patch dbgen's bm_utils.c (fstats)" >&2; exit 1; }
make clean && make dbgen


for SF in "$@"; do
    SF_DIR="$TPCH_DATA_DIR/sf${SF}"
    mkdir -p "$SF_DIR"
    ./dbgen -s "${SF}" -f
    mv *.tbl "$SF_DIR/"
    for TABLE in customer lineitem nation orders part partsupp region supplier; do
        TBL="$SF_DIR/$TABLE.tbl"
        if [ ! -r "$TBL" ]; then
            echo "SF=${SF}: $TABLE.tbl is missing or unreadable:" >&2
            ls -l "$TBL" >&2 || true
            exit 1
        fi
    done
    echo "Generated SF=${SF} data in $SF_DIR"
done

echo "Done. TPC-H data in $TPCH_DATA_DIR"
