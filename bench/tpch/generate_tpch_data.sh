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
make clean && make dbgen


for SF in "$@"; do
    SF_DIR="$TPCH_DATA_DIR/sf${SF}"
    mkdir -p "$SF_DIR"
    ./dbgen -s "${SF}" -f
    mv *.tbl "$SF_DIR/"
    echo "Generated SF=${SF} data in $SF_DIR"
done

echo "Done. TPC-H data in $TPCH_DATA_DIR"
