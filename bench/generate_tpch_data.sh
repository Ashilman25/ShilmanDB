#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TPCH_DATA_DIR="$PROJECT_DIR/bench/tpch_data"


DBGEN_DIR="/tmp/tpch-dbgen"
if [ ! -d "$DBGEN_DIR" ]; then
    git clone https://github.com/electrum/tpch-dbgen.git "$DBGEN_DIR"
fi

cd "$DBGEN_DIR"


sed -i '' 's/^CC *=.*/CC = cc/' makefile
sed -i '' 's/^DATABASE *=.*/DATABASE = NONE/' makefile
sed -i '' 's/^MACHINE *=.*/MACHINE = LINUX/' makefile
sed -i '' 's/^WORKLOAD *=.*/WORKLOAD = TPCH/' makefile
sed -i '' 's/#include <malloc.h>/#include <stdlib.h>/' bm_utils.c
make clean && make dbgen


for SF in 0.01 0.1 1.0; do
    SF_DIR="$TPCH_DATA_DIR/sf${SF}"
    mkdir -p "$SF_DIR"
    ./dbgen -s "${SF}" -f
    mv *.tbl "$SF_DIR/"
    echo "Generated SF=${SF} data in $SF_DIR"
done

echo "Done. TPC-H data in $TPCH_DATA_DIR"