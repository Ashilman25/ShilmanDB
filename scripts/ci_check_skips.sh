#!/bin/bash
# Fails when a TPC-H answer test was skipped: ci_check_skips.sh <ctest-log>.
# Those tests skip without bench/tpch_data/sf0.01/, which would let CI pass without checking answers.
set -euo pipefail

if [ $# -ne 1 ] || [ ! -f "$1" ]; then
    echo "usage: ci_check_skips.sh <ctest-log>" >&2
    exit 2
fi

# The test lines under ctest's "did not run" header whose names match a TPC-H suite
skipped=$(awk '
    /^The following tests did not run:/ { in_list = 1; next }
    in_list && /^[[:space:]]+[0-9]+ - / {
        if ($0 ~ /SqliteComparison|EvictionPolicyTpch|TpchEstimates|TpchStreams|JoinOrderParity|VectorizedTpchParity/)
            print
        next
    }
    { in_list = 0 }
' "$1")

if [ -n "$skipped" ]; then
    echo "TPC-H answer tests did not run (is bench/tpch_data/sf0.01/ there?):"
    echo "$skipped"
    exit 1
fi
echo "No TPC-H answer test was skipped."
