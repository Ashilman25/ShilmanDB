#!/bin/bash
# Fails when any test was skipped, except the known ones below: ci_check_skips.sh <ctest-log>.
# A data test that skips (no bench/tpch_data/sf0.01/) would otherwise let CI pass without checking answers.
set -euo pipefail

# Skips that are fine: direct I/O is macOS only, and CI has no exported V2 weights.
ALLOWED_SKIPS="DiskManagerTest.DirectIORoundTrip EvictionSimTest.V2ExportedWeightsMatchTorch"

if [ $# -ne 1 ] || [ ! -f "$1" ]; then
    echo "usage: ci_check_skips.sh <ctest-log>" >&2
    exit 2
fi

# The test lines under ctest's "did not run" header whose test name isn't allowed
skipped=$(awk -v allowed_list="$ALLOWED_SKIPS" '
    BEGIN { n = split(allowed_list, names, " "); for (i = 1; i <= n; i++) allowed[names[i]] = 1 }
    /^The following tests did not run:/ { in_list = 1; next }
    in_list && /^[[:space:]]+[0-9]+ - / {
        name = $0
        sub(/^[[:space:]]+[0-9]+ - /, "", name)
        sub(/[[:space:]]+\([^)]*\)[[:space:]]*$/, "", name)
        if (!(name in allowed))
            print
        next
    }
    { in_list = 0 }
' "$1")

if [ -n "$skipped" ]; then
    echo "Tests did not run (a TPC-H one? is bench/tpch_data/sf0.01/ there?):"
    echo "$skipped"
    exit 1
fi
echo "No test was skipped except the known ones."
