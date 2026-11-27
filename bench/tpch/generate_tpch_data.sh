#!/bin/bash
# Generates TPC-H data with dbgen: generate_tpch_data.sh [SF ...], default 0.01 0.1 1.0.
# Every SF also gets row_counts.csv; SF=100 is made in 16 chunks at once, then joined.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
TPCH_DATA_DIR="$PROJECT_DIR/bench/tpch_data"

TABLES="customer lineitem nation orders part partsupp region supplier"
# dbgen splits these into chunks; nation and region come whole from every process
CHUNKED_TABLES="customer lineitem orders part partsupp supplier"
SF100_CHUNKS=16


# TPC-H's row count for a table at SF=100.
tpch_sf100_rows() {
    case "$1" in
        customer) echo 15000000 ;;
        lineitem) echo 600037902 ;;
        nation) echo 25 ;;
        orders) echo 150000000 ;;
        part) echo 20000000 ;;
        partsupp) echo 80000000 ;;
        region) echo 5 ;;
        supplier) echo 1000000 ;;
        *) return 1 ;;
    esac
}


# Lines with something other than whitespace, as the Python runners count rows.
# grep exits 1 when nothing matches (a count of 0),
# and 2 when it can't read the file.
count_rows() {
    local n status=0
    n="$(LC_ALL=C grep -ac '[^[:space:]]' "$1")" || status=$?
    if [ "$status" -gt 1 ]; then
        echo "could not count the rows of $1" >&2
        return 1
    fi
    echo "${n:-0}"
}


# Writes SF_DIR/row_counts.csv.tmp (table,rows) for the 8 tables; fails if a count does.
count_tables() {
    local sf_dir="$1" table n
    {
        echo "table,rows"
        for table in $TABLES; do
            n="$(count_rows "$sf_dir/$table.tbl")" || return 1
            echo "$table,$n"
        done
    } > "$sf_dir/row_counts.csv.tmp"
}


# Appends chunks 1..N of a table, in that order, into OUT_DIR/<table>.tbl.
# Chunk i is CHUNK_ROOT/i/<table>.tbl.i; each is deleted once appended.
join_chunks() {
    local table="$1" chunks="$2" chunk_root="$3" out_dir="$4"
    local out="$out_dir/$table.tbl" chunk i
    : > "$out" || return 1
    for ((i = 1; i <= chunks; i++)); do
        chunk="$chunk_root/$i/$table.tbl.$i"
        if [ ! -f "$chunk" ]; then
            echo "missing chunk $chunk" >&2
            return 1
        fi
        cat "$chunk" >> "$out" || return 1
        rm -f "$chunk"
    done
}


# True when a counts file matches TPC-H's SF=100 counts; else prints both per table.
check_sf100_counts() {
    local counts_file="$1" table rows expected report="" ok=0
    while IFS=, read -r table rows; do
        [ "$table" = "table" ] && continue
        expected="$(tpch_sf100_rows "$table")"
        [ "$rows" = "$expected" ] || ok=1
        report="$report  $table: expected $expected, got $rows"$'\n'
    done < "$counts_file"
    if [ "$ok" -ne 0 ]; then
        echo "SF=100: the row counts don't match TPC-H's:" >&2
        printf '%s' "$report" >&2
    fi
    return "$ok"
}


# Stops any running dbgen, deletes everything a failed SF=100 run made, then exits 1.
discard_sf100() {
    local sf_dir="$1" running="$1/running_jobs" pid
    trap - EXIT INT TERM
    # Only the jobs bash still sees running: a chunk it already reaped may have a
    # reused pid. Written to a file, since jobs in a subshell may list nothing.
    if jobs -pr > "$running" 2>/dev/null; then
        while read -r pid; do
            kill "$pid" 2>/dev/null || true
        done < "$running"
    fi
    wait 2>/dev/null || true
    rm -rf "$sf_dir/chunks"
    rm -f "$sf_dir"/*.tbl "$sf_dir/row_counts.csv" "$sf_dir/row_counts.csv.tmp" "$running"
    echo "SF=100: kept nothing in $sf_dir" >&2
    exit 1
}


# One dbgen run for the whole SF, then the files moved, checked and counted.
generate_whole() {
    local sf="$1" sf_dir="$TPCH_DATA_DIR/sf$1" table tbl
    mkdir -p "$sf_dir"
    rm -f "$sf_dir/row_counts.csv"
    ./dbgen -s "$sf" -f
    mv *.tbl "$sf_dir/"
    for table in $TABLES; do
        tbl="$sf_dir/$table.tbl"
        if [ ! -r "$tbl" ]; then
            echo "SF=$sf: $table.tbl is missing or unreadable:" >&2
            ls -l "$tbl" >&2 || true
            exit 1
        fi
    done
    if ! count_tables "$sf_dir"; then
        rm -f "$sf_dir/row_counts.csv.tmp"
        echo "SF=$sf: wrote no row_counts.csv" >&2
        exit 1
    fi
    mv "$sf_dir/row_counts.csv.tmp" "$sf_dir/row_counts.csv"
    echo "Generated SF=$sf data in $sf_dir"
}


# 16 dbgen processes, each writing its chunk into its own folder, then the
# chunks joined in order and the counts checked against TPC-H's.
generate_sf100() {
    local sf="$1" sf_dir="$TPCH_DATA_DIR/sf$1"
    local chunk_root="$sf_dir/chunks" pids="" pid failed=0 i table
    mkdir -p "$sf_dir"
    rm -f "$sf_dir/row_counts.csv" "$sf_dir/row_counts.csv.tmp"
    rm -rf "$chunk_root"

    # Any stop from here on keeps nothing: an error exit, Ctrl-C (which the
    # background dbgen processes ignore) or TERM. Cleared once the data is whole.
    trap "discard_sf100 $(printf '%q' "$sf_dir")" EXIT INT TERM
    for ((i = 1; i <= SF100_CHUNKS; i++)); do
        mkdir -p "$chunk_root/$i"
        DSS_PATH="$chunk_root/$i" ./dbgen -s "$sf" -C "$SF100_CHUNKS" -S "$i" -f &
        pids="$pids $!"
    done
    i=0
    for pid in $pids; do
        i=$((i + 1))
        if ! wait "$pid"; then
            echo "SF=$sf: dbgen chunk $i of $SF100_CHUNKS failed" >&2
            failed=1
        fi
    done
    [ "$failed" -eq 0 ] || discard_sf100 "$sf_dir"

    for table in nation region; do
        mv "$chunk_root/1/$table.tbl" "$sf_dir/$table.tbl" || discard_sf100 "$sf_dir"
    done
    for table in $CHUNKED_TABLES; do
        join_chunks "$table" "$SF100_CHUNKS" "$chunk_root" "$sf_dir" || discard_sf100 "$sf_dir"
    done
    rm -rf "$chunk_root"

    count_tables "$sf_dir" || discard_sf100 "$sf_dir"
    check_sf100_counts "$sf_dir/row_counts.csv.tmp" || discard_sf100 "$sf_dir"
    mv "$sf_dir/row_counts.csv.tmp" "$sf_dir/row_counts.csv" || discard_sf100 "$sf_dir"
    trap - EXIT INT TERM
    echo "Generated SF=$sf data in $sf_dir"
}


main() {
    if [ $# -eq 0 ]; then
        set -- 0.01 0.1 1.0
    fi

    # Check every scale factor before cloning or building anything
    local SF
    for SF in "$@"; do
        case "$SF" in
            0.01|0.1|1.0|10|100) ;;
            *) echo "unknown scale factor '$SF' (choose from 0.01 0.1 1.0 10 100)" >&2; exit 1 ;;
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
        if [ "$SF" = "100" ]; then
            generate_sf100 "$SF"
        else
            generate_whole "$SF"
        fi
    done

    echo "Done. TPC-H data in $TPCH_DATA_DIR"
}


# Run only when executed, so a test can source the functions
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    main "$@"
fi
