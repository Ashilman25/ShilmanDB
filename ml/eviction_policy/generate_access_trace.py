#!/usr/bin/env python3

import argparse
import os
import subprocess
import sys
import tempfile


def count_data_lines(path: str) -> int:
    with open(path) as f:
        return sum(1 for _ in f) - 1


def run_load_tpch(binary: str, sf: str, data_dir: str, db_file: str, trace_path: str, pool_size: int) -> int:
    cmd = [
        binary,
        "--sf", sf,
        "--db-file", db_file,
        "--data-dir", data_dir,
        "--pool-size", str(pool_size),
        "--enable-tracing", trace_path,
        "--no-verify",
    ]
    print(f"  Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.stdout:
        for line in result.stdout.strip().split("\n"):
            print(f"    {line}")

    if result.returncode != 0:
        print(f"ERROR: load_tpch exited with code {result.returncode}", file=sys.stderr)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        sys.exit(1)

    return count_data_lines(trace_path)


def concatenate_traces(trace_paths: list[str], output: str) -> int:
    counter = 0
    with open(output, "w") as out_f:
        out_f.write("timestamp,page_id\n")
        for path in trace_paths:
            with open(path) as in_f:
                next(in_f)  # skip header
                for line in in_f:
                    parts = line.strip().split(",")
                    if len(parts) == 2:
                        out_f.write(f"{counter},{parts[1]}\n")
                        counter += 1
    return counter


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate page access traces from TPC-H queries")
    parser.add_argument("--binary", default="build/bench/load_tpch",help="Path to load_tpch binary (default: build/bench/load_tpch)")
    parser.add_argument("--data-dir", required=True, help="TPC-H data directory (e.g., bench/tpch_data/sf0.01/)")
    parser.add_argument("--sf", default="0.01", help="Scale factor label (default: 0.01)")
    parser.add_argument("--output", default="bench/tpch_data/access_trace.csv", help="Output trace CSV path")
    parser.add_argument("--min-lines", type=int, default=100_000, help="Minimum number of trace entries required (default: 100000)")
    args = parser.parse_args()

    print("=== Access Trace Generator ===\n")

    print("Building load_tpch ...")
    build_result = subprocess.run(
        ["cmake", "--build", "build", "--target", "load_tpch"],
        capture_output=True, text=True,
    )
    if build_result.returncode != 0:
        print("ERROR: Build failed", file=sys.stderr)
        if build_result.stderr:
            print(build_result.stderr, file=sys.stderr)
        sys.exit(1)
        
    print("  Build OK\n")


    pool_sizes = [256, 128, 64]

    for pool_size in pool_sizes:
        fd, db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        try:
            print(f"Attempt: pool_size={pool_size} frames ...")
            lines = run_load_tpch(args.binary, args.sf, args.data_dir, db_path, args.output, pool_size)
            print(f"  Trace entries: {lines}\n")

            if lines >= args.min_lines:
                print(f"SUCCESS: {lines} >= {args.min_lines} entries")
                print(f"Output: {args.output}")
                return

            print(f"  {lines} < {args.min_lines}, trying smaller pool ...\n")
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)

    # Fallback: concatenate multiple runs with the smallest pool size
    print(f"Single run insufficient. Concatenating multiple runs ...\n")

    all_traces: list[str] = []
    total_lines = 0
    run_idx = 0
    max_runs = 20

    try:
        while total_lines < args.min_lines and run_idx < max_runs:
            fd, db_path = tempfile.mkstemp(suffix=".db")
            os.close(fd)
            fd2, tmp_trace = tempfile.mkstemp(suffix=".csv")
            os.close(fd2)
            
            try:
                lines = run_load_tpch(args.binary, args.sf, args.data_dir, db_path, tmp_trace, pool_sizes[-1])
                total_lines += lines
                all_traces.append(tmp_trace)
                
            finally:
                if os.path.exists(db_path):
                    os.unlink(db_path)

            run_idx += 1
            print(f"  Run {run_idx}: +{lines} lines (total: {total_lines})\n")

        final_count = concatenate_traces(all_traces, args.output)
        print(f"Concatenated {run_idx} runs: {final_count} total entries")

        if final_count >= args.min_lines:
            print(f"SUCCESS: {final_count} >= {args.min_lines} entries")
        else:
            print(
                f"WARNING: Only {final_count} entries after {max_runs} runs "
                f"(target: {args.min_lines})"
            )

        print(f"Output: {args.output}")

    finally:
        for path in all_traces:
            if os.path.exists(path):
                os.unlink(path)


if __name__ == "__main__":
    main()
