"""A stand-in for analyst_bridge that speaks its protocol, for the client tests.

It ignores its arguments and echoes each request's SQL back as a one-cell result.
SQL "SLEEP" sleeps 5 s, "DIE" exits 1, and "WRONGID" answers with the wrong id.
"HOG" holds about 300 MB for 2 s, then answers.
"""
from __future__ import annotations

import json
import sys
import time

HOG_BYTES = 300 << 20
HOG_SECONDS = 2.0


# Print one JSON line and flush it, as the real bridge does.
def _send(record: dict) -> None:
    print(json.dumps(record), flush=True)


# Answer requests until stdin closes.
def main() -> None:
    _send({"ready": True, "tables": 0, "load_ms": 0, "rows": {}})
    for raw in sys.stdin:
        rid, _, sql = raw.rstrip("\r\n").partition("\t")
        if sql == "SLEEP":
            time.sleep(5)
        elif sql == "DIE":
            sys.exit(1)
        elif sql == "WRONGID":
            rid = "wrong"
        elif sql == "HOG":
            hog = b"x" * HOG_BYTES  # filled, so every page counts toward RSS
            time.sleep(HOG_SECONDS)
            del hog
        _send({"id": rid, "ok": True, "columns": [{"name": "sql", "type": "VARCHAR"}], "rows": [[sql]],
               "row_count": 1, "truncated": False, "profile": {}, "ms": 0})


if __name__ == "__main__":
    main()
