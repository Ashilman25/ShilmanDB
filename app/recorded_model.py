from __future__ import annotations

from pathlib import Path
from time import sleep

from app import sixd

an = sixd.an


# Every model reply a run record holds: the primary attempts, the rephrase, then its attempts.
def _replies(record: dict) -> list[dict]:
    rephrase = record["rephrase"]
    return [*record["primary"]["attempts"], rephrase, *rephrase["attempts"]]


# The model a run file's records were made with; None when no record names one.
def recorded_model_id(run_file: Path) -> str | None:
    for record in an.read_jsonl(run_file):
        if isinstance(record.get("model"), str):
            return record["model"]
    return None


# Stands in for the model by replaying a saved run's replies, found by the prompt's sha1.
# With pace, each reply takes as long as it did when it was recorded. Its id is the records' model unless given.
class RecordedModel:
    def __init__(self, run_file: Path, config_id: str | None = None, pace: bool = False):
        if not Path(run_file).is_file():
            raise FileNotFoundError(f"recorded run not found: {run_file}")

        records = an.read_jsonl(run_file)
        named = next((r["model"] for r in records if isinstance(r.get("model"), str)), None)
        self.config_id = config_id or named
        self.pace = pace
        self._replies: dict[str, tuple[str, float]] = {}
        for record in records:
            for reply in _replies(record):
                sha1 = reply["prompt_sha1"]
                if sha1 is not None and sha1 not in self._replies:
                    self._replies[sha1] = (reply["raw"], reply["seconds"])

    # The recorded text and model seconds for this exact prompt; KeyError if it was never asked.
    def generate(self, prompt: str) -> tuple[str, float]:
        found = self._replies.get(an.sha1_text(prompt))
        if found is None:
            raise KeyError("no recorded reply for this prompt")
        if self.pace:
            sleep(found[1])
        return found
