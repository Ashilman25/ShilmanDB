#Counting oracle: the only way an approach may read a model's answers.
import numpy as np

from .labels import TierLabels


class Oracle:
    def __init__(self, tier: str, labels: TierLabels):
        self.tier = tier
        self.__labels = labels
        self.__asked: set[tuple[str, str]] = set()
        self.calls = 0
        self.log: list[tuple[str, str]] = []

    def __charge(self, qid: str, key: str) -> None:
        if (qid, key) not in self.__asked:
            self.__asked.add((qid, key))
            self.log.append((qid, key))
            self.calls += 1

    def __lookup(self, table, qid: str, keys, what: str) -> dict:
        found = {}
        for key in keys:
            try:
                found[key] = table[qid][key]
            except KeyError:
                raise KeyError(f"no {what} for question {qid!r}, text key {key!r}") from None
        return found

    def ask(self, qid: str, keys) -> dict[str, str]:
        found = self.__lookup(self.__labels.answers, qid, keys, "answer")
        for key in found:
            self.__charge(qid, key)
            
        return found

    def probs(self, qid: str, keys) -> dict[str, np.ndarray]:
        found = self.__lookup(self.__labels.probs, qid, keys, "probabilities")
        for key in found:
            self.__charge(qid, key)
            
        return {key: p.copy() for key, p in found.items()}  # copies: callers can't change shared labels
