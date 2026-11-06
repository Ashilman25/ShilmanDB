#The ten fixed AI questions

from dataclasses import dataclass

@dataclass(frozen=True)
class Question:
    qid: str
    wording: str
    kind: str  # "yesno", "class" or "extract"
    options: tuple[str, ...] = ()


C2_CLASSES = (
    "shipping or delivery",
    "product quality or a defect",
    "a wrong, fake or missing item",
    "customer service or returns",
    "price or value",
    "other",
)


def _yesno(qid: str, wording: str) -> Question:
    return Question(qid, wording, "yesno", ("yes", "no"))


QUESTIONS: dict[str, Question] = {q.qid: q for q in (
    _yesno("F1", "Does the reviewer say the product stopped working or broke within a few weeks of getting it?"),
    Question("C2", "What is this review mainly about: shipping or delivery; product quality or a defect; a wrong, fake or missing item; customer service or returns; price or value; other?", "class", C2_CLASSES),
    _yesno("F3", "Does the reviewer complain that the item arrived late or that shipping was slow?"),
    _yesno("F4", "Does the reviewer say the item was counterfeit, fake, or not the genuine brand?"),
    _yesno("F5a", "Does the review talk about battery life?"),
    _yesno("F5b", "Is the reviewer positive about the product overall?"),
    _yesno("F6", "Does the reviewer say the case does not fit their phone?"),
    Question("E7", 'Which phone model does the reviewer say they use? Answer with the model name only, or "none".', "extract"),
    _yesno("F8", "Does the reviewer describe a safety hazard, such as overheating, melting, smoke, sparks, fire or a swelling battery?"),
    _yesno("F10", "Does the reviewer praise the seller's or maker's customer service?"),
)}
