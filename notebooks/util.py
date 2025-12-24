from typing import Callable, Literal

import dspy
import gqr
import ollama

system_prompt = (
    "You are a highly accurate text classifier. Your task is to categorize passages "
    "into one of four predefined domains. The ONLY valid categories are: "
    + ", ".join(gqr.label2domain.values())
    + ". Any passage that does not clearly belong to the domains above MUST be "
    "categorized as ood. You must respond with ONLY the category name, and nothing "
    "else. No explanations, no extra words."
)

user_prompt = (
    "Classify the following passage into one of the categories: "
    + ", ".join(gqr.label2domain.values())
    + ".\nPassage:\n{query}\nCategory:"
)


class Classify(dspy.Signature):
    """You are a highly accurate text classifier. Your task is to categorize passages into one of four predefined domains. The ONLY valid categories are: 'law', 'finance', 'healthcare', 'ood'. Any passage that does not clearly belong to the domains above MUST be categorized as ood. You must respond with ONLY the category name, and nothing else. No explanations, no extra words."""

    query: str = dspy.InputField()
    route: Literal["law", "finance", "healthcare", "ood"] = dspy.OutputField()


def normalize_route(value: str | None) -> str:
    if value in gqr.domain2label:
        return value
    return "ood"


class SafePredict(dspy.Module):
    def __init__(self, signature):
        super().__init__()
        self.predict = dspy.Predict(signature)

    def forward(self, **kwargs):
        try:
            return self.predict(**kwargs)
        except Exception:
            return dspy.Prediction(route="ood")


def metric(gold: dspy.Example, pred: dspy.Prediction, trace=None, pred_name=None, pred_trace=None) -> bool:
    pred_route = normalize_route(getattr(pred, "route", None))
    gold_route = normalize_route(getattr(gold, "route", None))
    try:
        pred_label = gqr.domain2label[pred_route]
        gold_label = gqr.domain2label[gold_route]
    except Exception:
        return False
    return gold_label == pred_label


def score_dspy(text: str, program: Callable[..., dspy.Prediction]) -> int:
    try:
        pred = program(query=text)
        pred_route = normalize_route(getattr(pred, "route", None))
        predicted_label = gqr.domain2label[pred_route]
    except Exception:
        predicted_label = gqr.domain2label.get("ood", 3)
    return predicted_label


def score_program(text: str, program: Callable[..., dspy.Prediction]) -> int:
    return score_dspy(text, program=program)


def make_dspy_scorer() -> Callable[[str], int]:
    program = SafePredict(Classify)
    return lambda text: score_dspy(text, program=program)

def score_ollama(text: str, model_name: str) -> int:
    try:
        formatted_user_prompt = user_prompt.format(query=text)
        response = ollama.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": formatted_user_prompt},
            ],
            think="low" if model_name.startswith("gpt-oss") else False,
            stream=False,
        )
        result = response["message"]["content"].strip().lower()
        if result in gqr.domain2label:
            predicted_label = gqr.domain2label[result]
        else:
            predicted_label = gqr.domain2label.get("ood", 3)
    except Exception:
        predicted_label = gqr.domain2label.get("ood", 3)
    return predicted_label


def build_examples(data, label_map=None):
    mapping = label_map or gqr.label2domain
    examples = []
    for text, label in zip(data["text"].values, data["label"].values):
        route = mapping[label]
        ex = dspy.Example(query=text, route=route)
        try:
            ex = ex.with_inputs("query")
        except Exception:
            pass
        examples.append(ex)
    return examples
