import argparse
import gc
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Literal

import dspy
import gqr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from gqr.core.evaluator import Evaluator, evaluate, evaluate_by_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
logging.getLogger("dspy.teleprompt.gepa.gepa").setLevel(logging.WARNING)
logging.getLogger("dspy.optimizers.bootstrap_fewshot").setLevel(logging.WARNING)

# -----------------------------------------------------------------------------
# config

MODELS = [
    "google/gemma-4-E2B-it",
    "Qwen/Qwen3.5-9B",
    "microsoft/phi-4",
    "ibm-granite/granite-4.0-tiny-preview",
    "mistralai/Mistral-7B-Instruct-v0.3",
]

STRATEGIES = ["baseline"] # "baseline", "dspy", "gepa", "fewshot"

MAX_NEW_TOKENS_BASELINE = 3
MAX_NEW_TOKENS_DSPY     = 50
MAX_INPUT_LEN   = 2048
GEPA_TRAIN_SIZE = 5000
GEPA_VAL_SIZE   = 1000
GEPA_AUTO       = "light"
GEPA_SEED       = 22
FS_TRAIN_SIZE   = 300
FS_MAX_DEMOS    = 6
FS_CANDIDATES   = 4

MODEL_CACHE = Path.home() / "models" / "raw"
RESULTS_DIR = Path("results")
SAVES_DIR   = Path("saves")

CATEGORIES = list(gqr.label2domain.values())  # law, finance, healthcare, ood

SYSTEM_PROMPT = (
    "You are a highly accurate text classifier. Your task is to categorize passages "
    "into one of four predefined domains. The ONLY valid categories are: "
    + ", ".join(CATEGORIES) +
    ". Any passage that does not clearly belong to the domains above MUST be "
    "categorized as ood. You must respond with ONLY the category name, and nothing "
    "else. No explanations, no extra words."
)

USER_PROMPT = (
    "Classify the following passage into one of the categories: "
    + ", ".join(CATEGORIES) + ".\nPassage:\n{query}\nCategory:"
)

COLORS = {"baseline": "#4c78a8", "dspy": "#e45756", "gepa": "#54a24b", "fewshot": "#72b7b2"}

# -----------------------------------------------------------------------------
# dspy plumbing

class Classify(dspy.Signature):
    """Classify a passage into exactly one of: law, finance, healthcare, ood."""
    query: str = dspy.InputField()
    route: Literal["law", "finance", "healthcare", "ood"] = dspy.OutputField()

class SafePredict(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(Classify)
    def forward(self, **kw):
        try:
            result = self.predict(**kw)
            #log.info("SafePredict OK: route=%r", result.route)
            return result
        except Exception as e:
            #log.info("SafePredict FAIL: %s", e)
            return dspy.Prediction(route="ood")

class ScoreWithFeedback(float):
    """GEPA needs a float that also carries a .feedback string."""
    def __new__(cls, val, feedback):
        obj = float.__new__(cls, val)
        obj.feedback = feedback
        return obj

def dspy_metric(gold, pred, trace=None):
    try:
        result = gqr.domain2label[pred.route] == gqr.domain2label[gold.route]
        log.info("METRIC gold=%r pred=%r → %s", gold.route, pred.route, result)
        return result
    except Exception as e:
        log.info("METRIC gold=%r pred=%r → FAIL: %s", getattr(gold, 'route', None), getattr(pred, 'route', None), e)
        return False

def gepa_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    correct = dspy_metric(gold, pred)
    if correct:
        return ScoreWithFeedback(1.0, "Correct.")
    return ScoreWithFeedback(0.0,
        f"Incorrect: predicted '{getattr(pred, 'route', None)}', expected '{gold.route}'.")

def predict_dspy(text, program):
    try:
        route = program(query=text).route
        #log.info("PREDICT route=%r", route)
        return gqr.domain2label[route]
    except Exception as e:
        #log.info("PREDICT FAIL: %s", e)
        return gqr.domain2label["ood"]

def build_examples(data):
    return [
        dspy.Example(query=t, route=gqr.label2domain[l]).with_inputs("query")
        for t, l in zip(data["text"].values, data["label"].values)
    ]

# -----------------------------------------------------------------------------
# model loading

def _cache(name):
    return MODEL_CACHE / name.replace("/", "_")

def load_model(name):
    log.info("Loading %s", name)
    cd = _cache(name)
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True, cache_dir=cd)
    mdl = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True, cache_dir=cd)
    mdl.eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return mdl, tok

def free_gpu():
    gc.collect()
    torch.cuda.empty_cache()

class HFLocalLM(dspy.LM):
    """Wrap an already-loaded HF model+tokenizer so DSPy uses it directly."""

    def __init__(self, name, hf_model, hf_tok):
        self.model = name
        self.model_type = "chat"
        self.cache = False
        self.callbacks = []
        self.history = []
        self.kwargs = {"temperature": 0.0, "max_tokens": MAX_NEW_TOKENS_DSPY}
        self.num_retries = 0
        self.provider = None
        self.finetuning_model = None
        self.launch_kwargs = {}
        self.train_kwargs = {}
        self.use_developer_role = False
        self._warned_zero_temp_rollout = False
        self._hf_model = hf_model
        self._hf_tok = hf_tok

    def forward(self, prompt=None, messages=None, **kwargs):
        from types import SimpleNamespace
        messages = messages or [{"role": "user", "content": prompt}]
        if hasattr(self._hf_tok, "apply_chat_template"):
            text = self._hf_tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
        else:
            text = "\n".join(m["content"] for m in messages)
        ids = self._hf_tok(text, return_tensors="pt", truncation=True,
                           max_length=MAX_INPUT_LEN).to(self._hf_model.device)
        try:
            with torch.no_grad():
                out = self._hf_model.generate(
                    **ids, max_new_tokens=MAX_NEW_TOKENS_DSPY, do_sample=False,
                    pad_token_id=self._hf_tok.pad_token_id)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            out = ids["input_ids"]  # empty generation
        ans = self._hf_tok.decode(
            out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        n_prompt = ids["input_ids"].shape[1]
        n_completion = out.shape[1] - n_prompt
        choice = SimpleNamespace(
            message=SimpleNamespace(content=ans, role="assistant", tool_calls=None),
            finish_reason="stop")
        usage = {"prompt_tokens": n_prompt, "completion_tokens": n_completion,
                 "total_tokens": n_prompt + n_completion}
        return SimpleNamespace(choices=[choice], model=self.model, usage=usage)

    def launch(self): pass
    def kill(self): pass

def make_dspy_lm(name, hf_model, hf_tok):
    return HFLocalLM(name, hf_model, hf_tok)

# -----------------------------------------------------------------------------
# scoring: returns (score_fn, latencies_list)

def scorer_baseline(model, tok):
    lats = []
    def fn(text):
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": USER_PROMPT.format(query=text)}]
        if hasattr(tok, "apply_chat_template"):
            prompt = tok.apply_chat_template(msgs, tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
        else:
            prompt = f"{SYSTEM_PROMPT}\n\n{USER_PROMPT.format(query=text)}"
        ids = tok(prompt, return_tensors="pt", truncation=True, max_length=MAX_INPUT_LEN).to(model.device)
        t0 = time.perf_counter()
        try:
            with torch.no_grad():
                out = model.generate(**ids, max_new_tokens=MAX_NEW_TOKENS_BASELINE,
                                     do_sample=False, pad_token_id=tok.pad_token_id)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            lats.append(time.perf_counter() - t0)
            return gqr.domain2label["ood"]
        lats.append(time.perf_counter() - t0)
        raw = tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip().lower()
        return gqr.domain2label.get(raw, gqr.domain2label["ood"])
    return fn, lats

def scorer_dspy(program=None):
    if program is None:
        program = SafePredict()
    lats = []
    def fn(text):
        t0 = time.perf_counter()
        label = predict_dspy(text, program)
        lats.append(time.perf_counter() - t0)
        return label
    return fn, lats

# -----------------------------------------------------------------------------
# evaluation

DRY_RUN = False  # set via --dry-run flag
DRY_RUN_N = 5    # samples per split in dry-run mode

def eval_model(name, strategy, score_fn, lats):
    """Run score_fn on ID + OOD splits, return a results dict."""
    id_data  = gqr.load_id_test_dataset()
    ood_data = gqr.load_ood_test_dataset()
    if DRY_RUN:
        id_data  = id_data.head(DRY_RUN_N).copy()
        ood_data = ood_data.head(DRY_RUN_N).copy()
    n = len(id_data) + len(ood_data)

    preds = []
    texts = list(id_data["text"].values) + list(ood_data["text"].values)
    for t in tqdm(texts, desc=f"{name} [{strategy}]"):
        preds.append(score_fn(t))
    id_data["pred"]  = preds[:len(id_data)]
    ood_data["pred"] = preds[len(id_data):]

    id_acc = evaluate(predictions=id_data["pred"], ground_truth=id_data["label"])["accuracy"]

    ood_by_ds = evaluate_by_dataset(ood_data, pred_col="pred", true_col="label", dataset_col="dataset")
    if ood_by_ds.empty:
        ood_acc = Evaluator.evaluate(predicted_labels=ood_data["pred"], true_labels=ood_data["label"])["accuracy"]
        ds_acc = {}
    else:
        ood_acc = ood_by_ds["accuracy"].mean()
        ds_acc = dict(zip(ood_by_ds["dataset"], ood_by_ds["accuracy"]))

    gqr_score = 2*id_acc*ood_acc/(id_acc+ood_acc) if (id_acc+ood_acc) > 0 else 0.0
    avg_lat = sum(lats)/len(lats) if lats else 0.0

    log.info("%-8s ID:%.4f OOD:%.4f GQR:%.4f Lat:%.3fs", strategy, id_acc, ood_acc, gqr_score, avg_lat)
    return dict(model=name, strategy=strategy,
                avg_latency=round(avg_lat,4), id_acc=round(id_acc,4),
                ood_acc=round(ood_acc,4), gqr_score=round(gqr_score,4),
                dataset_acc=str(ds_acc))

# -----------------------------------------------------------------------------
# plots

def plot_bars(df, path):
    pivot = df.pivot_table(index="model", columns="strategy", values="gqr_score", aggfunc="mean")
    pivot = pivot.reindex(columns=[s for s in STRATEGIES if s in pivot.columns])
    if pivot.empty: return

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("#f7f3ed"); ax.set_facecolor("#fbf7f0")
    x = np.arange(len(pivot.index))
    w = 0.8 / max(len(pivot.columns), 1)

    for i, s in enumerate(pivot.columns):
        off = (i - (len(pivot.columns)-1)/2) * w
        bars = ax.bar(x+off, pivot[s].values, width=w, color=COLORS.get(s,"#999"),
                      edgecolor="#2b2b2b", lw=1, label=s, zorder=3)
        for b in bars:
            h = b.get_height()
            if not np.isnan(h):
                ax.text(b.get_x()+b.get_width()/2, h+0.01, f"{h:.2f}",
                        ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set(xticks=x, ylabel="GQR-Score", title="GQR-Score by Model and Strategy", ylim=(0, 1.05))
    ax.set_xticklabels(pivot.index, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.2, zorder=0)
    ax.legend(loc="lower right", frameon=True, facecolor="#fff", edgecolor="#3a3a3a")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    log.info("Saved %s", path)

def plot_tradeoff(df, path):
    fig, ax = plt.subplots(figsize=(10, 6))
    for s in df["strategy"].unique():
        sub = df[df["strategy"] == s]
        ax.scatter(sub["avg_latency"], sub["gqr_score"], color=COLORS.get(s,"#999"),
                   edgecolor="#3a3a3a", lw=0.6, s=80, label=s, zorder=3)
        for _, r in sub.iterrows():
            ax.annotate(r["model"].split("/")[-1], (r["avg_latency"], r["gqr_score"]),
                        textcoords="offset points", xytext=(6,6), fontsize=7)

    ax.set(xscale="log", xlabel="Avg Latency (s, log)", ylabel="GQR-Score",
           title="Latency vs GQR-Score")
    ax.grid(True, alpha=0.15)
    ax.legend(loc="lower left", frameon=True, facecolor="#fff", edgecolor="#3a3a3a")
    # direction arrows
    kw = dict(xycoords="axes fraction", arrowprops=dict(arrowstyle="-|>", color="#7a7a7a", lw=1.2))
    ax.annotate("", xy=(0.35,0.97), xytext=(0.7,0.97), **kw)
    ax.text(0.9, 0.975, "FASTER", transform=ax.transAxes, ha="left", va="center", color="#7a7a7a", fontsize=11)
    ax.annotate("", xy=(0.03,0.7), xytext=(0.03,0.4), **kw)
    ax.text(0.02, 0.55, "BETTER", transform=ax.transAxes, rotation=90, ha="center", va="bottom", color="#7a7a7a", fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    log.info("Saved %s", path)

# -----------------------------------------------------------------------------
# main

def main():
    global DRY_RUN
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="quick sanity check with few samples")
    args = parser.parse_args()
    DRY_RUN = args.dry_run
    if DRY_RUN:
        log.info("DRY RUN — %d samples per split", DRY_RUN_N)

    RESULTS_DIR.mkdir(exist_ok=True)
    SAVES_DIR.mkdir(exist_ok=True)

    # output path encodes hyperparams
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    hp = (f"tok{MAX_NEW_TOKENS_BASELINE}-{MAX_NEW_TOKENS_DSPY}_ctx{MAX_INPUT_LEN}"
          f"_gepa{GEPA_TRAIN_SIZE}-{GEPA_VAL_SIZE}-{GEPA_AUTO}-s{GEPA_SEED}"
          f"_fs{FS_TRAIN_SIZE}-d{FS_MAX_DEMOS}-c{FS_CANDIDATES}")
    csv_path = RESULTS_DIR / f"hf_{'+'.join(STRATEGIES)}_{hp}_{ts}.csv"

    # preload training data once
    train_data, eval_data = gqr.load_train_dataset()
    n_gepa_t = DRY_RUN_N if DRY_RUN else GEPA_TRAIN_SIZE
    n_gepa_v = DRY_RUN_N if DRY_RUN else GEPA_VAL_SIZE
    n_fs     = DRY_RUN_N if DRY_RUN else FS_TRAIN_SIZE
    trainset_gepa  = build_examples(train_data)[:n_gepa_t]
    valset_gepa    = build_examples(eval_data)[:n_gepa_v]
    trainset_fs    = build_examples(train_data)[:n_fs]

    rows = []

    for name in MODELS:
        tag = name.replace("/", "_")
        print(f"\n{'='*60}\n  {name}\n{'='*60}")

        mdl = tok = lm = None

        def _skip(strategy, err):
            log.error("SKIP %s [%s]: %s", name, strategy, err)
            rows.append(dict(model=name, strategy=strategy, avg_latency=None,
                             id_acc=None, ood_acc=None, gqr_score=None, dataset_acc=None))

        # load model once, reuse for all strategies
        try:
            mdl, tok = load_model(name)
        except Exception as e:
            for s in STRATEGIES:
                _skip(s, e)
            pd.DataFrame(rows).to_csv(csv_path, index=False)
            continue

        try:
            # --- baseline: raw HF inference ---
            if "baseline" in STRATEGIES:
                try:
                    fn, lats = scorer_baseline(mdl, tok)
                    rows.append(eval_model(name, "baseline", fn, lats))
                except Exception as e:
                    _skip("baseline", e)

            # --- dspy / gepa / fewshot: all share one HFLocalLM ---
            dspy_strats = [s for s in ("dspy","gepa","fewshot") if s in STRATEGIES]
            if dspy_strats:
                lm = make_dspy_lm(name, mdl, tok)
                dspy.configure(lm=lm)

                if "dspy" in STRATEGIES:
                    try:
                        fn, lats = scorer_dspy()
                        rows.append(eval_model(name, "dspy", fn, lats))
                    except Exception as e:
                        _skip("dspy", e)

                if "gepa" in STRATEGIES:
                    try:
                        student = SafePredict()
                        opt = dspy.GEPA(metric=gepa_metric, reflection_lm=lm, auto=GEPA_AUTO, seed=GEPA_SEED)
                        prog = opt.compile(student, trainset=trainset_gepa, valset=valset_gepa)
                        prog.save(SAVES_DIR / f"gepa_{tag}.json")
                        fn, lats = scorer_dspy(program=prog)
                        rows.append(eval_model(name, "gepa", fn, lats))
                    except Exception as e:
                        _skip("gepa", e)

                if "fewshot" in STRATEGIES:
                    try:
                        student = SafePredict()
                        opt = dspy.BootstrapFewShotWithRandomSearch(
                            max_labeled_demos=FS_MAX_DEMOS, num_candidate_programs=FS_CANDIDATES, metric=dspy_metric)
                        prog = opt.compile(student, trainset=trainset_fs)
                        prog.save(SAVES_DIR / f"fewshot_{tag}.json")
                        fn, lats = scorer_dspy(program=prog)
                        rows.append(eval_model(name, "fewshot", fn, lats))
                    except Exception as e:
                        _skip("fewshot", e)
        finally:
            if lm is not None:
                lm._hf_model = None
                lm._hf_tok = None
            dspy.configure(lm=None)
            del mdl, tok, lm
            mdl = tok = lm = None
            free_gpu()

        # checkpoint after every model
        pd.DataFrame(rows).to_csv(csv_path, index=False)

    # final summary + plots
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    log.info("Results → %s", csv_path)

    valid = df.dropna(subset=["gqr_score"])
    if not valid.empty:
        plot_bars(valid, RESULTS_DIR / "barplots.png")
        plot_tradeoff(valid, RESULTS_DIR / "tradeoff.png")

    print(f"\n{'='*60}\n  done.\n{'='*60}")
    print(df[["model","strategy","gqr_score","avg_latency"]].to_string(index=False))

if __name__ == "__main__":
    main()
