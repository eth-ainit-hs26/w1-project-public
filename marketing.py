"""Helper functions for Project 1: a marketing agent that improves its emails with OPRO.

You do not need to change this file, but you are welcome to read it. Everything
here is a plain function working on plain data (strings, lists, dictionaries).

Overview:
  - Language model:  setup_model(), ask_llm(), ask_llm_many()
  - Data:            load_customers(), load_segments(), load_feedback()
  - Hidden customer: customer_reaction(), customer_reactions()
  - Campaign:        simulate_campaign()
  - OPRO:            baseline_meta_prompt(), run_opro()
  - Results:         show_best_hooks(), plot_runs(), compare_runs()
  - Looking at data: plot_customer_data(), plot_segments(), plot_campaign_outcomes(),
                     plot_all_hooks(), plot_score_vs_visit_rate()
"""
import base64
import csv
import json
import math
import random
import time
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

SEGMENTS = ["N", "H", "R", "L", "A"]
STARTING_HOOK = "Check out our latest offers!"

DEFAULT_MODEL = "google/gemma-4-E2B-it"     # judge and optimizer (chosen in the judge tests)
FALLBACK_MODEL = "Qwen/Qwen2.5-3B-Instruct"  # use this if the default model causes problems

# The loaded language model. A dictionary lets all functions below share it.
LLM = {"name": None, "tokenizer": None, "model": None, "test_mode": False}


# ---------------------------------------------------------------------------
# Language model
# ---------------------------------------------------------------------------

def setup_model(name=DEFAULT_MODEL):
    """Download (first time only) and load the language model onto the GPU."""
    import torch
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        print("Warning: no GPU found. In Colab: Runtime -> Change runtime type -> T4 GPU.")
    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(name)
    # Pad on the left, so that in a batch every prompt ends right where the answer starts.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Gemma overflows in float16, so it uses bfloat16 (tested on a T4, see the course notes).
    if "gemma" in name.lower():
        dtype = torch.bfloat16
    else:
        dtype = torch.float16
    # Newer models (Gemma 4) can also read images and need the multimodal class.
    config = AutoConfig.from_pretrained(name)
    if hasattr(config, "vision_config"):
        model_class = transformers.AutoModelForMultimodalLM
    else:
        model_class = AutoModelForCausalLM
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model_class.from_pretrained(name, dtype=dtype).to(device)
    model.eval()
    LLM["name"] = name
    LLM["tokenizer"] = tokenizer
    LLM["model"] = model
    LLM["test_mode"] = False
    print(f"Loaded {name} in {time.time() - start:.0f} seconds.")


def use_test_mode():
    """For course staff: replace the language model with a fast fake one, to test the code without a GPU."""
    LLM["name"] = "test mode (fake model)"
    LLM["test_mode"] = True


def _chat_text(messages):
    # enable_thinking=False stops "thinking" models from writing a long reasoning block first.
    return LLM["tokenizer"].apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def ask_llm_many(prompts, n=1, temperature=1.0, max_new_tokens=150):
    """Send several prompts at once and get n answers for each.

    Returns a list with one entry per prompt; each entry is a list of n answers.
    Asking for many answers in one go is much faster on a GPU than asking one by one."""
    if LLM["test_mode"]:
        return _fake_answers(prompts, n)
    if LLM["model"] is None:
        raise RuntimeError("No model loaded yet: run marketing.setup_model() first.")
    import torch

    tokenizer = LLM["tokenizer"]
    model = LLM["model"]
    texts = [_chat_text([{"role": "user", "content": prompt}]) for prompt in prompts]
    inputs = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            top_p=0.95 if temperature > 0 else None,
            max_new_tokens=max_new_tokens,
            num_return_sequences=n,
            pad_token_id=tokenizer.pad_token_id,
        )
    prompt_length = inputs["input_ids"].shape[1]
    answers = [tokenizer.decode(row[prompt_length:], skip_special_tokens=True).strip() for row in output]
    # The model returns the n answers of the first prompt, then those of the second prompt, ...
    return [answers[i * n:(i + 1) * n] for i in range(len(prompts))]


def ask_llm(prompt, temperature=1.0, max_new_tokens=150):
    """Send one prompt to the language model and return its answer as a string."""
    return ask_llm_many([prompt], n=1, temperature=temperature, max_new_tokens=max_new_tokens)[0][0]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_customers():
    """Read data/customers.csv into a list of dictionaries, one per customer.

    Numbers are converted from text to numbers. Customers who never bought
    anything have None for days_since_last_purchase and satisfaction."""
    customers = []
    with open(DATA_DIR / "customers.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            customer = {
                "customer_id": int(row["customer_id"]),
                "name": row["name"],
                "age": int(row["age"]),
                "n_purchases": int(row["n_purchases"]),
                "avg_order_value": float(row["avg_order_value"]),
                "days_since_last_purchase": None,
                "satisfaction": None,
            }
            if row["days_since_last_purchase"] != "":
                customer["days_since_last_purchase"] = int(row["days_since_last_purchase"])
            if row["satisfaction"] != "":
                customer["satisfaction"] = float(row["satisfaction"])
            customers.append(customer)
    return customers


def load_segments():
    """Read data/segments.csv: {"N": {"name": ..., "description": ...}, ...}"""
    segments = {}
    with open(DATA_DIR / "segments.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            segments[row["segment"]] = {"name": row["name"], "description": row["description"]}
    return segments


def load_feedback():
    """Read data/customer_feedback.csv: {"N": ["quote 1", "quote 2", ...], ...}"""
    feedback = {segment: [] for segment in SEGMENTS}
    with open(DATA_DIR / "customer_feedback.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            feedback[row["segment"]].append(row["quote"])
    return feedback


def _load_personas():
    # The personas are base64-encoded so nobody reads them by accident. Please don't decode them.
    with open(DATA_DIR / "personas.json", encoding="utf-8") as f:
        encoded = json.load(f)["personas"]
    return {segment: base64.b64decode(text).decode("utf-8") for segment, text in encoded.items()}


# ---------------------------------------------------------------------------
# The hidden customer (the "judge")
# ---------------------------------------------------------------------------

JUDGE_SYSTEM = "You are role-playing a customer of an online clothing shop. Stay in character.\nAbout you: {persona}"
JUDGE_USER = (
    'You receive a marketing email that starts with:\n"{hook}"\n\n'
    "Would you click through and visit the shop? Answer with exactly one word: YES or NO."
)
# The judge's YES-vs-NO preference ("gap") is turned into a score between 0 and 1 with
# sigmoid((gap - SCORE_CENTRE) / SCORE_SPREAD). The two numbers were fitted in the judge tests.
SCORE_CENTRE = -6.0
SCORE_SPREAD = 6.0

_reaction_cache = {}  # (segment, hook) -> score, so the same hook is never judged twice


def customer_reactions(segment_hook_pairs):
    """Score many (segment, hook) pairs at once. Returns a list of scores between 0 and 1.

    For each pair, the language model plays the hidden customer persona of that
    segment, reads the hook and is asked: "Would you visit the shop?". The score
    is how strongly it leans towards YES."""
    to_judge = [pair for pair in segment_hook_pairs if pair not in _reaction_cache]
    to_judge = list(dict.fromkeys(to_judge))  # drop duplicates, keep the order
    if to_judge:
        if LLM["test_mode"]:
            scores = [_fake_score(segment, hook) for segment, hook in to_judge]
        else:
            scores = _judge_batch(to_judge)
        for pair, score in zip(to_judge, scores):
            _reaction_cache[pair] = score
    return [_reaction_cache[pair] for pair in segment_hook_pairs]


def customer_reaction(segment, hook):
    """How much the hidden customer of this segment likes the hook: a score between 0 and 1."""
    return customer_reactions([(segment, hook)])[0]


def _judge_batch(segment_hook_pairs, batch_size=20):
    import torch

    if LLM["model"] is None:
        raise RuntimeError("No model loaded yet: run marketing.setup_model() first.")
    tokenizer = LLM["tokenizer"]
    model = LLM["model"]
    personas = _load_personas()
    yes_ids = _single_token_ids(["YES", "Yes", "yes"])
    no_ids = _single_token_ids(["NO", "No", "no"])
    scores = []
    for start in range(0, len(segment_hook_pairs), batch_size):
        chunk = segment_hook_pairs[start:start + batch_size]
        texts = []
        for segment, hook in chunk:
            messages = [
                {"role": "system", "content": JUDGE_SYSTEM.format(persona=personas[segment])},
                {"role": "user", "content": JUDGE_USER.format(hook=hook)},
            ]
            texts.append(_chat_text(messages))
        batch = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            # logits_to_keep=1: we only need the model's scores for the first answer word.
            logits = model(**batch, logits_to_keep=1).logits[:, -1, :].float()
        if not torch.isfinite(logits).all():
            raise RuntimeError("The model produced invalid numbers (NaN). Try marketing.setup_model(marketing.FALLBACK_MODEL).")
        gaps = torch.logsumexp(logits[:, yes_ids], dim=1) - torch.logsumexp(logits[:, no_ids], dim=1)
        for gap in gaps.tolist():
            scores.append(1 / (1 + math.exp(-(gap - SCORE_CENTRE) / SCORE_SPREAD)))
    return scores


def _single_token_ids(words):
    ids = []
    for word in words:
        tokens = LLM["tokenizer"].encode(word, add_special_tokens=False)
        if len(tokens) == 1:
            ids.append(tokens[0])
    return ids


# ---------------------------------------------------------------------------
# Campaign simulation
# ---------------------------------------------------------------------------

def simulate_campaign(customers, hook_per_segment, visit_probability, base_rate=0.05, lift=0.6, rng=None):
    """Send one email campaign and simulate who visits the shop.

    customers:          list of customer dictionaries, each with a "segment" key
    hook_per_segment:   {"N": "hook for N", ...}; segments without a hook get no email
    visit_probability:  your function visit_probability(score, base_rate, lift)

    Returns {customer_id: True/False}: did this customer visit the shop?
    Each customer flips a biased coin, so the result is a little different every time."""
    if rng is None:
        rng = random.Random()
    segments_with_hook = [segment for segment in hook_per_segment if hook_per_segment[segment]]
    pairs = [(segment, hook_per_segment[segment]) for segment in segments_with_hook]
    scores = dict(zip(segments_with_hook, customer_reactions(pairs)))
    visited = {}
    for customer in customers:
        segment = customer["segment"]
        if segment in scores:
            probability = visit_probability(scores[segment], base_rate, lift)
            visited[customer["customer_id"]] = rng.random() < probability
        else:
            visited[customer["customer_id"]] = False
    return visited


# ---------------------------------------------------------------------------
# OPRO
# ---------------------------------------------------------------------------

def baseline_meta_prompt(segment, history, k):
    """A simple meta-prompt to compare against: the segment description and the k best past hooks."""
    description = load_segments()[segment]["description"]
    best = sorted(history, key=lambda attempt: attempt["visit_rate"], reverse=True)[:k]
    lines = []
    for attempt in reversed(best):
        lines.append(f'Hook: "{attempt["hook"]}" - visit rate {round(attempt["visit_rate"] * 100)}%')
    past_hooks = "\n".join(lines)
    return (
        "You write the opening line (the hook) of marketing emails for an online clothing shop.\n"
        f"Target customers: {description}\n\n"
        f"Hooks we sent before, from worst to best:\n{past_hooks}\n\n"
        "Write one new hook that gets a higher visit rate than all of these. Reply with the hook only."
    )


def run_opro(customers, build_meta_prompt, clean_hook, visit_probability, visit_rate_per_segment,
             label="my run", rounds=6, candidates=4, k=8, temperature=1.0, seed=0,
             base_rate=0.05, lift=0.6, max_new_tokens=150):
    """Run the OPRO loop and return the results as a dictionary.

    Each round, for every segment:
      1. build_meta_prompt(segment, history, k) writes the meta-prompt (your function),
      2. the language model answers it `candidates` times,
      3. clean_hook(answer) picks out the new hook (your function),
      4. every new hook is sent in a simulated campaign, and
         visit_rate_per_segment(customers, visited) measures how it did (your function).
    Round 0 sends the starting hook to everybody, so every segment has a first result."""
    rng = random.Random(seed)
    if not LLM["test_mode"]:
        import torch

        torch.manual_seed(seed)
    history = {segment: [] for segment in SEGMENTS}
    best_per_round = []
    start = time.time()
    print(f"OPRO run '{label}': {rounds} rounds x {candidates} new hooks per segment")
    for round_number in range(rounds + 1):
        # 1-3: new hooks for every segment (round 0: the starting hook)
        if round_number == 0:
            campaigns = [{segment: STARTING_HOOK for segment in SEGMENTS}]
        else:
            prompts = [build_meta_prompt(segment, history[segment], k) for segment in SEGMENTS]
            answers = ask_llm_many(prompts, n=candidates, temperature=temperature, max_new_tokens=max_new_tokens)
            campaigns = [{} for _ in range(candidates)]
            for segment, segment_answers in zip(SEGMENTS, answers):
                seen = {attempt["hook"].lower() for attempt in history[segment]}
                for j, answer in enumerate(segment_answers):
                    hook = clean_hook(answer)
                    if hook and hook.lower() not in seen:
                        campaigns[j][segment] = hook
                        seen.add(hook.lower())
        # Judge all new hooks of this round in one go (faster), then run the campaigns.
        all_pairs = [(segment, hook) for campaign in campaigns for segment, hook in campaign.items()]
        customer_reactions(all_pairs)
        # 4: one simulated campaign per set of hooks
        for campaign in campaigns:
            if not campaign:
                continue
            visited = simulate_campaign(customers, campaign, visit_probability, base_rate, lift, rng)
            rates = visit_rate_per_segment(customers, visited)
            for segment in campaign:
                if segment not in rates:
                    raise ValueError(f"visit_rate_per_segment gave no visit rate for segment {segment}. Finish TODO 5 first.")
            for segment, hook in campaign.items():
                history[segment].append({
                    "round": round_number,
                    "hook": hook,
                    "visit_rate": rates[segment],
                    "score": customer_reaction(segment, hook),
                })
        best = {segment: max(attempt["visit_rate"] for attempt in history[segment]) for segment in SEGMENTS}
        best_per_round.append(best)
        best_text = "  ".join(f"{segment} {round(best[segment] * 100)}%" for segment in SEGMENTS)
        print(f"  round {round_number}: best visit rate so far  {best_text}   ({time.time() - start:.0f} s)")
    return {"label": label, "history": history, "best_per_round": best_per_round, "seconds": time.time() - start,
            "base_rate": base_rate, "lift": lift}


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

def best_hook(run, segment):
    """The attempt with the highest measured visit rate in one segment."""
    return max(run["history"][segment], key=lambda attempt: attempt["visit_rate"])


def show_best_hooks(run):
    """Print the best hook per segment of one run."""
    print(f"Best hooks of '{run['label']}':")
    for segment in SEGMENTS:
        attempt = best_hook(run, segment)
        print(f"  {segment}  {round(attempt['visit_rate'] * 100):3d}%  (round {attempt['round']})  {attempt['hook']}")


def plot_runs(runs):
    """Plot the best visit rate so far, per segment and on average, for one or more runs."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(SEGMENTS) + 1, figsize=(20, 3.5), sharey=True)
    for run in runs:
        x = list(range(len(run["best_per_round"])))
        for ax, segment in zip(axes, SEGMENTS):
            ax.plot(x, [best[segment] for best in run["best_per_round"]], marker="o", label=run["label"])
            ax.set_title(segment)
        average = [sum(best.values()) / len(best) for best in run["best_per_round"]]
        axes[-1].plot(x, average, marker="o", label=run["label"])
    axes[-1].set_title("average", fontsize=14)
    axes[0].set_ylabel("best visit rate so far", fontsize=12)
    for ax in axes:
        ax.set_xlabel("round", fontsize=12)
        ax.set_ylim(0, 0.8)
        _style(ax)
    
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", 
               bbox_to_anchor=(0.5, -0.12),  ncol=2, fontsize=14)
    fig.suptitle("Meta-prompt Comparison", fontsize=18)
    plt.tight_layout()
    plt.show()


def compare_runs(runs):
    """Show a table: the final best visit rate per segment for each run."""
    import pandas as pd

    rows = []
    for run in runs:
        row = {"run": run["label"]}
        for segment in SEGMENTS:
            row[segment] = round(best_hook(run, segment)["visit_rate"] * 100)
        row["average"] = round(sum(row[segment] for segment in SEGMENTS) / len(SEGMENTS))
        row["minutes"] = round(run["seconds"] / 60, 1)
        rows.append(row)
    return pd.DataFrame(rows).set_index("run")


# ---------------------------------------------------------------------------
# Looking at the data (plots only, nothing to fill in)
# ---------------------------------------------------------------------------

# One fixed colour per segment, the same in every plot.
# SEGMENT_COLORS = {"N": "#2a78d6", "H": "#eb6834", "R": "#1baf7a", "L": "#eda100", "A": "#e87ba4"}
SEGMENT_COLORS = {
    "N": "#648fff",  # Blue
    "H": "#fe6100",  # Orange
    "R": "#785ef0",  # Indigo / Purple
    "L": "#ffb000",  # Yellow
    "A": "#dc267f"   # Magenta
}
PLOT_COLOR = "#2a78d6"   # for plots with a single series
GRID_COLOR = "#e5e5e5"


def _style(ax):
    """Light grid behind the data, no box on the top and right."""
    ax.grid(True, color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_customer_data(customers):
    """One histogram per column of the customer data: how many customers have which value."""
    import matplotlib.pyplot as plt

    columns = [
        ("age", "age (years)"),
        ("n_purchases", "number of orders"),
        ("avg_order_value", "average order value (CHF)"),
        ("days_since_last_purchase", "days since last order"),
        ("satisfaction", "satisfaction (0 to 1)"),
    ]
    fig, axes = plt.subplots(1, len(columns), figsize=(20, 3.5))
    for ax, (column, title) in zip(axes, columns):
        # Customers who never ordered have no value (None) for some columns: leave them out.
        values = [customer[column] for customer in customers if customer[column] is not None]
        ax.hist(values, bins=25, color=PLOT_COLOR, edgecolor="white")
        missing = len(customers) - len(values)
        if missing > 0:
            title = f"{title}\n({missing} customers without a value)"
        ax.set_title(title, fontsize=10)
        _style(ax)
    axes[0].set_ylabel("number of customers")
    plt.tight_layout()
    plt.show()

def plot_segments(customers):
    """The spread of each metric within each segment, as box plots (needs the "segment" key)."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    metrics = [
        ("n_purchases", "number of orders"),
        ("avg_order_value", "average order value (CHF)"),
        ("days_since_last_purchase", "days since last order"),
        ("satisfaction", "satisfaction (0 to 1)"),
    ]

    fig = make_subplots(rows=1, cols=4, horizontal_spacing=0.06,
                        subplot_titles=[title for _, title in metrics])

    for i, (column, _) in enumerate(metrics):
        for segment in SEGMENTS:
            if segment != 'N':
                fig.add_trace(go.Box(
                    y=[customer[column] for customer in customers
                      if customer["segment"] == segment and customer[column] is not None],
                    name=segment, marker_color=SEGMENT_COLORS[segment],
                    boxpoints="outliers", showlegend=False,
                ), row=1, col=i + 1)

    fig.update_layout(height=450, template="plotly_white", margin=dict(l=50, r=20, t=60, b=40),
                      title=dict(text="Customer Analysis", x=0.5, y=0.98, xanchor="center")
    )
    fig.show()

def plot_campaign_outcomes(customers, hook_per_segment, visit_probability, visit_rate_per_segment, repeats=30, seed=0):
    """Send the same hooks `repeats` times and draw how much the visit rate jumps around per segment."""
    import matplotlib.pyplot as plt

    rng = random.Random(seed)
    rates_per_segment = {segment: [] for segment in SEGMENTS}
    for _ in range(repeats):
        visited = simulate_campaign(customers, hook_per_segment, visit_probability, rng=rng)
        rates = visit_rate_per_segment(customers, visited)
        for segment in SEGMENTS:
            if segment not in rates:
                print(f"visit_rate_per_segment gave no visit rate for segment {segment}. Finish TODO 5 first, then run this cell again.")
                return
            rates_per_segment[segment].append(rates[segment] * 100)

    fig, axes = plt.subplots(1, len(SEGMENTS), figsize=(20, 3.2), sharex=True, sharey=True)
    for ax, segment in zip(axes, SEGMENTS):
        rates = rates_per_segment[segment]
        ax.hist(rates, bins=range(0, 101, 2), color=SEGMENT_COLORS[segment], edgecolor="white")
        ax.set_title(f"{segment}: from {round(min(rates))}% to {round(max(rates))}%", fontsize=14)
        ax.set_xlabel("visit rate (%)", fontsize=12)
        _style(ax)
    lowest = min(min(rates) for rates in rates_per_segment.values())
    highest = max(max(rates) for rates in rates_per_segment.values())
    axes[0].set_xlim(max(0, lowest - 5), min(100, highest + 5))
    axes[0].set_ylabel(f"campaigns (of {repeats})", fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_all_hooks(run):
    """Every hook of one run as a dot (round vs measured visit rate), with the average and the best so far."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(SEGMENTS), figsize=(20, 3.8), sharey=True)
    for ax, segment in zip(axes, SEGMENTS):
        history = run["history"][segment]
        rounds = sorted(set(attempt["round"] for attempt in history))
        # Every hook as a grey dot. Hooks of the same round are spread out a little so they don't overlap.
        for round_number in rounds:
            rates = [attempt["visit_rate"] for attempt in history if attempt["round"] == round_number]
            offsets = [(j - (len(rates) - 1) / 2) * 0.08 for j in range(len(rates))]
            ax.scatter([round_number + offset for offset in offsets], rates, s=24, color="#a0a0a0",
                       label="one hook" if round_number == rounds[0] else None)
        average = []
        best_so_far = []
        best = 0
        for round_number in rounds:
            rates = [attempt["visit_rate"] for attempt in history if attempt["round"] == round_number]
            average.append(sum(rates) / len(rates))
            best = max(best, max(rates))
            best_so_far.append(best)
        ax.plot(rounds, average, color=PLOT_COLOR, linewidth=2, marker="o", label="average of the round")
        ax.plot(rounds, best_so_far, color="#eb6834", linewidth=2, linestyle="--", label="best so far")
        ax.set_title(segment, fontsize=14)
        ax.set_xlabel("round", fontsize=12)
        ax.set_ylim(0, 0.8)
        _style(ax)
    axes[0].set_ylabel("measured visit rate", fontsize=12)
    
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", 
               bbox_to_anchor=(0.5, -0.1),  ncol=3, fontsize=14)
    
    fig.suptitle(f"All hooks of '{run['label']}'", fontsize=18)
    plt.tight_layout()
    plt.show()


def plot_score_vs_visit_rate(run):
    """The hidden customer's score of each hook against the visit rate it got in its campaign."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for segment in SEGMENTS:
        history = run["history"][segment]
        ax.scatter([attempt["score"] for attempt in history], [attempt["visit_rate"] for attempt in history],
                   s=24, alpha=0.8, color=SEGMENT_COLORS[segment], label=segment)
    # Where the dots would lie without the coin flips: visit rate = base rate + lift x score
    ax.plot([0, 1], [run["base_rate"], run["base_rate"] + run["lift"]], color="#555555", linestyle="--",
            linewidth=1.5, label="without coin flips")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.8)
    ax.set_xlabel("score from the hidden customer (0 to 1)")
    ax.set_ylabel("measured visit rate")
    ax.set_title(f"Score vs. visit rate, '{run['label']}'", fontsize=10)
    ax.legend(fontsize=8)
    _style(ax)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Test mode (fake model, for course staff)
# ---------------------------------------------------------------------------

_FAKE_PARTS = ["Free returns", "New season", "Early access", "Bundle deal", "Sorry for the delay",
               "Real reviews", "CHF 19", "Free shipping", "Just for you", "Our new collection"]


def _fake_answers(prompts, n):
    rng = random.Random(len(_reaction_cache))
    return [[f"Thinking: (test mode) these customers might like this.\nHook: {rng.choice(_FAKE_PARTS)} - {rng.choice(_FAKE_PARTS).lower()}."
             for _ in range(n)] for _ in prompts]


def _fake_score(segment, hook):
    return random.Random(f"{segment}|{hook}").random()
