"""Checks for the TODOs in project1.ipynb. Each check prints ✅ or ❌ with a hint.

A ✅ means your function gives the right answer on a few examples. It does not
prove that it is correct everywhere, so read the hints even when everything is green.
"""


def _report(passed, message):
    if passed:
        print("✅", message)
    else:
        print("❌", message)
    return passed


def _all_passed(results):
    if all(results):
        print("\nAll checks passed. 🎉")
    else:
        print("\nSome checks failed. Read the ❌ lines, fix your function, and run this cell again.")


def check_my_hooks(my_hooks, scores):
    results = [
        _report(isinstance(my_hooks, list) and len(my_hooks) == 3, "my_hooks is a list with 3 hooks"),
        _report(all(isinstance(hook, str) and len(hook.strip()) > 0 for hook in my_hooks), "every hook is a non-empty string"),
        _report(len(set(my_hooks)) == len(my_hooks), "the 3 hooks are different"),
        _report(len(scores) == len(my_hooks), "every hook got a score"),
    ]
    _all_passed(results)


def check_assign_segment(assign_segment):
    examples = [
        ({"n_purchases": 0, "avg_order_value": 0.0, "days_since_last_purchase": None, "satisfaction": None}, "N", "never bought anything"),
        ({"n_purchases": 6, "avg_order_value": 80.0, "days_since_last_purchase": 20, "satisfaction": 0.30}, "A", "4+ purchases but unhappy (satisfaction 0.30)"),
        ({"n_purchases": 6, "avg_order_value": 80.0, "days_since_last_purchase": 100, "satisfaction": 0.90}, "A", "4+ purchases, last one 100 days ago"),
        ({"n_purchases": 4, "avg_order_value": 80.0, "days_since_last_purchase": 61, "satisfaction": 0.90}, "A", "exactly 4 purchases, last one 61 days ago"),
        ({"n_purchases": 8, "avg_order_value": 60.0, "days_since_last_purchase": 30, "satisfaction": 0.80}, "H", "many recent purchases, CHF 60 per order"),
        ({"n_purchases": 5, "avg_order_value": 45.0, "days_since_last_purchase": 60, "satisfaction": 0.46}, "H", "on the H borders: 5 purchases, CHF 45, 60 days, satisfaction 0.46"),
        ({"n_purchases": 4, "avg_order_value": 60.0, "days_since_last_purchase": 30, "satisfaction": 0.80}, "R", "only 4 purchases, so not H"),
        ({"n_purchases": 2, "avg_order_value": 15.0, "days_since_last_purchase": 120, "satisfaction": 0.40}, "R", "few purchases (A needs 4+), CHF 15 per order"),
        ({"n_purchases": 3, "avg_order_value": 14.99, "days_since_last_purchase": 10, "satisfaction": 0.90}, "L", "CHF 14.99 per order"),
    ]
    results = []
    for customer, expected, why in examples:
        try:
            answer = assign_segment(customer)
        except Exception as error:
            answer = f"an error: {error!r}"
        results.append(_report(answer == expected, f"{why}: expected {expected!r}, got {answer!r}"))
    _all_passed(results)


def check_count_per_segment(count_per_segment):
    customers = [{"segment": s} for s in ["N", "H", "H", "R", "A", "A", "A"]]
    try:
        counts = count_per_segment(customers)
    except Exception as error:
        _report(False, f"your function raised an error: {error!r}")
        return
    results = [
        _report(isinstance(counts, dict), "returns a dictionary"),
        _report(counts.get("N") == 1 and counts.get("H") == 2 and counts.get("R") == 1 and counts.get("A") == 3,
                f"counts N=1, H=2, R=1, A=3 (got {counts})"),
        _report(counts.get("L", 0) == 0, "a segment with no customers has count 0 or is missing"),
    ]
    _all_passed(results)


def check_visit_probability(visit_probability):
    examples = [
        ((0.0, 0.05, 0.6), 0.05, "score 0 gives just the base rate"),
        ((1.0, 0.05, 0.6), 0.65, "score 1 gives base rate + lift"),
        ((0.5, 0.05, 0.6), 0.35, "score 0.5 gives 0.05 + 0.6 * 0.5"),
        ((0.5, 0.10, 0.2), 0.20, "other numbers: 0.10 + 0.2 * 0.5"),
    ]
    results = []
    for arguments, expected, why in examples:
        try:
            answer = visit_probability(*arguments)
            ok = answer is not None and abs(answer - expected) < 1e-9
        except Exception as error:
            answer, ok = f"an error: {error!r}", False
        results.append(_report(ok, f"{why}: expected {expected}, got {answer}"))
    _all_passed(results)


def check_visit_rate_per_segment(visit_rate_per_segment):
    customers = [
        {"customer_id": 1, "segment": "N"}, {"customer_id": 2, "segment": "N"},
        {"customer_id": 3, "segment": "H"}, {"customer_id": 4, "segment": "H"},
        {"customer_id": 5, "segment": "H"}, {"customer_id": 6, "segment": "H"},
        {"customer_id": 7, "segment": "A"},
    ]
    visited = {1: True, 2: False, 3: True, 4: True, 5: True, 6: False, 7: False}
    try:
        rates = visit_rate_per_segment(customers, visited)
    except Exception as error:
        _report(False, f"your function raised an error: {error!r}")
        return
    results = [
        _report(isinstance(rates, dict), "returns a dictionary"),
        _report(abs(rates.get("N", -1) - 0.5) < 1e-9, f"N: 1 of 2 visited, expected 0.5, got {rates.get('N')}"),
        _report(abs(rates.get("H", -1) - 0.75) < 1e-9, f"H: 3 of 4 visited, expected 0.75, got {rates.get('H')}"),
        _report(abs(rates.get("A", -1) - 0.0) < 1e-9, f"A: 0 of 1 visited, expected 0.0, got {rates.get('A')}"),
    ]
    _all_passed(results)


def check_format_history(format_history):
    history = [
        {"round": 0, "hook": "Hook A", "visit_rate": 0.20},
        {"round": 1, "hook": "Hook B", "visit_rate": 0.50},
        {"round": 1, "hook": "Hook C", "visit_rate": 0.35},
        {"round": 2, "hook": "Hook D", "visit_rate": 0.10},
    ]
    try:
        text = format_history(history, 3)
    except Exception as error:
        _report(False, f"your function raised an error: {error!r}")
        return
    results = [_report(isinstance(text, str), "returns a string")]
    if isinstance(text, str):
        results.append(_report("Hook B" in text and "Hook C" in text and "Hook A" in text, "contains the 3 best hooks (B, C, A)"))
        results.append(_report("Hook D" not in text, "leaves out the worst hook (D), because k=3"))
        results.append(_report("50%" in text and "35%" in text and "20%" in text, "shows the visit rates as percentages (50%, 35%, 20%)"))
        order_ok = "Hook A" in text and "Hook B" in text and text.index("Hook A") < text.index("Hook C") < text.index("Hook B")
        results.append(_report(order_ok, "sorted from worst to best (A, then C, then B), so the best hook comes last"))
        print("\nYour text looks like this:\n" + text)
    _all_passed(results)


def check_clean_hook(clean_hook):
    examples = [
        ("Free returns on your first order.", "Free returns on your first order.", "a clean answer stays the same"),
        ('"Free returns on your first order."', "Free returns on your first order.", "quotes around the hook are removed"),
        ("  Free returns on your first order.  \n", "Free returns on your first order.", "spaces and line breaks around it are removed"),
        ("Thinking: they worry about quality.\nHook: Free returns on your first order.", "Free returns on your first order.", "with a 'Hook:' line, only the text after 'Hook:' is kept"),
        ("Hi [Name], welcome to our shop!", "", "a hook with a placeholder like [Name] is thrown away (empty string)"),
        ("", "", "an empty answer gives an empty string"),
    ]
    results = []
    for text, expected, why in examples:
        try:
            answer = clean_hook(text)
        except Exception as error:
            answer = f"an error: {error!r}"
        results.append(_report(answer == expected, f"{why}: expected {expected!r}, got {answer!r}"))
    _all_passed(results)


def check_meta_prompt(build_meta_prompt):
    history = [
        {"round": 0, "hook": "Check out our latest offers!", "visit_rate": 0.21},
        {"round": 1, "hook": "We miss you! Come back and see what's new.", "visit_rate": 0.18},
        {"round": 1, "hook": "Your favourite styles are back in stock.", "visit_rate": 0.33},
    ]
    try:
        prompt = build_meta_prompt("A", history, 2)
    except Exception as error:
        _report(False, f"your function raised an error: {error!r}")
        return
    results = [_report(isinstance(prompt, str) and len(prompt) > 50, "returns a prompt text (a string of some length)")]
    if isinstance(prompt, str):
        results.append(_report("Your favourite styles are back in stock." in prompt, "contains the best past hook"))
        results.append(_report("We miss you!" not in prompt, "leaves out hooks beyond the k=2 best"))
        results.append(_report("33%" in prompt, "shows the past visit rates"))
        print("\nYour meta-prompt for segment A looks like this:\n" + "-" * 60 + "\n" + prompt + "\n" + "-" * 60)
    _all_passed(results)
