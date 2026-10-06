"""Held-out evaluation: plants the demo never uses (seeds 21, 31, 41), each with the same supplier notice (lot B-177, five days of
date codes). Every recall strategy is built from the records alone and scored on the simulation truth: which units really hold a
capacitor from the window's reels.

    python -m rc.evaluate > docs/evaluation.md
"""
import copy
import time

import numpy as np

from . import engine as E, world as W

SEEDS = (21, 31, 41)
NAMES = {"model": "Model: minimum defensible set", "mes": "MES serial genealogy (deterministic)", "work_order": "Whole work orders that used a suspect reel", "broad": "Every unit built while B-177 was on a line"}


def run(seed, rec=None, tr=None, **policy):
    if rec is None:
        rec, tr = W.generate(seed)
    d0, d1 = tr["window"]
    out = E.scope(rec, "B-177", d0, d1, **policy)
    affected = W.affected_boards(tr, "B-177", d0, d1)[tr["unit_final"]]
    return out, affected, E.score(out["sets"], affected, out["cost"]), tr


def m(v):
    return f"${v / 1e3:,.0f}k"


def miscount(rec):
    """Every advance ship notice claims one reel more of its first lot than arrived."""
    r = copy.deepcopy(rec)
    for dl in r["deliveries"]:
        a = next(iter(dl["lots"]))
        d = next(iter(dl["lots"][a]))
        dl["lots"][a][d] += 1
    return r


def main():
    t0 = time.time()
    out = ["# Evaluation", "", f"Three plants the demo never uses (seeds {', '.join(map(str, SEEDS))}), each with sixty days of production and the same supplier notice:",
           "lot B-177, five days of date codes. Every strategy is built from the records alone; the simulation truth (which units really hold a",
           "capacitor from the window's reels) scores them. Reproduce with `python -m rc.evaluate`.", "",
           "## Recall strategies against the truth", "", "| Plant | Strategy | Units recalled | Cost | Affected units recalled | Affected units missed | Unaffected units recalled |", "|---|---|---|---|---|---|---|"]
    runs = {}
    for seed in SEEDS:
        o, aff, sc, tr = run(seed)
        runs[seed] = (o, aff, sc, tr)
        out.append(f"| {seed} | Truth | {aff.sum():,} | {m(o['cost'][aff].sum())} | | | |")
        for k in ("model", "mes", "work_order", "broad"):
            s = sc[k]
            out.append(f"| {seed} | {NAMES[k]} | {s['units']:,} | {m(s['cost'])} | {s['affected_caught']:,} | {s['affected_missed']:,} | {s['unaffected_recalled']:,} |")
    out += ["", "## Why the MES gets it wrong", "", "| Plant | Missed: cause (units) | Recalled without cause: reason (units) |", "|---|---|---|"]
    for seed, (o, aff, sc, tr) in runs.items():
        rec, _ = W.generate(seed)
        ex = E.explain(rec, tr, E.mes(rec), "B-177", *tr["window"], o["sets"]["mes"], aff)
        f = lambda d: "; ".join(f"{k} ({v:,})" for k, v in d.items()) or "none"     # noqa: E731
        out.append(f"| {seed} | {f(ex['missed'])} | {f(ex['unaffected_recalled'])} |")
    p = np.concatenate([runs[s][0]["p_unit"] for s in SEEDS])
    a = np.concatenate([runs[s][1] for s in SEEDS])
    out += ["", "## Calibration of the unit probabilities (three plants pooled)", "", "| Predicted band | Units | Mean predicted | Share truly affected |", "|---|---|---|---|"]
    out += [f"| {c['band']} | {c['units']:,} | {c['predicted']:.3f} | {c['observed']:.3f} |" for c in E.calibration(p, a)]
    out += ["", "## What the record models learned", "", "| Plant | Lot-code character error (simulator 0.016) | Scan delay, line 3, minutes (simulator: 7 on line 3, 10 on lines 1-2) | Boards per reel (simulator about 490) | Splices observed |", "|---|---|---|---|---|"]
    for seed, (o, *_rest) in runs.items():
        f = o["fit"]
        out.append(f"| {seed} | {f['char_error']:.4f} | {f['delay_mean']:.1f} | {f['boards_per_reel']:.1f} ± {f['boards_per_reel_sd']:.1f} | {f['splices_observed']} |")
    out += ["", "## Policy: how much risk to leave out (three plants summed)", "", "| Max expected escapes (per plant) | Exclude above | Units recalled | Cost | Affected missed | Expected escapes (model, sum) |", "|---|---|---|---|---|---|"]
    for eps, ex in ((0.05, 0.001), (0.5, 0.01), (2.0, 0.05), (5.0, 0.2)):
        tot = [0, 0.0, 0, 0.0]
        for seed in SEEDS:
            o, aff, sc, _ = run(seed, max_expected_escapes=eps, exclude_above=ex)
            tot = [tot[0] + sc["model"]["units"], tot[1] + sc["model"]["cost"], tot[2] + sc["model"]["affected_missed"], tot[3] + o["expected_escapes"]]
        out.append(f"| {eps} | {ex} | {tot[0]:,} | {m(tot[1])} | {tot[2]} | {tot[3]:.2f} |")
    out += ["", "## When the model's assumptions are wrong (three plants summed)", "",
            "The model assumes three things it cannot check: stores hand out the oldest spares, lines 1 and 2 scan like line 3, and the advance ship notices count reels correctly. Each row breaks one in the simulator and leaves the model alone.", "",
            "| Simulator | Model units | Model affected missed | Expected escapes (model) | MES affected missed |", "|---|---|---|---|---|"]
    variants = [("As built (the rows above)", {}, False), ("Reworks take any spare on the shelf, not one of the five oldest", {"spare_pick": None}, False),
                ("Lines 1-2 scan delays three times line 3's", {"slow_scans": 3.0}, False), ("Every notice claims one reel more than arrived", {}, True)]
    for name, kw, mis in variants:
        tot = [0, 0, 0.0, 0]
        for seed in SEEDS:
            rec, tr = W.generate(seed, **kw)
            o, aff, sc, _ = run(seed, miscount(rec) if mis else rec, tr)
            tot = [tot[0] + sc["model"]["units"], tot[1] + sc["model"]["affected_missed"], tot[2] + o["expected_escapes"], tot[3] + sc["mes"]["affected_missed"]]
        out.append(f"| {name} | {tot[0]:,} | {tot[1]} | {tot[2]:.2f} | {tot[3]} |")
    out += ["", f"Run time {time.time() - t0:.0f} s on one core."]
    print("\n".join(out))


if __name__ == "__main__":
    main()
