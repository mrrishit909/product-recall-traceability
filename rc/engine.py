"""Genealogy under uncertainty, the recall-scope optimiser and the baselines, all computed from the records alone.

Three things are learned from the records before any recall: how often a lot code is misread (from deliveries that carried one
lot, where the right answer is known), how late operators scan a new reel after the splice (from line 3, whose splice sensors log
the true time), and how many boards a reel lasts (also line 3). Then:

    reel -> supplier lot     a posterior over the lots in the reel's delivery: OCR likelihood of the read code x the delivery's
                             advance ship notice counts, confident matches first, the rest share what the notice has left
    reel -> date code        read, or the share of the lot's notice in the window not accounted for by reels read in it
    board -> reels           P(splice before the board) from the scan time and the learned delay, or from the board count
                             when the scan was forgotten; a board at a splice can hold both reels
    unit -> board            the recorded board, the recorded replacement, or, when a rework ticket lost the serial, the
                             average over the spares that could have been fitted
"""
import math

import numpy as np

from . import world as W

VERSION = "rc-genealogy-1"
POLICY = {"max_expected_escapes": 0.5, "exclude_above": 0.01}
SPARE_CANDIDATES = 8
ALPHABET = 30


def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def lines_of(rec):
    b = rec["boards"]
    return {L: np.flatnonzero(b["line"] == L) for L in W.LINES}     # board indices on each line, in time order


def fit(rec):
    reels = rec["reels"]
    num = den = 0
    for i, s in enumerate(reels["lot_read"]):
        lots = rec["deliveries"][reels["delivery"][i]]["lots"]
        if s is not None and len(lots) == 1:
            code = next(iter(lots))
            num, den = num + lev(s, code), den + len(code)
    sensor = {e["reel"]: e["at"] for e in rec["sensor"]}
    delays = sorted(e["at"] - sensor[e["reel"]] for e in rec["scans"] if e["reel"] in sensor)
    b, idx = rec["boards"], lines_of(rec)[W.SENSOR_LINE]
    k = np.searchsorted(b["t"][idx], sorted(sensor.values()))
    bpr = np.diff(k) if len(k) > 2 else np.array([490.0, 490.0])
    return {"char_error": (num + 0.5) / (den + 1), "chars_checked": den, "lot_unread": float(np.mean([s is None for s in reels["lot_read"]])),
            "date_unread": float(np.mean([s is None for s in reels["date_read"]])), "delays": [round(float(d), 2) for d in delays],
            "delay_mean": float(np.mean(delays)) if delays else 10.0, "boards_per_reel": float(bpr.mean()), "boards_per_reel_sd": float(max(bpr.std(ddof=1), 1.0)),
            "splices_observed": len(delays)}


def read_lik(s, code, q):
    """P(the receiving scanner wrote `s` | the label said `code`): each character survives with 1 - q; a misread becomes one
    particular wrong character out of about thirty."""
    if s is None:
        return 1.0
    d = lev(s, code)
    return (q / ALPHABET) ** d * (1 - q) ** max(len(code) - d, 0)


def lot_posterior(rec, f):
    """-> (R x lots) matrix of P(reel came from lot). Exact under the advance ship notice's counts: in a delivery of two lots, which
    n_A of the reels are lot A is a subset; its weight is the product of the read likelihoods, and a reel's marginal is a ratio of
    elementary symmetric polynomials."""
    reels, q = rec["reels"], f["char_error"]
    post = np.zeros((len(reels["ref"]), len(W.LOT_CODES)))
    for d in sorted(set(reels["delivery"].tolist())):
        m = np.flatnonzero(reels["delivery"] == d)
        lots = {c: sum(v.values()) for c, v in rec["deliveries"][d]["lots"].items()}
        codes = list(lots)
        lik = np.array([[read_lik(reels["lot_read"][i], c, q) for c in codes] for i in m])
        cols = [W.LOT_CODES.index(c) for c in codes]
        if len(codes) == 1:
            post[m, cols[0]] = 1.0
        elif len(codes) == 2 and lots[codes[0]] + lots[codes[1]] == len(m):     # ponytail: two lots per delivery here; Sinkhorn for more
            r = lik[:, 0] / lik[:, 1]
            nA = lots[codes[0]]
            pa = np.array([r[i] * esp(np.delete(r, i), nA - 1) for i in range(len(m))]) / esp(r, nA)
            post[m, cols[0]], post[m, cols[1]] = pa, 1 - pa
        else:
            p = lik * np.array([lots[c] for c in codes])
            post[np.ix_(m, cols)] = p / p.sum(1, keepdims=True)
    return post


def esp(x, k):
    """k-th elementary symmetric polynomial of x."""
    e = np.zeros(k + 1)
    e[0] = 1.0
    for v in x:
        e[1:] = e[1:] + v * e[:-1]
    return e[k] if k >= 0 else 0.0


def window_prob(rec, post, lot, d_from, d_to):
    """P(reel is from `lot` with a date code in the window)."""
    if lot not in W.LOT_CODES:
        return np.zeros(len(rec["reels"]["ref"]))
    j = W.LOT_CODES.index(lot)
    pl = post[:, j]
    dates = rec["reels"]["date_read"]
    read = np.array([d is not None for d in dates])
    inwin = np.array([d is not None and d_from <= d <= d_to for d in dates])
    notice = sum(c for dl in rec["deliveries"] for code, counts in dl["lots"].items() if code == lot for d, c in counts.items() if d_from <= d <= d_to)
    missing = pl[~read].sum()
    q = float(np.clip((notice - pl[inwin].sum()) / missing, 0, 1)) if missing > 1e-9 else 0.0
    return pl * np.where(read, inwin.astype(float), q)


def delay_cdf(f):
    """Empirical CDF of the scan delay with exponential tails, so a delay longer than any seen on line 3 is unlikely, not impossible."""
    d = np.array(f["delays"] or [10.0])
    n, th = len(d), max(float(d.std()), 2.0)
    ps = (np.arange(n) + 0.5) / n

    def F(x):
        x = np.asarray(x, float)
        mid = np.interp(x, d, ps)
        lo = 0.5 / n * np.exp(np.minimum(x - d[0], 0) / th)
        hi = 1 - 0.5 / n * np.exp(-np.maximum(x - d[-1], 0) / th)
        return np.where(x < d[0], lo, np.where(x > d[-1], hi, mid))
    return F


def Phi(z):
    return 0.5 * (1 + np.vectorize(math.erf)(np.asarray(z, float) / math.sqrt(2)))


def mounts(rec):
    """The reels each line mounted, in order (stores' issue log), with the scan and sensor times where they exist."""
    scans = {(e["line"], e["reel"]): e["at"] for e in rec["scans"]}
    sensor = {(e["line"], e["reel"]): e["at"] for e in rec["sensor"]}
    out = {}
    for L in W.LINES:
        out[L] = [{"reel": e["reel"], "scan": scans.get((L, e["reel"])), "sensor": sensor.get((L, e["reel"]))} for e in rec["issues"] if e["line"] == L]
    return out


def board_reels(rec, f):
    """-> (board index, reel index, P(board holds parts from the reel)) for every pair above 1e-4, and the splice estimates."""
    F = delay_cdf(f)
    b = rec["boards"]
    rate_sd = float(np.std(f["delays"]) if f["delays"] else 8.0) / ((W.DAY_END - W.DAY_START) / W.BOARDS_PER_DAY)
    bi, ri, pi, splices = [], [], [], []
    for L, idx in lines_of(rec).items():
        t, tp = b["t"][idx], b["tp"][idx]
        n = len(t)
        ms = mounts(rec)[L]
        G, Gp, khat = [], [], []                 # P(splice i before board end), P(splice i before board start), expected splice board
        for i, m in enumerate(ms):
            if i == 0:
                G.append(np.ones(n)), Gp.append(np.ones(n)), khat.append((0.0, 0.0))
            elif m["sensor"] is not None:
                G.append((m["sensor"] < t) * 1.0), Gp.append((m["sensor"] <= tp) * 1.0)     # a splice exactly at a board's start: the board is all new reel
                khat.append((float(np.searchsorted(t, m["sensor"])), 0.0))
            elif m["scan"] is not None:
                G.append(1 - F(m["scan"] - t)), Gp.append(1 - F(m["scan"] - tp))
                khat.append((float(np.searchsorted(t, m["scan"] - f["delay_mean"])), rate_sd ** 2))
            else:
                G.append(None), Gp.append(None), khat.append(None)
        for i, m in enumerate(ms):                # forgotten scans: count boards from the neighbours
            if G[i] is None:
                est = []
                if khat[i - 1] is not None:
                    est.append((khat[i - 1][0] + f["boards_per_reel"], khat[i - 1][1] + f["boards_per_reel_sd"] ** 2))
                if i + 1 < len(ms) and khat[i + 1] is not None:
                    est.append((khat[i + 1][0] - f["boards_per_reel"], khat[i + 1][1] + f["boards_per_reel_sd"] ** 2))
                w = np.array([1 / v for _, v in est])
                mu, var = float(np.dot(w, [e for e, _ in est]) / w.sum()), float(1 / w.sum())
                khat[i] = (mu, var)
                k = np.arange(n)
                G[i], Gp[i] = Phi((k - mu + 0.5) / math.sqrt(var)), Phi((k - 1 - mu + 0.5) / math.sqrt(var))
        for i, m in enumerate(ms):
            after = 1 - Gp[i + 1] if i + 1 < len(ms) else np.ones(n)
            p = G[i] * after
            k = np.flatnonzero(p > 1e-4)
            bi.append(idx[k]), ri.append(np.full(len(k), m["reel"])), pi.append(p[k])
            if i:
                kind = "sensor" if m["sensor"] is not None else "scan" if m["scan"] is not None else "count"
                splices.append({"line": L, "reel": m["reel"], "kind": kind, "board": int(round(khat[i][0])), "sd_boards": round(math.sqrt(khat[i][1]), 1)})
    return np.concatenate(bi), np.concatenate(ri), np.concatenate(pi), splices


def unit_boards(rec):
    """Recorded final board per unit, and for reworks that lost the serial, the spares that could have been fitted."""
    u, b = rec["units"], rec["boards"]
    final = u["board"].copy()
    used = np.zeros(len(b["t"]), bool)
    used[u["board"]] = True
    for r in rec["reworks"]:
        if r["new_board"] is not None:
            final[r["unit"]] = r["new_board"]
            used[r["new_board"]] = True       # recorded replacements are known to be gone from the shelf, whenever they were fitted
    spares = np.flatnonzero(b["spare"])
    spares = spares[np.argsort(b["t"][spares])]
    unknown = {}
    for r in sorted(rec["reworks"], key=lambda r: r["at"]):
        if r["new_board"] is None:
            pool = spares[(b["t"][spares] < r["at"]) & ~used[spares]][:SPARE_CANDIDATES]    # ponytail: stores hand out the oldest spares; a ticket audit would replace this guess
            unknown[r["unit"]] = pool
            final[r["unit"]] = -1
            if len(pool):
                used[pool[0]] = True          # one of them left the shelf; assume the oldest, for the reworks that follow
    return final, unknown


def unit_prob(rec, p_board):
    final, unknown = unit_boards(rec)
    p = np.where(final >= 0, p_board[np.maximum(final, 0)], 0.0)
    for u, pool in unknown.items():
        p[u] = p_board[pool].mean() if len(pool) else 0.0
    return p, final, unknown


def board_prob(rec, f, post, lot, d_from, d_to, pairs=None):
    bi, ri, pi, splices = pairs or board_reels(rec, f)
    w = window_prob(rec, post, lot, d_from, d_to)
    keep = np.ones(len(rec["boards"]["t"]))
    np.multiply.at(keep, bi, 1 - pi * w[ri])
    return 1 - keep


# ---------------------------------------------------------------- the baselines: what the plant's own system says

def mes(rec):
    """Deterministic genealogy as an MES records it: the reel last scanned on the line before the board, the lot whose code
    matches the read exactly, the recorded replacement board (or the original when the ticket lost it)."""
    b = rec["boards"]
    board_reel = np.full(len(b["t"]), -1)
    events = {L: sorted([(e["at"], e["reel"]) for e in rec["scans"] if e["line"] == L] + [(e["at"], e["reel"]) for e in rec["sensor"] if e["line"] == L]) for L in W.LINES}
    for L, idx in lines_of(rec).items():
        at = np.array([a for a, _ in events[L]])
        reels = np.array([r for _, r in events[L]])
        k = np.searchsorted(at, b["t"][idx], "right") - 1
        board_reel[idx] = np.where(k >= 0, reels[np.maximum(k, 0)], -1)
    reel_lot = np.array([s if s in W.LOT_CODES else None for s in rec["reels"]["lot_read"]], dtype=object)
    unit_board = rec["units"]["board"].copy()
    for r in rec["reworks"]:
        if r["new_board"] is not None:
            unit_board[r["unit"]] = r["new_board"]
    return {"board_reel": board_reel, "reel_lot": reel_lot, "unit_board": unit_board}


def baselines(rec, m, lot, d_from, d_to):
    b, dates = rec["boards"], rec["reels"]["date_read"]
    of_lot = np.array([x == lot for x in m["reel_lot"]])
    suspect = of_lot & np.array([d is None or d_from <= d <= d_to for d in dates])          # an unread date code counts as suspect
    br = m["board_reel"]
    sb = (br >= 0) & suspect[np.maximum(br, 0)]
    ub = m["unit_board"]
    wo_bad = set(b["wo"][sb].tolist())
    lotb = np.flatnonzero((br >= 0) & of_lot[np.maximum(br, 0)])
    t0, t1 = (b["t"][lotb].min(), b["t"][lotb].max()) if len(lotb) else (np.inf, -np.inf)
    return {"broad": (b["t"][ub] >= t0) & (b["t"][ub] <= t1),
            "work_order": np.isin(b["wo"][ub], list(wo_bad)),
            "mes": sb[ub]}


# ---------------------------------------------------------------- scope

def optimise(p, cost, max_expected_escapes=0.5, exclude_above=0.01):
    """Cheapest set to recall such that the expected number of affected units left out stays within the policy and no unit above
    `exclude_above` is left out. Leaving units out is a knapsack on risk; greedy by cost saved per unit of risk is within one unit
    of optimal here because the risks are tiny."""
    include = p > exclude_above
    cand = np.flatnonzero(~include & (p > 0))
    o = cand[np.argsort(-cost[cand] / p[cand], kind="stable")]
    over = np.cumsum(p[o]) > max_expected_escapes
    include[o[over]] = True
    return include, float(p[~include].sum())


def unit_costs(rec):
    s = rec["unit_shipment"]
    ship_t = np.array([rec["shipments"][i]["at"] if i >= 0 else np.nan for i in s])
    return W.stage_cost((rec["as_of"] - ship_t) / 1440), ship_t


def stage_names(rec, ship_t):
    d = (rec["as_of"] - ship_t) / 1440
    return np.where(np.isnan(d), "in our warehouse", np.where(d < 10, "at the customer", "at the customer or in the field"))


def scope(rec, lot, d_from, d_to, f=None, max_expected_escapes=0.5, exclude_above=0.01, cache=None):
    f = f or fit(rec)
    cache = cache if cache is not None else {}
    if "post" not in cache:
        cache["post"] = lot_posterior(rec, f)
        cache["pairs"] = board_reels(rec, f)
    post, pairs = cache["post"], cache["pairs"]
    pb = board_prob(rec, f, post, lot, d_from, d_to, pairs)
    pu, final, unknown = unit_prob(rec, pb)
    cost, ship_t = unit_costs(rec)
    inc, esc = optimise(pu, cost, max_expected_escapes, exclude_above)
    sets = {"model": inc, **baselines(rec, mes(rec), lot, d_from, d_to)}
    return {"fit": f, "post": post, "pairs": pairs, "p_board": pb, "p_unit": pu, "final": final, "unknown": unknown,
            "cost": cost, "ship_t": ship_t, "sets": sets, "expected_escapes": esc, "window_reels": window_prob(rec, post, lot, d_from, d_to)}


def score(sets, affected, cost):
    out = {}
    for k, s in sets.items():
        out[k] = {"units": int(s.sum()), "cost": float(cost[s].sum()), "affected_caught": int((s & affected).sum()),
                  "affected_missed": int((~s & affected).sum()), "unaffected_recalled": int((s & ~affected).sum())}
    return out


def calibration(p, affected):
    edges = [1e-4, 0.05, 0.25, 0.75, 0.95, 0.9999]
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi)
        if m.any():
            out.append({"band": f"{lo:g}-{hi:g}", "units": int(m.sum()), "predicted": round(float(p[m].mean()), 3), "observed": round(float(affected[m].mean()), 3)})
    return out


def returns_by_lot(rec, m, lot):
    """Field returns per supplier lot (as the MES links units to lots), per 1,000 unit-days since shipping, so lots used early (more
    time in the field) are not blamed for being old; the rate ratio is against every other unit, with a 90% interval. And per date
    code within `lot`."""
    shipped = rec["unit_shipment"] >= 0
    ship_t = np.array([rec["shipments"][s]["at"] if s >= 0 else np.nan for s in rec["unit_shipment"]])
    days = np.nan_to_num((rec["as_of"] - ship_t) / 1440)
    br, ub = m["board_reel"], m["unit_board"]
    reel = br[ub]
    ulot = np.array([(m["reel_lot"][r] or "unread code") if r >= 0 else "unread code" for r in reel], dtype=object)
    returned, e07 = np.zeros(len(ub), bool), np.zeros(len(ub), bool)
    for r in rec["returns"]:
        returned[r["unit"]] = True
        e07[r["unit"]] |= r["symptom"].startswith("E07")
    rows = []
    for code in sorted(set(ulot[shipped].tolist())):
        a, rest = shipped & (ulot == code), shipped & (ulot != code)
        ra, rr, da, dr = returned[a].sum(), returned[rest].sum(), days[a].sum(), days[rest].sum()
        ratio = (ra / da) / (rr / dr) if ra and rr else None
        se = math.sqrt(1 / max(ra, 0.5) + 1 / max(rr, 0.5))
        rows.append({"lot": code, "units_shipped": int(a.sum()), "unit_days": round(float(da)), "returns": int(ra), "e07_returns": int((returned & e07 & a).sum()),
                     "per_1000_unit_days": round(float(1000 * ra / da), 3) if da else 0, "rate_ratio": round(float(ratio), 2) if ratio else None,
                     "ratio_90": [round(float(ratio * math.exp(-1.645 * se)), 2), round(float(ratio * math.exp(1.645 * se)), 2)] if ratio else None})
    dates = rec["reels"]["date_read"]
    of_lot = np.array([r >= 0 and m["reel_lot"][r] == lot for r in reel])
    udate = np.array([dates[r] if ok and dates[r] is not None else -999 for r, ok in zip(reel, of_lot)])
    by_date = []
    for d in sorted(set(udate[of_lot & shipped].tolist()) - {-999}):
        a = shipped & of_lot & (udate == d)
        by_date.append({"date_code": W.day_to_date(d), "units_shipped": int(a.sum()), "returns": int(returned[a].sum()), "e07_returns": int((returned & e07 & a).sum()),
                        "per_1000_unit_days": round(float(1000 * returned[a].sum() / days[a].sum()), 3)})
    return sorted(rows, key=lambda r: -(r["rate_ratio"] or 0)), by_date, int((shipped & of_lot & (udate == -999)).sum())


def explain(rec, truth, m, lot, d_from, d_to, chosen, affected):
    """Why a strategy missed affected units and recalled unaffected ones, from the simulation truth (for the evaluation only)."""
    bad_reel = np.array([W.LOT_CODES[l] == lot for l in truth["reel_lot"]]) & (truth["reel_date"] >= d_from) & (truth["reel_date"] <= d_to)
    ub, tf = m["unit_board"], truth["unit_final"]
    br = m["board_reel"]
    r1, r2 = truth["board_r1"], truth["board_r2"]

    def cause(u):
        if ub[u] != tf[u]:
            return "rework ticket without the replacement board's serial"
        b = tf[u]
        if br[b] not in (r1[b], r2[b]) or (r2[b] >= 0 and bad_reel[r1[b]] != bad_reel[r2[b]]):
            return "reel scanned late or not at all at a splice"
        r = br[b]
        if m["reel_lot"][r] != W.LOT_CODES[truth["reel_lot"][r]]:
            return "lot code misread or unread at receiving"
        if rec["reels"]["date_read"][r] is None:
            return "date code unread, counted as suspect"
        return "other"
    out = {}
    for name, sel in (("missed", affected & ~chosen), ("unaffected_recalled", chosen & ~affected)):
        counts = {}
        for u in np.flatnonzero(sel):
            k = cause(u)
            counts[k] = counts.get(k, 0) + 1
        out[name] = dict(sorted(counts.items(), key=lambda kv: -kv[1]))
    return out
