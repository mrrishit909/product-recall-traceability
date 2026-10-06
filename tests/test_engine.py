"""The simulator's invariants, the genealogy posteriors, the optimiser and the scores, without a database."""
import itertools

import numpy as np

from rc import engine as E, world as W


def test_world_invariants():
    rec, tr = W.generate(21)
    b, u = rec["boards"], rec["units"]
    mixed = tr["board_r2"] >= 0
    assert 0 < mixed.sum() < 100 and (tr["board_r2"][mixed] != tr["board_r1"][mixed]).all()     # only boards at a splice hold two reels
    assert len(set(tr["unit_final"].tolist())) == len(u["serial"]) and not b["spare"][u["board"]].any()
    assert all(b["spare"][r["old_board"]] == 0 and b["spare"][r["new_board"]] for r in tr["reworks"])
    assert (np.diff(b["t"][b["line"] == 1]) > 0).all() and (b["tp"] < b["t"]).all()
    for dl in rec["deliveries"]:                           # every reel the notice lists was received
        for code, counts in dl["lots"].items():
            assert sum(counts.values()) == (tr["reel_lot"][rec["reels"]["delivery"] == rec["deliveries"].index(dl)] == W.LOT_CODES.index(code)).sum()


def test_lot_posterior_is_exact_under_the_notice_counts():
    q = 0.02
    reads = ["B-177", "B-I77", None, "B-178", "B-17T", "B-178"]
    codes, nA = ["B-177", "B-178"], 3
    lik = np.array([[E.read_lik(s, c, q) for c in codes] for s in reads])
    brute = np.zeros(len(reads))
    for A in itertools.combinations(range(len(reads)), nA):            # every way of choosing which reels are B-177
        w = np.prod([lik[i, 0] if i in A else lik[i, 1] for i in range(len(reads))])
        brute[list(A)] += w
    brute /= sum(np.prod([lik[i, 0] if i in A else lik[i, 1] for i in range(len(reads))]) for A in itertools.combinations(range(len(reads)), nA))
    rec = {"reels": {"ref": np.array([f"R{i}" for i in range(6)]), "delivery": np.zeros(6, int), "lot_read": reads}, "deliveries": [{"lots": {"B-177": {0: 3}, "B-178": {0: 3}}}]}
    post = E.lot_posterior(rec, {"char_error": q})
    assert np.allclose(post[:, W.LOT_CODES.index("B-177")], brute)
    assert post[0, W.LOT_CODES.index("B-177")] > 0.999 and 0.3 < post[4, W.LOT_CODES.index("B-177")] < 0.7     # 'B-17T' is one misread from either


def test_optimiser_keeps_policy_and_matches_brute_force():
    rng = np.random.default_rng(0)
    p = np.r_[rng.uniform(0, 0.012, 9), 0.5, 0.0]
    cost = rng.uniform(10, 400, 11)
    inc, esc = E.optimise(p, cost, max_expected_escapes=0.02, exclude_above=0.01)
    assert inc[p > 0.01].all() and esc <= 0.02 + 1e-12 and not inc[p == 0].any()
    best = min(cost[[i for i in range(11) if i in S or p[i] > 0.01]].sum() for k in range(12) for S in itertools.combinations(range(11), k)
               if p[[i for i in range(11) if i not in S and p[i] <= 0.01]].sum() <= 0.02 and all(p[i] <= 0.01 or i in S for i in range(11)))
    assert cost[inc].sum() <= best * 1.15


def test_learned_record_models_and_the_scope_against_truth():
    rec, tr = W.generate(31)
    out = E.scope(rec, "B-177", *tr["window"])
    f = out["fit"]
    assert abs(f["boards_per_reel"] - 490) < 5 and f["splices_observed"] > 20 and 0 < f["char_error"] < 0.05
    affected = W.affected_boards(tr, "B-177", *tr["window"])[tr["unit_final"]]
    s = E.score(out["sets"], affected, out["cost"])
    assert s["model"]["affected_missed"] <= 3 and s["model"]["units"] < s["broad"]["units"] and s["model"]["cost"] < s["work_order"]["cost"]
    assert s["mes"]["affected_missed"] > s["model"]["affected_missed"]
    assert out["expected_escapes"] <= E.POLICY["max_expected_escapes"] and (out["p_unit"][~out["sets"]["model"]] <= E.POLICY["exclude_above"]).all()
