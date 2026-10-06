"""The simulated plant. Arden Drive Systems (invented) builds e-bike drive units. Each unit holds one controller board; each board
carries four DC-link capacitors (part CAP-470) from one supplier, Kestrel Components (invented), delivered in lots of 8-12 reels,
each reel printed with the day the supplier made it (its date code).

Three SMT lines draw reels from stores, oldest lot first but not always. When a reel runs out the operator splices the next one
onto its tail, so the board at the splice can hold parts from both. Every 25th board goes to a spare pool; units that fail the
end-of-line test get a spare board fitted. Units ship to eight customers.

`generate(seed)` returns two things. `records` is what the plant's systems wrote down, with the flaws real plants have: lot codes
read wrongly or not at all at receiving, smudged date codes, reel scans made minutes after the splice or forgotten, rework tickets
without the replacement board's serial. `truth` is what really happened. Models only ever see the records.
"""
import functools

import numpy as np

DAYS = 60                      # production days; day 0 is 2026-07-01
LINES = (1, 2, 3)
SENSOR_LINE = 3                # line 3's feeders have splice sensors: its splice times are logged exactly
DAY_START, DAY_END = 360, 1320  # 06:00-22:00, minutes after midnight
BOARDS_PER_DAY = 232           # per line
CAPS_PER_BOARD = 4
REEL_PARTS = 2000
SPARE_EVERY = 25
AS_OF = 61 * 1440 + 480        # 2026-08-31 08:00, when the supplier's notice arrives
LOT_CODES = [f"B-{n}" for n in range(172, 190)]
SUPPLIER, PART = "Kestrel Components", "CAP-470 DC-link capacitor"
DEFECT_LOT = "B-177"
CUSTOMERS = ["Velora Bikes", "Northline Cycles", "Ostra Mobility", "Pell & Co Distribution", "Cadence Urban", "Hallam Bike Works", "Rivetta", "Bergmark Sport"]
CUSTOMER_WEIGHTS = np.array([5, 4, 3, 3, 2, 2, 1, 1], float)
SYMPTOMS = ["E07 DC-link fault", "E03 motor sensor", "No power", "Intermittent assist", "Noise"]

# How the records go wrong. None of these numbers is given to the models; they are learned from the records where they can be.
FLAWS = {"lot_char_misread": 0.016, "lot_unread": 0.04, "date_unread": 0.08, "scan_forgotten": 0.03, "rework_serial_missing": 0.25}
SCAN_DELAY = {1: (2.0, 6.5), 2: (2.0, 6.5), 3: (2.0, 5.0)}     # gamma(shape, scale) minutes, minus 3: line 3's crew scans a little sooner
CONFUSE = {"B": "8", "8": "B", "1": "I7", "7": "1T", "0": "OD", "2": "Z", "5": "S", "-": "_"}
EOL_FAIL, EOL_FAIL_DEFECTIVE = 0.03, 0.05
RETURN_HAZARD, RETURN_HAZARD_DEFECTIVE = 0.0003, 0.0025      # per day in the field, after a week


def misread(code, rng):
    out = []
    for ch in code:
        if rng.random() < FLAWS["lot_char_misread"]:
            alt = CONFUSE.get(ch, "")
            if ch == "-" and rng.random() < 0.5:
                continue                      # a dropped hyphen
            out.append(alt[rng.integers(len(alt))] if alt else ch)
        else:
            out.append(ch)
    return "".join(out)


def stage_cost(days_since_ship):
    """Expected cost to recover one unit. In our warehouse: hold and rework ($18). Shipped under ten days: at the customer's line or
    stock ($95, freight and rework). Older: a growing share is on a rider's bike ($380, dealer retrofit and notification)."""
    d = np.asarray(days_since_ship, float)
    sold = np.clip((d - 10) / 40, 0, 0.85)
    return np.where(np.isnan(d), 18.0, (1 - sold) * 95 + sold * 380)


@functools.lru_cache(maxsize=8)
def generate(seed, spare_pick=5, slow_scans=1.0):
    """`spare_pick`: rework takes one of the N oldest spares (None: any spare on the shelf). `slow_scans` stretches lines 1-2's scan
    delays; both exist to test the models when their assumptions are wrong."""
    rng = np.random.default_rng(seed)

    # ---- supplier lots, deliveries, receipts
    deliveries = []
    plan = [[0, 1]]
    nxt = 2
    for _ in range(1, 9):
        n = 2 if rng.random() < 0.4 else 1
        plan.append(list(range(nxt, nxt + n)))
        nxt += n
    reel_lot, reel_date, reel_delivery = [], [], []
    for di, lots in enumerate(plan):
        day = -1 if di == 0 else 7 * di
        deliveries.append({"ref": f"DLV-{di + 1:02d}", "arrival": day * 1440 + 600, "lots": {}})
        for li in lots:
            code = LOT_CODES[li]
            n = int(rng.integers(8, 13))
            d0, dates = day - int(rng.integers(18, 30)), []
            while len(dates) < n:
                dates += [d0] * int(rng.integers(1, 3))
                d0 += 1
            dates = dates[:n]
            counts = {}
            for d in dates:
                counts[d] = counts.get(d, 0) + 1
            deliveries[-1]["lots"][code] = counts
            order = rng.permutation(n)
            for j in order:
                reel_lot.append(li)
                reel_date.append(dates[j])
                reel_delivery.append(di)
    R = len(reel_lot)
    reel_lot, reel_date, reel_delivery = np.array(reel_lot), np.array(reel_date), np.array(reel_delivery)
    reel_ref = np.array([f"R-{i + 1:04d}" for i in range(R)])
    lot_read = [None if rng.random() < FLAWS["lot_unread"] else misread(LOT_CODES[reel_lot[i]], rng) for i in range(R)]
    date_read = [None if rng.random() < FLAWS["date_unread"] else int(reel_date[i]) for i in range(R)]

    # ---- boards on each line
    step = (DAY_END - DAY_START) / BOARDS_PER_DAY
    lines = {}
    for L in LINES:
        j = np.arange(BOARDS_PER_DAY)
        t = np.concatenate([d * 1440 + DAY_START + (j + 1) * step + rng.uniform(-0.3, 0.3, BOARDS_PER_DAY) * step for d in range(DAYS)])
        tp = np.r_[t[0] - step, t[:-1]]
        first = np.arange(len(t)) % BOARDS_PER_DAY == 0
        tp[first] = t[first] - step          # the first board of a day starts at 06:00, not overnight
        lines[L] = {"t": t, "tp": tp, "loads": []}

    # ---- reels through the lines: splices in time order across lines, stores issue oldest lot first (mostly)
    def splice_time(L, P):
        kb, r = divmod(P, CAPS_PER_BOARD)
        t, tp = lines[L]["t"], lines[L]["tp"]
        if kb >= len(t):
            return None
        return tp[kb] + (r / CAPS_PER_BOARD) * (t[kb] - tp[kb])

    issued = np.zeros(R, bool)
    arrival = np.array([deliveries[d]["arrival"] for d in reel_delivery])
    queue = [(-60.0, L, 0) for L in LINES]          # (splice time, line, first part index of the new reel)
    while queue:
        queue.sort()
        s, L, P = queue.pop(0)
        t_issue = s - rng.uniform(30, 180) if s > 0 else -90.0
        avail = np.flatnonzero(~issued & (arrival <= t_issue))
        if not len(avail):
            raise RuntimeError("stores ran dry")
        lots_avail = sorted(set(reel_lot[avail]))
        pick_lot = lots_avail[1] if len(lots_avail) > 1 and rng.random() < 0.25 else lots_avail[0]
        cand = avail[reel_lot[avail] == pick_lot]
        r = int(cand[rng.integers(len(cand))])
        issued[r] = True
        cap = int(REEL_PARTS + rng.integers(-8, 9) - REEL_PARTS * np.clip(rng.normal(0.02, 0.006), 0.002, 0.05))
        lines[L]["loads"].append({"reel": r, "P": P, "s": s, "issued": t_issue})
        s2 = splice_time(L, P + cap)
        if s2 is not None:
            queue.append((s2, L, P + cap))

    # ---- truth: which reels each board holds (exact, from part indices)
    b_line, b_t, b_tp, b_k, b_r1, b_r2 = [], [], [], [], [], []
    for L in LINES:
        n = len(lines[L]["t"])
        loads = lines[L]["loads"]
        P = np.array([ld["P"] for ld in loads] + [10 ** 12])
        first, last = np.arange(n) * CAPS_PER_BOARD, np.arange(n) * CAPS_PER_BOARD + CAPS_PER_BOARD - 1
        i1 = np.searchsorted(P, first, "right") - 1
        i2 = np.searchsorted(P, last, "right") - 1
        reels = np.array([ld["reel"] for ld in loads])
        b_line.append(np.full(n, L))
        b_t.append(lines[L]["t"])
        b_tp.append(lines[L]["tp"])
        b_k.append(np.arange(n))
        b_r1.append(reels[i1])
        b_r2.append(np.where(i2 != i1, reels[np.minimum(i2, len(reels) - 1)], -1))
    b_line, b_t, b_tp, b_k = map(np.concatenate, (b_line, b_t, b_tp, b_k))
    b_r1, b_r2 = np.concatenate(b_r1), np.concatenate(b_r2)
    B = len(b_t)
    b_serial = np.array([f"CB{L}-{int(k) // BOARDS_PER_DAY:02d}{int(k) % BOARDS_PER_DAY:03d}" for L, k in zip(b_line, b_k)])
    b_wo = np.array([f"WO-{L}{int(k) // BOARDS_PER_DAY:02d}" for L, k in zip(b_line, b_k)])
    b_spare = b_k % SPARE_EVERY == SPARE_EVERY - 1

    dates_sorted = sorted(set(reel_date[reel_lot == LOT_CODES.index(DEFECT_LOT)].tolist()))
    window = (dates_sorted[1], dates_sorted[1] + 4)                # the supplier's notice: five days of B-177's date codes
    bad_reel = (reel_lot == LOT_CODES.index(DEFECT_LOT)) & (reel_date >= window[0]) & (reel_date <= window[1])
    b_bad = bad_reel[b_r1] | ((b_r2 >= 0) & bad_reel[np.maximum(b_r2, 0)])

    # ---- units: boards in build order, a day or so after the board; end-of-line test and rework with spares
    order = np.flatnonzero(~b_spare)
    order = order[np.argsort(b_t[order], kind="stable")]
    u_built = b_t[order] + rng.uniform(720, 2160, len(order))
    keep = u_built <= AS_OF
    u_board, u_built = order[keep], u_built[keep]
    o = np.argsort(u_built, kind="stable")
    u_board, u_built = u_board[o], u_built[o]
    U = len(u_board)
    u_serial = np.array([f"ADU-{i + 1:06d}" for i in range(U)])
    u_final = u_board.copy()
    fail = rng.random(U) < np.where(b_bad[u_board], EOL_FAIL_DEFECTIVE, EOL_FAIL)
    rw_at = u_built + rng.uniform(60, 1440, U)
    spare_ids = np.flatnonzero(b_spare)
    spare_ids = spare_ids[np.argsort(b_t[spare_ids])]
    spare_used = np.zeros(B, bool)
    reworks = []
    for u in np.flatnonzero(fail & (rw_at <= AS_OF))[np.argsort(rw_at[fail & (rw_at <= AS_OF)])]:
        pool = spare_ids[(b_t[spare_ids] < rw_at[u]) & ~spare_used[spare_ids]][:spare_pick]
        if not len(pool):
            continue
        nb = int(pool[rng.integers(len(pool))])
        spare_used[nb] = True
        u_final[u] = nb
        reworks.append({"unit": int(u), "at": float(rw_at[u]), "old_board": int(u_board[u]), "new_board": nb, "ticket_has_serial": bool(rng.random() >= FLAWS["rework_serial_missing"])})
    rw_time = np.full(U, -1.0)
    for r in reworks:
        rw_time[r["unit"]] = r["at"]

    # ---- shipping
    u_ship = np.full(U, -1)
    shipments = []
    for d in range(1, 62):
        t_ship = d * 1440 + 960
        if t_ship > AS_OF:
            break
        ready = np.flatnonzero((u_ship < 0) & (u_built <= d * 1440 + DAY_START) & (rw_time < t_ship))
        if not len(ready):
            continue
        custs = rng.choice(len(CUSTOMERS), size=int(rng.integers(2, 5)), replace=False, p=CUSTOMER_WEIGHTS / CUSTOMER_WEIGHTS.sum())
        w = CUSTOMER_WEIGHTS[custs] / CUSTOMER_WEIGHTS[custs].sum()
        who = custs[rng.choice(len(custs), size=len(ready), p=w)]
        for c in custs:
            shipments.append({"ref": f"SH-{d:02d}-{c + 1}", "customer": CUSTOMERS[c], "at": float(t_ship)})
            u_ship[ready[who == c]] = len(shipments) - 1

    # ---- field returns: the real defect raises the return rate
    defective_unit = b_bad[u_final]
    ship_t = np.array([shipments[s]["at"] if s >= 0 else np.nan for s in u_ship])
    exposure = np.clip((AS_OF - ship_t) / 1440 - 7, 0, None)
    exposure = np.nan_to_num(exposure)
    p_ret = 1 - np.exp(-np.where(defective_unit, RETURN_HAZARD_DEFECTIVE, RETURN_HAZARD) * exposure)
    ret = np.flatnonzero(rng.random(U) < p_ret)
    returns = []
    for u in ret:
        sym = SYMPTOMS[0] if defective_unit[u] and rng.random() < 0.7 else SYMPTOMS[int(rng.choice(len(SYMPTOMS), p=[.05, .3, .25, .25, .15]))]
        returns.append({"unit": int(u), "at": float(ship_t[u] + 7 * 1440 + rng.uniform(0, exposure[u]) * 1440), "symptom": sym})
    returns.sort(key=lambda r: r["at"])

    # ---- the records the plant wrote down
    issues, scans, sensor = [], [], []
    for L in LINES:
        for i, ld in enumerate(lines[L]["loads"]):
            issues.append({"reel": ld["reel"], "line": L, "at": float(ld["issued"])})
            if i == 0:
                scans.append({"reel": ld["reel"], "line": L, "at": float(DAY_START - 10)})
                continue
            if L == SENSOR_LINE:
                sensor.append({"reel": ld["reel"], "line": L, "at": float(ld["s"])})
            elif rng.random() < FLAWS["scan_forgotten"]:
                continue
            k_, th = SCAN_DELAY[L]
            scans.append({"reel": ld["reel"], "line": L, "at": float(ld["s"] + (rng.gamma(k_, th) - 3) * (slow_scans if L != SENSOR_LINE else 1))})
    issues.sort(key=lambda e: e["at"])
    scans.sort(key=lambda e: e["at"])

    records = {
        "seed": seed, "as_of": AS_OF,
        "deliveries": deliveries,
        "reels": {"ref": reel_ref, "delivery": reel_delivery, "lot_read": lot_read, "date_read": date_read, "received": arrival.astype(float)},
        "issues": issues, "scans": scans, "sensor": sensor,
        "boards": {"serial": b_serial, "line": b_line, "t": b_t, "tp": b_tp, "spare": b_spare, "wo": b_wo},
        "units": {"serial": u_serial, "board": u_board, "built": u_built},
        "reworks": [{"unit": r["unit"], "at": r["at"], "new_board": r["new_board"] if r["ticket_has_serial"] else None} for r in reworks],
        "shipments": shipments, "unit_shipment": u_ship,
        "returns": [{"unit": r["unit"], "at": r["at"], "symptom": r["symptom"]} for r in returns],
    }
    truth = {"reel_lot": reel_lot, "reel_date": reel_date, "board_r1": b_r1, "board_r2": b_r2, "unit_final": u_final,
             "window": window, "lines": {L: lines[L]["loads"] for L in LINES}, "reworks": reworks}
    return records, truth


def affected_boards(truth, lot, d_from, d_to):
    """Truth for a notice: boards holding at least one part from a reel of `lot` with a date code in [d_from, d_to]."""
    bad = (truth["reel_lot"] == LOT_CODES.index(lot)) & (truth["reel_date"] >= d_from) & (truth["reel_date"] <= d_to) if lot in LOT_CODES else np.zeros(len(truth["reel_lot"]), bool)
    r2 = truth["board_r2"]
    return bad[truth["board_r1"]] | ((r2 >= 0) & bad[np.maximum(r2, 0)])


def day_to_date(d):
    import datetime
    return (datetime.date(2026, 7, 1) + datetime.timedelta(days=int(d))).isoformat()


def date_to_day(s):
    import datetime
    return (datetime.date.fromisoformat(str(s)) - datetime.date(2026, 7, 1)).days
