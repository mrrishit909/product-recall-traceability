"""Public API (modular monolith). Blueprint services map to: ERP/MES connectors (the plant history load and the assembly-event
endpoint, each event checked on its own), serialization (boards and drive units), genealogy graph (reel -> lot and reel -> board
edges with probabilities), quality events (field returns and the supplier's notice), recall engine (the scope job and its
optimiser), customer trace (shipments and the notices), supplier portal (the advance ship notices the reels are matched to) and the
audit ledger (the kit's hash chain).

    uvicorn rc.api:app          python -m core.jobs rc.api      # the worker
"""
import csv
import datetime
import hashlib
import io
import time
import uuid

import numpy as np
from fastapi import Depends, Header
from fastapi.encoders import jsonable_encoder
from psycopg.types.json import Jsonb
from pydantic import BaseModel, Field

from core import audit, db, jobs
from core.app import Ctx, Problem, create_app, run

from . import engine as E, world as W

READ = {"plant:read", "jobs:read"}
WORK = READ | {"events:write", "incidents:write", "recalls:scope", "notices:export"}
PERMISSIONS = {"viewer": READ, "quality_engineer": WORK, "head_of_quality": WORK | {"plant:generate", "recalls:approve", "audit:read"}}
app = create_app("product-recall-traceability", PERMISSIONS)
auth = app.state.auth
IdemKey = Header(None, alias="Idempotency-Key")
START = datetime.datetime(2026, 7, 1, tzinfo=datetime.UTC)
STEP = (W.DAY_END - W.DAY_START) / W.BOARDS_PER_DAY


def ts(m):
    return START + datetime.timedelta(minutes=float(m))


def mins(t):
    return (t - START).total_seconds() / 60


def sid(t, kind, ref):
    return uuid.uuid5(t, f"{kind}-{ref}")


def worker_ctx(job):
    return Ctx(job["tenant_id"], uuid.UUID(job["payload"]["actor_id"]), "worker", "system")


def plant(c):
    p = c.execute("SELECT * FROM plant").fetchone()
    if not p:
        raise Problem(409, "no_plant", "load the plant history first: POST /v1/events/assembly with generate")
    return p


def money(v):
    return round(float(v))


# ---------------------------------------------------------------- ERP / MES connectors

class Generate(BaseModel):
    seed: int = Field(11, ge=0, le=10 ** 6)


class Event(BaseModel):
    event_id: str = Field(min_length=1, max_length=60, description="the source system's id; a replay is ignored")
    kind: str = Field(pattern="^(unit_build|rework|reel_scan)$")
    at: datetime.datetime
    serial: str | None = Field(None, max_length=40)
    board_serial: str | None = Field(None, max_length=40)
    reel: str | None = Field(None, max_length=40)
    line: int | None = Field(None, ge=1, le=9)


class EventsIn(BaseModel):
    generate: Generate | None = None
    events: list[Event] | None = Field(None, max_length=2000)


@app.post("/v1/events/assembly", status_code=202, tags=["connectors"], summary="Assembly events from the MES. `generate` loads sixty days of the synthetic plant's history in a background job; `events` records new events, each checked on its own (duplicates, unknown parts, out-of-order, boards already fitted)")
def assembly_events(body: EventsIn, ctx: Ctx = Depends(auth("events:write")), idem: str | None = IdemKey):
    if (body.generate is None) == (body.events is None):
        raise Problem(422, "generate_or_events", "send exactly one of generate or events")
    if body.generate and "plant:generate" not in PERMISSIONS[ctx.role]:
        raise Problem(403, "forbidden", "loading a synthetic plant needs the head of quality")

    def work(c):
        if body.generate:
            if c.execute("SELECT 1 FROM plant").fetchone():
                raise Problem(409, "plant_exists")
            return 202, jobs.enqueue(c, ctx, "plant.generate", body.generate.model_dump())
        p = plant(c)
        accepted, rejected = [], []
        for e in body.events:
            why = check_event(c, p, e)
            if why:
                rejected.append({"event_id": e.event_id, "kind": e.kind, "why": why})
                continue
            if e.kind == "unit_build":       # a build is the unit's own record
                c.execute("INSERT INTO serial_unit (unit_id, tenant_id, serial, kind, built_at, board_serial) VALUES (%s,%s,%s,'drive_unit',%s,%s)", [sid(ctx.tenant_id, "sn", e.serial), ctx.tenant_id, e.serial, e.at, e.board_serial])
            else:
                c.execute("INSERT INTO assembly_event (tenant_id, event_id, kind, at, line, reel_ref, serial, board_serial, detail) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                          [ctx.tenant_id, e.event_id, e.kind, e.at, e.line, e.reel, e.serial, e.board_serial, Jsonb({"source": "api"})])
            accepted.append({"event_id": e.event_id, "kind": e.kind, "serial": e.serial, "board_serial": e.board_serial})
        audit.record(c, ctx, "events.recorded", "assembly_event", f"{len(accepted)} events", {"accepted": len(accepted), "rejected": len(rejected)})
        return 201, {"accepted": accepted, "rejected": rejected}
    return run(ctx, idem, body, work)


def check_event(c, p, e):
    """-> the reason an event is refused, or None."""
    if c.execute("SELECT 1 FROM assembly_event WHERE event_id = %s", [e.event_id]).fetchone() or (e.kind == "unit_build" and c.execute("SELECT 1 FROM serial_unit WHERE serial = %s AND kind = 'drive_unit' AND built_at = %s", [e.serial, e.at]).fetchone()):
        return "duplicate: this event is already recorded"
    if e.at.tzinfo is None:
        return "timestamp without a time zone"
    if e.at > p["as_of"] + datetime.timedelta(hours=1):
        return "timestamp is after the plant's clock"
    board = None
    if e.kind in ("unit_build", "rework") and e.board_serial:
        board = c.execute("SELECT serial, built_at FROM serial_unit WHERE serial = %s AND kind = 'controller_board'", [e.board_serial]).fetchone()
        if not board:
            return f"unknown board {e.board_serial}"
        fitted = c.execute("""SELECT serial FROM serial_unit WHERE board_serial = %s AND kind = 'drive_unit'
                              UNION ALL SELECT serial FROM assembly_event WHERE kind = 'rework' AND board_serial = %s LIMIT 1""", [e.board_serial, e.board_serial]).fetchone()
        if fitted:
            return f"board {e.board_serial} is already fitted to {fitted['serial']}"
        if board["built_at"] > e.at:
            return "out of order: the board was built after this event"
    if e.kind == "unit_build":
        if not e.serial or not e.board_serial:
            return "a unit build needs serial and board_serial"
        if c.execute("SELECT 1 FROM serial_unit WHERE serial = %s", [e.serial]).fetchone():
            return f"serial {e.serial} already exists"
    elif e.kind == "rework":
        unit = c.execute("SELECT built_at FROM serial_unit WHERE serial = %s AND kind = 'drive_unit'", [e.serial or ""]).fetchone()
        if not unit:
            return f"unknown unit {e.serial}"
        if unit["built_at"] > e.at:
            return "out of order: rework before the unit was built"
    else:
        if not e.reel or not e.line:
            return "a reel scan needs reel and line"
        if not c.execute("SELECT 1 FROM assembly_event WHERE kind = 'reel_issue' AND reel_ref = %s AND line = %s", [e.reel, e.line]).fetchone():
            return f"reel {e.reel} was never issued to line {e.line}"
    return None


@jobs.handler("plant.generate")
def generate_job(c, job):
    t, seed = job["tenant_id"], job["payload"]["seed"]
    rec, _ = W.generate(seed)
    t0 = time.perf_counter()
    c.execute("INSERT INTO plant (plant_id, tenant_id, name, seed, as_of, metadata) VALUES (%s,%s,%s,%s,%s,%s)",
              [sid(t, "plant", "1"), t, "Arden drive-unit plant (synthetic)", seed, ts(rec["as_of"]), Jsonb({"lines": len(W.LINES), "sensor_line": W.SENSOR_LINE, "part": W.PART, "supplier": W.SUPPLIER, "customers": W.CUSTOMERS})])
    db.load(c, "material_batch", ["material_batch_id", "tenant_id", "external_ref", "name", "metadata"],
            [(sid(t, "lot", code), t, code, W.PART, Jsonb({"supplier": W.SUPPLIER, "delivery": dl["ref"], "arrival": ts(dl["arrival"]).isoformat(), "reels_by_date_code": {W.day_to_date(d): n for d, n in sorted(counts.items())}}))
             for dl in rec["deliveries"] for code, counts in dl["lots"].items()])
    r = rec["reels"]
    db.load(c, "component_lot", ["component_lot_id", "tenant_id", "status", "observed_at", "external_ref", "delivery_ref", "lot_code_read", "date_code_read"],
            [(sid(t, "reel", r["ref"][i]), t, "received", ts(r["received"][i]), str(r["ref"][i]), rec["deliveries"][r["delivery"][i]]["ref"], r["lot_read"][i], W.day_to_date(r["date_read"][i]) if r["date_read"][i] is not None else None) for i in range(len(r["ref"]))])
    b, u = rec["boards"], rec["units"]
    rows = [(sid(t, "sn", b["serial"][i]), t, str(b["serial"][i]), "controller_board", int(b["line"][i]), ts(b["t"][i]), str(b["wo"][i]), None, bool(b["spare"][i])) for i in range(len(b["t"]))]
    rows += [(sid(t, "sn", u["serial"][i]), t, str(u["serial"][i]), "drive_unit", None, ts(u["built"][i]), None, str(b["serial"][u["board"][i]]), False) for i in range(len(u["serial"]))]
    db.load(c, "serial_unit", ["unit_id", "tenant_id", "serial", "kind", "line", "built_at", "work_order", "board_serial", "spare"], rows)
    ev = [(t, f"ISS-{r['ref'][e['reel']]}", "reel_issue", ts(e["at"]), e["line"], str(r["ref"][e["reel"]]), None, None) for e in rec["issues"]]
    ev += [(t, f"SCN-{r['ref'][e['reel']]}", "reel_scan", ts(e["at"]), e["line"], str(r["ref"][e["reel"]]), None, None) for e in rec["scans"]]
    ev += [(t, f"SPL-{r['ref'][e['reel']]}", "splice_sensor", ts(e["at"]), e["line"], str(r["ref"][e["reel"]]), None, None) for e in rec["sensor"]]
    ev += [(t, f"RWK-{u['serial'][e['unit']]}", "rework", ts(e["at"]), None, None, str(u["serial"][e["unit"]]), str(b["serial"][e["new_board"]]) if e["new_board"] is not None else None) for e in rec["reworks"]]
    db.load(c, "assembly_event", ["tenant_id", "event_id", "kind", "at", "line", "reel_ref", "serial", "board_serial"], ev)
    db.load(c, "shipment", ["shipment_id", "tenant_id", "external_ref", "customer", "shipped_at"], [(sid(t, "ship", s["ref"]), t, s["ref"], s["customer"], ts(s["at"])) for s in rec["shipments"]])
    us = rec["unit_shipment"]
    db.load(c, "customer_delivery", ["unit_id", "shipment_id", "tenant_id"], [(sid(t, "sn", u["serial"][i]), sid(t, "ship", rec["shipments"][us[i]]["ref"]), t) for i in np.flatnonzero(us >= 0)])
    db.load(c, "quality_event", ["quality_event_id", "tenant_id", "kind", "serial", "at", "detail"],
            [(sid(t, "return", f"{u['serial'][e['unit']]}-{k}"), t, "field_return", str(u["serial"][e["unit"]]), ts(e["at"]), Jsonb({"symptom": e["symptom"]})) for k, e in enumerate(rec["returns"])])
    audit.record(c, worker_ctx(job), "plant.loaded", "plant", seed, {"boards": len(b["t"]), "units": len(u["serial"]), "reels": len(r["ref"]), "events": len(ev)})
    return {"boards": len(b["t"]), "units": len(u["serial"]), "reels": len(r["ref"]), "lots": sum(len(dl["lots"]) for dl in rec["deliveries"]), "events": len(ev),
            "shipments": len(rec["shipments"]), "returns": len(rec["returns"]), "seconds": round(time.perf_counter() - t0, 1)}


def records_from_db(c):
    """The records exactly as the engine reads them, rebuilt from the tables."""
    p = plant(c)
    lots = c.execute("SELECT external_ref, metadata FROM material_batch ORDER BY metadata->>'arrival', external_ref").fetchall()
    deliveries, dix = [], {}
    for lt in lots:
        m = lt["metadata"]
        if m["delivery"] not in dix:
            dix[m["delivery"]] = len(deliveries)
            deliveries.append({"ref": m["delivery"], "arrival": mins(datetime.datetime.fromisoformat(m["arrival"])), "lots": {}})
        deliveries[dix[m["delivery"]]]["lots"][lt["external_ref"]] = {W.date_to_day(d): n for d, n in m["reels_by_date_code"].items()}
    reels = c.execute("SELECT external_ref, delivery_ref, lot_code_read, date_code_read, observed_at FROM component_lot ORDER BY external_ref").fetchall()
    rix = {r["external_ref"]: i for i, r in enumerate(reels)}
    boards = c.execute("SELECT serial, line, built_at, work_order, spare FROM serial_unit WHERE kind = 'controller_board' ORDER BY line, built_at").fetchall()
    bix = {r["serial"]: i for i, r in enumerate(boards)}
    bt = np.array([mins(r["built_at"]) for r in boards])
    wo = np.array([r["work_order"] for r in boards])
    tp = np.r_[bt[0] - STEP, bt[:-1]]
    first = np.r_[True, wo[1:] != wo[:-1]]
    tp[first] = bt[first] - STEP
    units = c.execute("SELECT serial, board_serial, built_at FROM serial_unit WHERE kind = 'drive_unit' ORDER BY built_at, serial").fetchall()
    uix = {r["serial"]: i for i, r in enumerate(units)}
    events = c.execute("SELECT kind, at, line, reel_ref, serial, board_serial FROM assembly_event ORDER BY at, event_id").fetchall()
    ships = c.execute("SELECT shipment_id, external_ref, customer, shipped_at FROM shipment ORDER BY shipped_at, external_ref").fetchall()
    six = {r["shipment_id"]: i for i, r in enumerate(ships)}
    us = np.full(len(units), -1)
    for r in c.execute("SELECT su.serial, cd.shipment_id FROM customer_delivery cd JOIN serial_unit su USING (unit_id)"):
        us[uix[r["serial"]]] = six[r["shipment_id"]]
    rets = c.execute("SELECT serial, at, detail FROM quality_event WHERE kind = 'field_return' ORDER BY at, serial").fetchall()
    ev = lambda kind: [{"reel": rix[e["reel_ref"]], "line": e["line"], "at": mins(e["at"])} for e in events if e["kind"] == kind]     # noqa: E731
    return {"seed": p["seed"], "as_of": mins(p["as_of"]), "deliveries": deliveries,
            "reels": {"ref": np.array([r["external_ref"] for r in reels]), "delivery": np.array([dix[r["delivery_ref"]] for r in reels]), "lot_read": [r["lot_code_read"] for r in reels],
                      "date_read": [W.date_to_day(r["date_code_read"]) if r["date_code_read"] else None for r in reels], "received": np.array([mins(r["observed_at"]) for r in reels])},
            "issues": ev("reel_issue"), "scans": ev("reel_scan"), "sensor": ev("splice_sensor"),
            "boards": {"serial": np.array([r["serial"] for r in boards]), "line": np.array([r["line"] for r in boards]), "t": bt, "tp": tp, "spare": np.array([r["spare"] for r in boards]), "wo": wo},
            "units": {"serial": np.array([r["serial"] for r in units]), "board": np.array([bix[r["board_serial"]] for r in units]), "built": np.array([mins(r["built_at"]) for r in units])},
            "reworks": [{"unit": uix[e["serial"]], "at": mins(e["at"]), "new_board": bix[e["board_serial"]] if e["board_serial"] else None} for e in events if e["kind"] == "rework"],
            "shipments": [{"ref": s["external_ref"], "customer": s["customer"], "at": mins(s["shipped_at"])} for s in ships], "unit_shipment": us,
            "returns": [{"unit": uix[r["serial"]], "at": mins(r["at"]), "symptom": r["detail"]["symptom"]} for r in rets]}


def truth_for(rec):
    """The simulation truth aligned to the records (units added through the API keep the board they were recorded with)."""
    wrec, tr = W.generate(rec["seed"])
    wu = {s: i for i, s in enumerate(wrec["units"]["serial"])}
    final = np.array([tr["unit_final"][wu[s]] if s in wu else rec["units"]["board"][i] for i, s in enumerate(rec["units"]["serial"])])
    for r in rec["reworks"]:
        if rec["units"]["serial"][r["unit"]] not in wu and r["new_board"] is not None:
            final[r["unit"]] = r["new_board"]
    return {**tr, "unit_final": final}


@app.get("/v1/plant", tags=["connectors"], summary="(+) What the plant's systems hold, and how good the records are")
def get_plant(ctx: Ctx = Depends(auth("plant:read"))):
    with db.tx(ctx.tenant_id) as c:
        p = plant(c)
        q = lambda sql: c.execute(sql).fetchone()["n"]     # noqa: E731
        lots = {r["external_ref"]: r["metadata"] for r in c.execute("SELECT external_ref, metadata FROM material_batch")}
        reels = c.execute("SELECT lot_code_read, date_code_read, delivery_ref FROM component_lot").fetchall()
        issued = q("SELECT count(*) AS n FROM assembly_event WHERE kind = 'reel_issue'")
        by_line = c.execute("""SELECT i.line, count(*) AS issued, count(s.event_id) AS scanned, count(p.event_id) AS sensed FROM assembly_event i
                                 LEFT JOIN assembly_event s ON s.kind = 'reel_scan' AND s.reel_ref = i.reel_ref AND s.line = i.line
                                 LEFT JOIN assembly_event p ON p.kind = 'splice_sensor' AND p.reel_ref = i.reel_ref
                                WHERE i.kind = 'reel_issue' GROUP BY i.line ORDER BY i.line""").fetchall()
        out = {"plant": p["name"], "as_of": p["as_of"], "production": {"from": ts(0).date(), "days": W.DAYS, "lines": len(W.LINES), "sensor_line": W.SENSOR_LINE},
               "boards": q("SELECT count(*) AS n FROM serial_unit WHERE kind = 'controller_board'"), "spare_boards": q("SELECT count(*) AS n FROM serial_unit WHERE kind = 'controller_board' AND spare"),
               "units": q("SELECT count(*) AS n FROM serial_unit WHERE kind = 'drive_unit'"), "units_shipped": q("SELECT count(*) AS n FROM customer_delivery"),
               "shipments": q("SELECT count(*) AS n FROM shipment"), "customers": q("SELECT count(DISTINCT customer) AS n FROM shipment"),
               "supplier": W.SUPPLIER, "part": W.PART, "lots": len(lots), "deliveries": len({m["delivery"] for m in lots.values()}), "reels_received": len(reels), "reels_issued": issued,
               "reworks": q("SELECT count(*) AS n FROM assembly_event WHERE kind = 'rework'"), "field_returns": q("SELECT count(*) AS n FROM quality_event WHERE kind = 'field_return'"),
               "record_quality": {"lot_code_unread": sum(r["lot_code_read"] is None for r in reels),
                                  "lot_code_matches_no_notice": sum(r["lot_code_read"] is not None and r["lot_code_read"] not in lots for r in reels),
                                  "date_code_unread": sum(r["date_code_read"] is None for r in reels),
                                  "reel_changes_without_scan": sum(r["issued"] - r["scanned"] - 1 for r in by_line if r["line"] != W.SENSOR_LINE),
                                  "reworks_without_board_serial": q("SELECT count(*) AS n FROM assembly_event WHERE kind = 'rework' AND board_serial IS NULL")},
               "by_line": [{"line": r["line"], "reels": r["issued"], "scanned": r["scanned"], "splice_sensor": r["sensed"] > 0} for r in by_line]}
    return jsonable_encoder(out)


# ---------------------------------------------------------------- quality events

class IncidentIn(BaseModel):
    supplier_lot: str = Field(min_length=1, max_length=20)
    date_code_from: datetime.date
    date_code_to: datetime.date
    source: str = Field("supplier notice", max_length=120)
    description: str = Field("", max_length=1000)


@app.post("/v1/quality/incidents", status_code=201, tags=["quality"], summary="Record a supplier's defect notice for a lot and a window of date codes; returns the field-return evidence per lot and per date code")
def incident(body: IncidentIn, ctx: Ctx = Depends(auth("incidents:write")), idem: str | None = IdemKey):
    def work(c):
        plant(c)
        lot = c.execute("SELECT metadata FROM material_batch WHERE external_ref = %s", [body.supplier_lot]).fetchone()
        if not lot:
            raise Problem(404, "unknown_lot", f"no advance ship notice lists lot {body.supplier_lot}")
        if body.date_code_to < body.date_code_from:
            raise Problem(422, "window_reversed")
        if (body.date_code_to - body.date_code_from).days > 30:
            raise Problem(422, "window_too_wide", "a notice covers at most 31 days of date codes")
        codes = lot["metadata"]["reels_by_date_code"]
        in_window = {d: n for d, n in codes.items() if body.date_code_from.isoformat() <= d <= body.date_code_to.isoformat()}
        if not in_window:
            raise Problem(422, "window_outside_lot", f"lot {body.supplier_lot} has date codes {min(codes)} to {max(codes)}")
        iid = sid(ctx.tenant_id, "incident", f"{body.supplier_lot}-{body.date_code_from}-{body.date_code_to}")
        if c.execute("SELECT 1 FROM quality_event WHERE quality_event_id = %s", [iid]).fetchone():
            raise Problem(409, "incident_exists", str(iid))
        rec = records_from_db(c)
        m = E.mes(rec)
        by_lot, by_date, unread = E.returns_by_lot(rec, m, body.supplier_lot)
        reels = c.execute("SELECT external_ref, lot_code_read, date_code_read, delivery_ref FROM component_lot WHERE delivery_ref = %s ORDER BY external_ref", [lot["metadata"]["delivery"]]).fetchall()
        detail = {"supplier_lot": body.supplier_lot, "date_code_from": body.date_code_from, "date_code_to": body.date_code_to, "source": body.source, "description": body.description, "notice_reels_in_window": sum(in_window.values())}
        c.execute("INSERT INTO quality_event (quality_event_id, tenant_id, kind, at, detail, created_by) VALUES (%s,%s,'supplier_notice',%s,%s,%s)", [iid, ctx.tenant_id, ts(rec["as_of"]), Jsonb(jsonable_encoder(detail)), ctx.actor_id])
        audit.record(c, ctx, "incident.recorded", "quality_event", iid, {"lot": body.supplier_lot, "from": str(body.date_code_from), "to": str(body.date_code_to)})
        return 201, {"incident_id": iid, **detail, "notice": {"delivery": lot["metadata"]["delivery"], "reels_by_date_code": codes, "reels_in_window": sum(in_window.values()), "reels_in_lot": sum(codes.values())},
                     "delivery_reels": [{"reel": r["external_ref"], "lot_code_read": r["lot_code_read"], "date_code_read": r["date_code_read"]} for r in reels],
                     "returns_by_lot": by_lot, "returns_by_date_code": by_date, "returns_lot_units_with_unread_date": unread,
                     "how": "units shipped and units returned, grouped by the lot the MES links each unit's capacitors to (exact read of the lot code); rate ratio against every other unit with a 90% interval; the symptom E07 is the DC-link fault"}
    return run(ctx, idem, body, work)


# ---------------------------------------------------------------- recall engine

class ScopeIn(BaseModel):
    incident_id: uuid.UUID
    max_expected_escapes: float = Field(E.POLICY["max_expected_escapes"], ge=0, le=50, description="affected units the recall may expect to leave out")
    exclude_above: float = Field(E.POLICY["exclude_above"], ge=0, le=0.5, description="no unit with a higher probability is left out")


@app.post("/v1/recalls/scope", status_code=202, tags=["recalls"], summary="Compute the minimum defensible recall set for a notice in a background job: genealogy with probabilities, the cost of getting each unit back, the optimiser, and the plant's own deterministic answers alongside")
def recall_scope(body: ScopeIn, ctx: Ctx = Depends(auth("recalls:scope")), idem: str | None = IdemKey):
    def work(c):
        inc = c.execute("SELECT * FROM quality_event WHERE quality_event_id = %s AND kind = 'supplier_notice'", [body.incident_id]).fetchone()
        if not inc:
            raise Problem(404, "incident_not_found")
        rid = sid(ctx.tenant_id, "recall", f"{body.incident_id}-{body.max_expected_escapes}-{body.exclude_above}")
        if c.execute("SELECT 1 FROM recall_case WHERE recall_id = %s", [rid]).fetchone():
            raise Problem(409, "recall_exists", str(rid))
        c.execute("INSERT INTO recall_case (recall_id, tenant_id, incident_id, status, params, created_by) VALUES (%s,%s,%s,'scoping',%s,%s)", [rid, ctx.tenant_id, body.incident_id, Jsonb(body.model_dump(mode="json")), ctx.actor_id])
        return 202, {**jobs.enqueue(c, ctx, "recall.scope", {"recall_id": str(rid), **body.model_dump(mode="json")}), "recall_id": rid}
    return run(ctx, idem, body, work)


def examples(rec, out, sets, lot):
    """Units worth tracing on the page: one at an uncertain splice, one whose rework ticket lost the board, one on a misread reel."""
    p, final = out["p_unit"], out["final"]
    unc = np.flatnonzero((p > 0.2) & (p < 0.8) & (final >= 0))
    ex = {}
    if len(unc):
        ex["splice"] = str(rec["units"]["serial"][unc[np.argmin(np.abs(p[unc] - 0.5))]])
    rw = [u for u in out["unknown"] if 0.05 < p[u] < 0.95] or list(out["unknown"])
    if rw:
        ex["rework"] = str(rec["units"]["serial"][max(rw, key=lambda u: p[u])])
    m = E.mes(rec)
    off = np.flatnonzero(sets["model"] & ~sets["mes"] & (p > 0.95) & (final >= 0))
    misread = [u for u in off if (r := m["board_reel"][final[u]]) >= 0 and m["reel_lot"][r] != lot and W.LOT_CODES[out["post"][r].argmax()] == lot]      # a B-177 reel whose code was misread
    if len(off):
        ex["mes_missed"] = str(rec["units"]["serial"][(misread or off)[0]])
    return ex


@jobs.handler("recall.scope")
def scope_job(c, job):
    pl = job["payload"]
    t, rid = job["tenant_id"], uuid.UUID(pl["recall_id"])
    inc = c.execute("SELECT detail FROM quality_event WHERE quality_event_id = %s", [pl["incident_id"]]).fetchone()["detail"]
    lot, d0, d1 = inc["supplier_lot"], W.date_to_day(inc["date_code_from"]), W.date_to_day(inc["date_code_to"])
    t0 = time.perf_counter()
    rec = records_from_db(c)
    secs_load = time.perf_counter() - t0
    out = E.scope(rec, lot, d0, d1, max_expected_escapes=pl["max_expected_escapes"], exclude_above=pl["exclude_above"])
    secs = time.perf_counter() - t0
    f, sets, p, cost = out["fit"], out["sets"], out["p_unit"], out["cost"]
    ver = f"{E.VERSION}-{rid.hex[:8]}"
    mid = uuid.uuid5(t, f"model-{ver}")
    c.execute("INSERT INTO model_artifact (model_id, tenant_id, model_name, version, artifact_uri, feature_schema, metrics, params) VALUES (%s,%s,'genealogy',%s,%s,%s,%s,%s)",
              [mid, t, ver, f"db://genealogy_edge/{ver}", Jsonb({"inputs": ["read lot codes", "advance ship notices", "reel issue log", "reel scans", "line 3 splice sensor", "board times", "rework tickets"]}),
               Jsonb({"splices_observed": f["splices_observed"], "chars_checked": f["chars_checked"]}), Jsonb({k: v for k, v in f.items() if k != "delays"} | {"delays_n": len(f["delays"])})])
    refs, bser = rec["reels"]["ref"], rec["boards"]["serial"]
    lot_edges = [(t, ver, "reel_of_lot", W.LOT_CODES[j], str(refs[i]), float(out["post"][i, j])) for i, j in zip(*np.nonzero(out["post"] > 1e-6))]
    bi, ri, pi, splices = out["pairs"]
    board_edges = [(t, ver, "reel_in_board", str(refs[r]), str(bser[b]), min(float(p_), 1.0)) for b, r, p_ in zip(bi, ri, pi)]
    unit_edges = [(t, ver, "board_in_unit", str(bser[b]), str(rec["units"]["serial"][u]), 1 / len(pool)) for u, pool in out["unknown"].items() for b in pool]
    db.load(c, "genealogy_edge", ["tenant_id", "model_version", "relation", "parent_ref", "child_ref", "probability"], lot_edges + board_edges + unit_edges)
    any_set = sets["model"] | sets["mes"] | sets["work_order"] | sets["broad"] | (p > 0)
    stage = E.stage_names(rec, out["ship_t"])
    us = rec["unit_shipment"]
    ships = rec["shipments"]
    db.load(c, "recall_scope", ["recall_id", "unit_id", "tenant_id", "serial", "p_affected", "cost", "customer", "shipment_ref", "stage", "in_model", "in_mes", "in_work_order", "in_broad"],
            [(rid, sid(t, "sn", rec["units"]["serial"][u]), t, str(rec["units"]["serial"][u]), float(p[u]), float(cost[u]), ships[us[u]]["customer"] if us[u] >= 0 else None, ships[us[u]]["ref"] if us[u] >= 0 else None,
              str(stage[u]), bool(sets["model"][u]), bool(sets["mes"][u]), bool(sets["work_order"][u]), bool(sets["broad"][u])) for u in np.flatnonzero(any_set)])
    # the simulation truth, for judging only
    tr = truth_for(rec)
    affected = W.affected_boards(tr, lot, d0, d1)[tr["unit_final"]]
    scored = E.score(sets, affected, cost)
    expl = E.explain(rec, tr, E.mes(rec), lot, d0, d1, sets["mes"], affected)
    truth_params = {"char_error": W.FLAWS["lot_char_misread"], "scan_delay_mean_lines_1_2": round(2 * 6.5 - 3, 1), "scan_delay_mean_line_3": round(2 * 5.0 - 3, 1)}
    metrics = {"lot": lot, "window": [inc["date_code_from"], inc["date_code_to"]], "units": int(len(p)), "seconds": round(secs, 2), "seconds_loading_records": round(secs_load, 2),
               "expected_escapes": out["expected_escapes"], "policy": {"max_expected_escapes": pl["max_expected_escapes"], "exclude_above": pl["exclude_above"]},
               "strategies": {k: {"units": int(v.sum()), "cost": float(cost[v].sum())} for k, v in sets.items()},
               "uncertain_units": int(((p > 0.01) & (p < 0.99)).sum()), "certain_units": int((p >= 0.99).sum()),
               "fit": {k: v for k, v in f.items() if k != "delays"}, "delays": f["delays"], "splices": {k: sum(1 for s in splices if s["kind"] == k) for k in ("sensor", "scan", "count")},
               "window_reels": [{"reel": str(refs[i]), "p": round(float(out["window_reels"][i]), 4), "lot_code_read": rec["reels"]["lot_read"][i], "date_code_read": W.day_to_date(rec["reels"]["date_read"][i]) if rec["reels"]["date_read"][i] is not None else None} for i in np.flatnonzero(out["window_reels"] > 1e-4)],
               "examples": examples(rec, out, sets, lot), "model_version": ver,
               "simulation_truth": {"what": "which units really hold parts from the window's reels, from the simulator; no model or baseline sees it", "affected": int(affected.sum()), "scored": scored, "mes_explained": expl,
                                    "calibration": E.calibration(p, affected), "record_flaws": truth_params}}
    metrics = jsonable_encoder(metrics)
    c.execute("UPDATE recall_case SET status = 'proposed', metrics = %s WHERE recall_id = %s", [Jsonb(metrics), rid])
    c.execute("INSERT INTO decision_record (decision_id, tenant_id, subject_type, subject_id, decision_type, model_id, inputs_hash, result, confidence, policy_state) VALUES (%s,%s,'recall_case',%s,'recall_scope',%s,%s,%s,%s,%s)",
              [uuid.uuid5(t, f"decision-{rid}"), t, rid, mid, hashlib.sha256(f"{lot}{d0}{d1}{len(p)}{len(rec['reworks'])}{len(rec['scans'])}".encode()).hexdigest(),
               Jsonb({"units": int(sets["model"].sum()), "cost": money(cost[sets["model"]].sum())}), 1 - min(1.0, out["expected_escapes"]), Jsonb(metrics["policy"] | {"approval": "required"})])
    audit.record(c, worker_ctx(job), "recall.scoped", "recall_case", rid, {"units": int(sets["model"].sum()), "cost": money(cost[sets["model"]].sum()), "expected_escapes": round(out["expected_escapes"], 3)})
    return {"recall_id": rid, "units": int(sets["model"].sum()), "cost": money(cost[sets["model"]].sum()), "expected_escapes": round(out["expected_escapes"], 3), "seconds": round(secs, 2)}


def recall(c, rid):
    r = c.execute("SELECT * FROM recall_case WHERE recall_id = %s", [rid]).fetchone()
    if not r:
        raise Problem(404, "recall_not_found")
    if r["status"] == "scoping":
        raise Problem(409, "recall_not_ready", "the scope job is still running")
    return r


@app.get("/v1/recalls/{recall_id}/impact", tags=["recalls"], summary="The recall set against every alternative: units, cost, customers and shipments, the uncertain units, the evidence and the model that produced it")
def impact(recall_id: uuid.UUID, ctx: Ctx = Depends(auth("plant:read"))):
    with db.tx(ctx.tenant_id) as c:
        r = recall(c, recall_id)
        m = r["metrics"]
        cust = c.execute("""SELECT coalesce(customer, 'In our warehouse') AS customer, count(*) AS units, sum(cost) AS cost, count(DISTINCT shipment_ref) AS shipments,
                                   count(*) FILTER (WHERE in_broad) AS broad_units FROM recall_scope WHERE recall_id = %s AND in_model GROUP BY 1 ORDER BY 2 DESC""", [recall_id]).fetchall()
        stages = c.execute("SELECT stage, count(*) AS units, sum(cost) AS cost FROM recall_scope WHERE recall_id = %s AND in_model GROUP BY 1 ORDER BY 3 DESC", [recall_id]).fetchall()
        unsure = c.execute("SELECT serial, p_affected, stage, customer, in_mes FROM recall_scope WHERE recall_id = %s AND p_affected > 0.01 AND p_affected < 0.99 ORDER BY abs(p_affected - 0.5) LIMIT 10", [recall_id]).fetchall()
        hist = c.execute("""SELECT width_bucket(p_affected, 0, 1.0000001, 10) AS b, count(*) AS n FROM recall_scope WHERE recall_id = %s AND p_affected > 0 GROUP BY 1 ORDER BY 1""", [recall_id]).fetchall()
        both = c.execute("""SELECT count(*) FILTER (WHERE in_model AND NOT in_mes) AS model_only, count(*) FILTER (WHERE in_mes AND NOT in_model) AS mes_only, count(*) FILTER (WHERE in_mes AND in_model) AS both
                              FROM recall_scope WHERE recall_id = %s""", [recall_id]).fetchone()
    s = m["strategies"]
    return jsonable_encoder({"recall_id": recall_id, "status": r["status"], "approved": r["approved_by"] is not None, "lot": m["lot"], "window": m["window"], "model_version": m["model_version"],
                             "recall": {"units": s["model"]["units"], "cost": money(s["model"]["cost"]), "expected_escapes": round(m["expected_escapes"], 3), "certain_units": m["certain_units"], "uncertain_units": m["uncertain_units"]},
                             "alternatives": [{"strategy": k, "units": v["units"], "cost": money(v["cost"])} for k, v in s.items()],
                             "avoided_cost_vs_broad": money(s["broad"]["cost"] - s["model"]["cost"]), "agreement_with_mes": both,
                             "by_customer": [{**x, "cost": money(x["cost"])} for x in cust], "by_stage": [{**x, "cost": money(x["cost"])} for x in stages],
                             "uncertain": [{**x, "p_affected": round(x["p_affected"], 3)} for x in unsure], "probability_histogram": [{"from": (h["b"] - 1) / 10, "units": h["n"]} for h in hist],
                             "window_reels": m["window_reels"], "fit": m["fit"], "delays": m["delays"], "splices": m["splices"], "policy": m["policy"], "examples": m["examples"], "seconds": m["seconds"],
                             "simulation_truth": m["simulation_truth"]})


class ApproveIn(BaseModel):
    note: str = Field(min_length=3, max_length=500)


@app.post("/v1/recalls/{recall_id}/approve", tags=["recalls"], summary="(+) Approve the recall set; a second person must approve, never the one who scoped it")
def approve(recall_id: uuid.UUID, body: ApproveIn, ctx: Ctx = Depends(auth("recalls:scope")), idem: str | None = IdemKey):
    def work(c):
        r = recall(c, recall_id)
        if r["created_by"] == ctx.actor_id:
            raise Problem(409, "four_eyes", "the person who scoped the recall cannot approve it")
        if "recalls:approve" not in PERMISSIONS[ctx.role]:
            raise Problem(403, "forbidden", "approval needs the head of quality")
        if r["approved_by"]:
            raise Problem(409, "already_approved")
        c.execute("UPDATE recall_case SET status = 'approved', approved_by = %s, approved_at = now() WHERE recall_id = %s", [ctx.actor_id, recall_id])
        c.execute("UPDATE model_artifact SET approved = true WHERE version = %s", [r["metrics"]["model_version"]])
        s = r["metrics"]["strategies"]["model"]
        audit.record(c, ctx, "recall.approved", "recall_case", recall_id, {"note": body.note, "units": s["units"], "cost": money(s["cost"])})
        return 200, {"recall_id": recall_id, "approved_by": ctx.actor_name, "units": s["units"], "cost": money(s["cost"]), "model_approved": r["metrics"]["model_version"]}
    return run(ctx, idem, body, work)


# ---------------------------------------------------------------- customer trace

class NoticesIn(BaseModel):
    recall_id: uuid.UUID


@app.post("/v1/notices/export", status_code=201, tags=["customer trace"], summary="Customer notices for an approved recall: per customer the shipments and serials, as JSON and CSV; units still in our warehouse go on a hold list")
def notices(body: NoticesIn, ctx: Ctx = Depends(auth("notices:export")), idem: str | None = IdemKey):
    def work(c):
        r = recall(c, body.recall_id)
        if not r["approved_by"]:
            raise Problem(409, "not_approved", "a recall must be approved before notices go out")
        rows = c.execute("SELECT serial, customer, shipment_ref, p_affected, stage FROM recall_scope WHERE recall_id = %s AND in_model ORDER BY customer, shipment_ref, serial", [body.recall_id]).fetchall()
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["customer", "shipment", "serial", "probability_affected"])
        out, hold = {}, []
        for x in rows:
            if x["customer"] is None:
                hold.append(x["serial"])
                continue
            w.writerow([x["customer"], x["shipment_ref"], x["serial"], round(x["p_affected"], 4)])
            n = out.setdefault(x["customer"], {"customer": x["customer"], "units": 0, "shipments": set(), "serials_sample": []})
            n["units"] += 1
            n["shipments"].add(x["shipment_ref"])
            if len(n["serials_sample"]) < 5:
                n["serials_sample"].append(x["serial"])
        notices_ = [{**n, "shipments": sorted(n["shipments"])} for n in sorted(out.values(), key=lambda n: -n["units"])]
        audit.record(c, ctx, "notices.exported", "recall_case", body.recall_id, {"customers": len(notices_), "units": len(rows) - len(hold), "warehouse_hold": len(hold)})
        return 201, {"recall_id": body.recall_id, "customers": len(notices_), "notices": notices_, "warehouse_hold": {"units": len(hold), "serials_sample": hold[:5]},
                     "csv_rows": len(rows) - len(hold), "csv_head": "\n".join(buf.getvalue().splitlines()[:4]), "csv_sha256": hashlib.sha256(buf.getvalue().encode()).hexdigest()}
    return run(ctx, idem, body, work)


@app.get("/v1/units/{serial}/genealogy", tags=["genealogy"], summary="One unit's genealogy: the board it holds (or the spares it might hold after a rework), the reels on each board with probabilities, the lots those reels came from, where the unit went, and every recall it is in")
def genealogy(serial: str, ctx: Ctx = Depends(auth("plant:read"))):
    with db.tx(ctx.tenant_id) as c:
        p = plant(c)
        u = c.execute("SELECT * FROM serial_unit WHERE serial = %s AND kind = 'drive_unit'", [serial]).fetchone()
        if not u:
            raise Problem(404, "unit_not_found")
        rw = c.execute("SELECT at, board_serial FROM assembly_event WHERE kind = 'rework' AND serial = %s ORDER BY at", [serial]).fetchall()
        ver = c.execute("SELECT version FROM model_artifact WHERE model_name = 'genealogy' ORDER BY approved DESC, created_at DESC LIMIT 1").fetchone()
        ver = ver["version"] if ver else None
        boards = [{"role": "fitted at build", "serial": u["board_serial"], "weight": 1.0}]
        if rw:
            boards[0]["role"], boards[0]["weight"] = "removed at rework", 0.0
            if rw[-1]["board_serial"]:
                boards.append({"role": "fitted at rework", "serial": rw[-1]["board_serial"], "weight": 1.0})
            else:
                cands = c.execute("SELECT parent_ref AS serial, probability FROM genealogy_edge WHERE model_version = %s AND relation = 'board_in_unit' AND child_ref = %s ORDER BY parent_ref", [ver, serial]).fetchall() if ver else []
                boards += [{"role": "may have been fitted at rework (the ticket has no serial)", "serial": x["serial"], "weight": x["probability"]} for x in cands]
        out_boards = []
        for bd in boards:
            b = c.execute("SELECT serial, line, built_at, work_order, spare FROM serial_unit WHERE serial = %s", [bd["serial"]]).fetchone()
            reels = c.execute("""SELECT e.parent_ref AS reel, e.probability, cl.lot_code_read, cl.date_code_read, cl.delivery_ref FROM genealogy_edge e JOIN component_lot cl ON cl.external_ref = e.parent_ref
                                  WHERE e.model_version = %s AND e.relation = 'reel_in_board' AND e.child_ref = %s ORDER BY e.probability DESC""", [ver, bd["serial"]]).fetchall() if ver else []
            for x in reels:
                x["lots"] = c.execute("SELECT parent_ref AS lot, probability FROM genealogy_edge WHERE model_version = %s AND relation = 'reel_of_lot' AND child_ref = %s ORDER BY probability DESC", [ver, x["reel"]]).fetchall()
                x["scan"] = c.execute("SELECT kind, at FROM assembly_event WHERE kind IN ('reel_scan', 'splice_sensor') AND reel_ref = %s ORDER BY kind LIMIT 1", [x["reel"]]).fetchone()
            out_boards.append({**bd, "board": b, "reels": reels})
        ship = c.execute("SELECT s.external_ref, s.customer, s.shipped_at FROM customer_delivery cd JOIN shipment s USING (shipment_id) WHERE cd.unit_id = %s", [u["unit_id"]]).fetchone()
        rets = c.execute("SELECT at, detail FROM quality_event WHERE kind = 'field_return' AND serial = %s", [serial]).fetchall()
        recalls = c.execute("SELECT recall_id, p_affected, in_model, in_mes, in_work_order, in_broad, stage, cost FROM recall_scope WHERE serial = %s", [serial]).fetchall()
    wrec, tr = W.generate(p["seed"])
    wu = {s: i for i, s in enumerate(wrec["units"]["serial"])}
    truth = None
    if serial in wu:
        fb = int(tr["unit_final"][wu[serial]])
        rs = [int(r) for r in (tr["board_r1"][fb], tr["board_r2"][fb]) if r >= 0]
        truth = {"what": "from the simulator; nothing above sees it", "board": str(wrec["boards"]["serial"][fb]),
                 "reels": [{"reel": str(wrec["reels"]["ref"][r]), "lot": W.LOT_CODES[tr["reel_lot"][r]], "date_code": W.day_to_date(tr["reel_date"][r])} for r in rs]}
    return jsonable_encoder({"serial": serial, "built_at": u["built_at"], "model_version": ver, "boards": out_boards, "reworks": rw, "shipment": ship, "returns": rets, "recalls": recalls, "simulation_truth": truth})
