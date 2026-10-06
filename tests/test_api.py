"""Integration, end-to-end and security tests against a real Postgres."""
import numpy as np
import psycopg
import pytest

from core import db, jobs, scenario
from rc import api, world as W

from .conftest import bearer

HEAD, ENG, VIEWER, OTHER = bearer("arden", "head_of_quality"), bearer("arden", "quality_engineer"), bearer("arden", "viewer"), bearer("other", "head_of_quality")


def test_nothing_runs_before_the_plant_is_loaded(client):
    assert client.get("/v1/plant", headers=VIEWER).json()["code"] == "no_plant"
    assert client.post("/v1/events/assembly", headers=ENG, json={"generate": {"seed": 1}}).status_code == 403
    assert client.post("/v1/events/assembly", headers=HEAD, json={}).json()["code"] == "generate_or_events"


@pytest.fixture(scope="module")
def demo(client):
    results, _ = scenario.run(client, drain=jobs.drain)
    return results


def one(sql, params=(), tenant=None):
    with db.tx(tenant) as c:
        return c.execute(sql, params).fetchone()


def test_records_round_trip_through_the_database(demo):
    with db.tx() as c:
        got = api.records_from_db(c)
    rec, _ = W.generate(11)
    assert (got["boards"]["serial"] == rec["boards"]["serial"]).all() and np.allclose(got["boards"]["tp"], rec["boards"]["tp"], atol=1e-4)
    assert got["reels"]["lot_read"] == rec["reels"]["lot_read"] and len(got["scans"]) == len(rec["scans"]) and len(got["units"]["serial"]) == len(rec["units"]["serial"]) + 2
    assert sum(r["new_board"] is None for r in got["reworks"]) == sum(r["new_board"] is None for r in rec["reworks"])


def test_demo_journey(demo):
    assert demo["gen"]["result"]["units"] > 39000 and demo["plant"]["units"] == demo["gen"]["result"]["units"] + 2
    ev = demo["events"]
    assert len(ev["accepted"]) == 2 and {r["event_id"]: r["why"].split(":")[0].split(" ")[0] for r in ev["rejected"]} == {
        "MES-UB-90001": "duplicate", "MES-UB-90003": "board", "MES-UB-90004": "unknown", "MES-RW-90005": "out of order".split()[0], "MES-UB-90006": "timestamp", "MES-SC-90007": "reel"}
    q = demo["plant"]["record_quality"]
    assert q["lot_code_unread"] + q["lot_code_matches_no_notice"] > 5 and q["reworks_without_board_serial"] > 200 and q["reel_changes_without_scan"] >= 1
    top = demo["inc"]["returns_by_lot"][0]
    assert top["lot"] == "B-177" and top["ratio_90"][0] > 1.5
    im = demo["impact"]
    t = im["simulation_truth"]["scored"]
    assert t["model"]["affected_missed"] <= 2 and t["model"]["units"] < t["broad"]["units"] / 2 and t["mes"]["affected_missed"] > 100
    assert im["recall"]["expected_escapes"] <= 0.5 and im["avoided_cost_vs_broad"] > 500000
    assert one("SELECT count(*) AS n FROM recall_scope WHERE in_model")["n"] == im["recall"]["units"] and one("SELECT count(*) AS n FROM genealogy_edge")["n"] > 41000
    assert len(demo["unit_splice"]["boards"][0]["reels"]) == 2 and "no serial" in demo["unit_rework"]["boards"][-1]["role"]
    assert not demo["unit_missed"]["recalls"][0]["in_mes"] and demo["unit_missed"]["recalls"][0]["in_model"]
    assert demo["self"]["code"] == "four_eyes" and demo["approve"]["units"] == im["recall"]["units"]
    n = demo["notices"]
    assert sum(x["units"] for x in n["notices"]) + n["warehouse_hold"]["units"] == im["recall"]["units"] and n["customers"] >= 5
    assert demo["audit"]["chain_valid"] and {"plant.loaded", "events.recorded", "incident.recorded", "recall.scoped", "recall.approved", "notices.exported"} <= {e["action"] for e in demo["audit"]["events"]}


def test_validation_replay_and_approval_rules(client, demo):
    rid, iid = demo["scope"]["result"]["recall_id"], demo["inc"]["incident_id"]
    body = {"supplier_lot": "B-177", "date_code_from": "2026-07-09", "date_code_to": "2026-07-05"}
    assert client.post("/v1/quality/incidents", headers=ENG, json=body).json()["code"] == "window_reversed"
    assert client.post("/v1/quality/incidents", headers=ENG, json={**body, "supplier_lot": "B-999"}).json()["code"] == "unknown_lot"
    assert client.post("/v1/quality/incidents", headers=ENG, json={**body, "date_code_from": "2026-01-01", "date_code_to": "2026-01-05"}).json()["code"] == "window_outside_lot"
    assert client.post("/v1/recalls/scope", headers=ENG, json={"incident_id": iid}).json()["code"] == "recall_exists"
    assert client.post(f"/v1/recalls/{rid}/approve", headers=HEAD, json={"note": "again"}).json()["code"] == "already_approved"
    assert client.post(f"/v1/recalls/{rid}/approve", headers=VIEWER, json={"note": "me"}).status_code == 403
    h = {**ENG, "Idempotency-Key": "notice-replay"}
    a, b = client.post("/v1/notices/export", headers=h, json={"recall_id": rid}), client.post("/v1/notices/export", headers=h, json={"recall_id": rid})
    assert a.json() == b.json() and b.headers.get("Idempotent-Replay") == "true"
    # a second, wider notice: scoped but not approved, so no notices
    inc2 = client.post("/v1/quality/incidents", headers=ENG, json={"supplier_lot": "B-177", "date_code_from": "2026-07-04", "date_code_to": "2026-07-12"}).json()
    job = client.post("/v1/recalls/scope", headers=ENG, json={"incident_id": inc2["incident_id"], "max_expected_escapes": 2}).json()
    jobs.drain()
    assert client.post("/v1/notices/export", headers=ENG, json={"recall_id": job["recall_id"]}).json()["code"] == "not_approved"
    assert client.get(f"/v1/recalls/{job['recall_id']}/impact", headers=VIEWER).json()["recall"]["units"] > demo["impact"]["recall"]["units"]


def test_tenant_isolation_and_append_only_audit(client, demo):
    rid = demo["scope"]["result"]["recall_id"]
    assert client.get(f"/v1/recalls/{rid}/impact", headers=OTHER).status_code == 404
    assert client.get(f"/v1/units/{demo['impact']['examples']['splice']}/genealogy", headers=OTHER).json()["code"] == "no_plant"
    with db.tx() as c:
        other = c.execute("SELECT DISTINCT tenant_id FROM api_tokens WHERE actor_name LIKE '%%Tamsin%%'").fetchone()["tenant_id"]
    assert one("SELECT count(*) AS n FROM serial_unit", tenant=other)["n"] == 0 and one("SELECT count(*) AS n FROM recall_scope", tenant=other)["n"] == 0
    with pytest.raises(psycopg.errors.RaiseException), db.tx() as c:
        c.execute("UPDATE audit_events SET action = 'edited'")
