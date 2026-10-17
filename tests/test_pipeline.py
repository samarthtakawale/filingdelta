import json

import httpx

from evals.run import HERE, evaluate, main
from filingdelta.analyze import claims, flatten_facts, verify
from filingdelta.diff import diff, paragraphs, risk_factors
from filingdelta.edgar import Edgar
from filingdelta.graph import analyze, build_graph, grounded

CASES = json.loads((HERE / "fixtures/cases.json").read_text())
FACTS = {("Revenues", 2023): 1_234_000_000, ("ResearchAndDevelopmentExpense", 2023): 520_000_000}


def test_edgar_retries_then_filters_by_form():
    calls = []

    def handler(req):
        calls.append(req)
        if len(calls) == 1:
            return httpx.Response(429)
        recent = {"accessionNumber": ["0-2", "0-1"], "filingDate": ["2024-02-01", "2023-05-01"],
                  "form": ["10-K", "10-Q"], "primaryDocument": ["a.htm", "b.htm"]}
        return httpx.Response(200, json={"filings": {"recent": recent}})

    ed = Edgar("t t@t.com", client=httpx.Client(transport=httpx.MockTransport(handler)))
    ed.backoff = 0
    assert ed.filings(320193) == [{"accession": "0-2", "filed": "2024-02-01", "doc": "a.htm"}]
    assert len(calls) == 2 and "CIK0000320193" in str(calls[0].url)


def test_risk_factors_skips_table_of_contents():
    html = """<div>Item 1A. Risk Factors</div><div>Item 1B. Unresolved Staff Comments</div>
    <div>Item 1A. Risk Factors</div><p>Risk one.</p><p>Risk two.</p><div>Item 1B. Unresolved</div>"""
    assert risk_factors(paragraphs(html)) == ["Risk one.", "Risk two."]


def test_diff_pairs_edits_and_flags_adds_and_removes():
    kinds = {c["kind"] for c in diff(
        ["Our revenue was $1.0 billion in 2022.", "Old risk about tariffs."],
        ["Our revenue was $1.2 billion in 2023.", "A brand new going concern warning."])}
    assert kinds == {"modified", "added", "removed"}


def test_claims_and_verify_catch_a_wrong_number():
    ok, bad = claims("Our revenue was $1.2 billion in 2023."), claims("Our research and development expense was $450 million in 2023.")
    assert verify(ok[0], FACTS)["status"] == "verified"
    assert verify(bad[0], FACTS)["status"] == "contradicted"
    assert verify(claims("Net income was $5 million in 2023.")[0], FACTS)["status"] == "unverifiable"


def test_flatten_facts_keeps_annual_and_latest_filing_only():
    row = lambda start, end, val, filed: {"start": start, "end": end, "val": val, "filed": filed}
    facts = {"facts": {"us-gaap": {"Revenues": {"units": {"USD": [
        row("2023-01-01", "2023-12-31", 100, "2024-02-01"),
        row("2023-01-01", "2023-12-31", 110, "2024-08-01"),  # amended, later filing wins
        row("2023-10-01", "2023-12-31", 30, "2024-02-01"),   # quarter, dropped
    ]}}}}}
    assert flatten_facts(facts) == {("Revenues", 2023): 110}


def test_grounded_rejects_missing_or_unknown_citations():
    ev = {"E0": {}}
    assert grounded("x [E0]", ev) and not grounded("x [E9]", ev) and not grounded("x", ev)


def test_cosmetic_only_filing_skips_the_expensive_path():
    state = analyze(CASES[1]["old"], CASES[1]["new"], {})
    assert "brief" not in state and not state["material"]


def test_contradicted_claim_pauses_for_review_then_resumes():
    case = CASES[0]
    g, cfg = build_graph(), {"configurable": {"thread_id": "pause"}}
    analyze(case["old"], case["new"], FACTS, g, thread="pause")
    assert g.get_state(cfg).next == ("review",)
    assert any("contradicted" in l for l in g.get_state(cfg).values["brief"])
    g.invoke(None, cfg)
    assert g.get_state(cfg).next == ()


def test_eval_gate_passes_on_baseline_and_fails_on_a_broken_classifier():
    assert main() == 0
    assert evaluate(CASES, classifier=lambda c: "cosmetic")["material_recall"] == 0


def test_year_bump_alone_is_cosmetic_but_a_new_figure_is_not():
    from filingdelta.analyze import classify
    c = {"kind": "modified", "old": "Sales fell in fiscal 2024.", "new": "Sales fell in fiscal 2025.", "sim": 0.9, "label": ""}
    assert classify(c) == "cosmetic"
    assert classify({**c, "new": "Sales fell 5% in fiscal 2025."}) == "numbers_changed"


def test_risk_factors_handles_heading_split_across_table_cells():
    cell = lambda a, b: f"<table><tr><td><p>{a}</p></td><td><p>{b}</p></td></tr></table>"
    html = (cell("Item 1A.", "RISK FACTORS") + cell("Item 1B.", "UNRESOLVED STAFF COMMENTS")
            + cell("Item 1A.", "RISK FACTORS") + "<p>Risk one.</p><p>Risk two.</p>" + cell("Item 1B.", "UNRESOLVED STAFF COMMENTS"))
    assert risk_factors(paragraphs(html)) == ["Risk one.", "Risk two."]

def test_risk_factors_handles_broken_heading_and_page_noise():
    html = ("<p>Item 1A.</p><p>Risk Factors</p><p>14</p><p>Item 1B.</p><p>Unresolved Staff Comments</p>"
            "<p>ITEM 1A. RIS K FACTORS</p><p>Risk one.</p><p>Item 1A</p><p>15</p><p>Risk two.</p>"
            "<p>Item 1B, 1C</p><p>ITEM 1B. UNRESOLVE D STAFF COMMENTS</p>")
    assert risk_factors(paragraphs(html)) == ["Risk one.", "Risk two."]