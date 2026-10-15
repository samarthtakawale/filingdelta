"""Per-change analysis: label it, pull numeric claims out of it, check them against XBRL facts."""
import re
from datetime import date

KEYWORDS = {  # first match wins, so order is priority
    "going_concern": r"going concern|substantial doubt",
    "restatement": r"restate|non-reliance|material weakness",
    "litigation": r"lawsuit|litigation|class action|subpoena|investigation",
}
NUM = re.compile(r"\d+(?:[.,]\d+)*")
METRICS = {  # phrase -> XBRL us-gaap concepts, tried in order
    "net income": ["NetIncomeLoss"],
    "operating income": ["OperatingIncomeLoss"],
    "research and development": ["ResearchAndDevelopmentExpense"],
    "total assets": ["Assets"],
    "cash and cash equivalents": ["CashAndCashEquivalentsAtCarryingValue"],
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"],
}
METRIC_RE = re.compile("|".join(METRICS), re.I)
MONEY = re.compile(r"\$\s?(\d[\d,]*(\.\d+)?)\s*(thousand|million|billion)?", re.I)
YEAR = re.compile(r"\b(20\d{2})\b")
SCALE = {"": 1, "thousand": 1e3, "million": 1e6, "billion": 1e9}


def classify(c: dict) -> str:
    """shortcut: regex baseline that ignores negation ('no material weakness').
    Swap in a distilled Hugging Face text classifier with the same signature; the eval harness scores either."""
    labels = lambda t: [k for k, p in KEYWORDS.items() if re.search(p, t, re.I)]
    if c["kind"] == "removed":
        return "removed_disclosure"
    fresh = [k for k in labels(c["new"]) if k not in labels(c["old"])]
    if fresh:
        return fresh[0]
    if c["kind"] == "added":
        return "new_disclosure"
    return "numbers_changed" if set(NUM.findall(c["new"])) != set(NUM.findall(c["old"])) else "cosmetic"


def claims(text: str) -> list[dict]:
    """Dollar amounts paired with the nearest metric before and the first year after (else last year before)."""
    out = []
    for sent in re.split(r"(?<=[.;])\s+(?=[A-Z])", text):
        for m in MONEY.finditer(sent):
            metric = list(METRIC_RE.finditer(sent, 0, m.start()))
            year = YEAR.search(sent, m.end()) or next(iter(list(YEAR.finditer(sent, 0, m.start()))[::-1]), None)
            if not metric or not year:
                continue
            scale = SCALE[(m.group(3) or "").lower()]
            decimals = len(m.group(2)) - 1 if m.group(2) else 0
            out.append({
                "metric": metric[-1].group().lower(), "year": int(year.group(1)), "text": sent,
                "value": float(m.group(1).replace(",", "")) * scale,
                "tol": 0.5 * 10**-decimals * scale,  # half a unit of the precision the filing stated
            })
    return out


def flatten_facts(companyfacts: dict) -> dict[tuple[str, int], float]:
    """(concept, year of period end) -> USD value. Annual-duration or instant facts only; latest filing wins.
    shortcut: calendar year of period end, so 52/53-week and non-December fiscal years can be off by one."""
    best: dict = {}
    for concept, body in companyfacts.get("facts", {}).get("us-gaap", {}).items():
        for f in body.get("units", {}).get("USD", []):
            end = date.fromisoformat(f["end"])
            if "start" in f and not 350 <= (end - date.fromisoformat(f["start"])).days <= 380:
                continue
            if (concept, end.year) not in best or f["filed"] > best[(concept, end.year)][0]:
                best[(concept, end.year)] = (f["filed"], f["val"])
    return {k: v for k, (_, v) in best.items()}


def verify(c: dict, facts: dict) -> dict:
    for concept in METRICS[c["metric"]]:
        actual = facts.get((concept, c["year"]))
        if actual is not None:
            ok = abs(actual - c["value"]) <= c["tol"]
            return {"status": "verified" if ok else "contradicted", "concept": concept, "actual": actual}
    return {"status": "unverifiable", "concept": None, "actual": None}
