"""python -m filingdelta <CIK>: diff the latest two 10-Ks' risk factors and verify their numeric claims.
Set FILINGDELTA_UA to 'Your Name you@example.com' (SEC requires it)."""
import os
import sys

from .analyze import flatten_facts
from .diff import paragraphs, risk_factors
from .edgar import Edgar
from .graph import analyze


def main(cik: int) -> None:
    ed = Edgar(os.environ["FILINGDELTA_UA"])
    new, old = ed.filings(cik)[:2]
    section = lambda f: risk_factors(paragraphs(ed.html(cik, f["accession"], f["doc"])))
    state = analyze(section(old), section(new), flatten_facts(ed.facts(cik)))
    print(f"{old['accession']} ({old['filed']}) -> {new['accession']} ({new['filed']})")
    print("\n".join(state.get("brief") or ["No material changes."]))


if __name__ == "__main__":
    main(int(sys.argv[1]))
