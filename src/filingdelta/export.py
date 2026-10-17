"""python -m filingdelta.export 320193 789019 ...
Writes changes.csv: every change found between each company's latest two 10-Ks, ready to label in Excel."""
import csv
import os
import sys

from .analyze import classify
from .diff import diff, paragraphs, risk_factors
from .edgar import Edgar

COLS = ["cik", "old_accession", "new_accession", "kind", "sim", "predicted_label", "old_text", "new_text", "your_label", "notes"]


def rows(ed, cik: int):
    new, old = ed.filings(cik)[:2]
    section = lambda f: risk_factors(paragraphs(ed.html(cik, f["accession"], f["doc"])))
    o, n = section(old), section(new)
    if min(len(o), len(n)) < 5:  # almost certainly a parser miss, not a real filing
        raise ValueError(f"Item 1A not found properly ({len(o)} and {len(n)} paragraphs)")
    for c in diff(o, n):
        yield [cik, old["accession"], new["accession"], c["kind"], c["sim"], classify(c), c["old"], c["new"], "", ""]


if __name__ == "__main__":
    ed = Edgar(os.environ["FILINGDELTA_UA"])
    with open("changes.csv", "w", newline="", encoding="utf-8-sig") as f:  # utf-8-sig so Excel shows curly quotes right
        w = csv.writer(f)
        w.writerow(COLS)
        for cik in map(int, sys.argv[1:]):
            try:
                batch = list(rows(ed, cik))
            except Exception as e:
                print(f"{cik}: skipped ({e})")
                continue
            w.writerows(batch)
            print(f"{cik}: {len(batch)} changes")