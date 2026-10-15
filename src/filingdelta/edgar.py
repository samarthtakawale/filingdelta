"""Minimal SEC EDGAR client: polite (declared User-Agent, rate limit, retry) and nothing else."""
import time

import httpx


class Edgar:
    def __init__(self, user_agent: str, rps: float = 8, client: httpx.Client | None = None):
        # SEC fair access: declare who you are, stay under 10 requests/second.
        self.http = client or httpx.Client(headers={"User-Agent": user_agent}, timeout=30, follow_redirects=True)
        self.gap, self.backoff, self._last = 1 / rps, 1.0, 0.0

    def _get(self, url: str) -> httpx.Response:
        for attempt in range(4):
            time.sleep(max(0.0, self._last + self.gap - time.monotonic()))
            self._last = time.monotonic()
            r = self.http.get(url)
            if r.status_code not in (429, 500, 502, 503, 504):
                break
            time.sleep(self.backoff * 2**attempt)
        r.raise_for_status()
        return r

    def filings(self, cik: int, form: str = "10-K") -> list[dict]:
        """Newest first. shortcut: only the 'recent' page (~1000 filings); follow filings.files for older history."""
        r = self._get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json").json()["filings"]["recent"]
        rows = zip(r["accessionNumber"], r["filingDate"], r["form"], r["primaryDocument"])
        return [{"accession": a, "filed": d, "doc": p} for a, d, f, p in rows if f == form]

    def html(self, cik: int, accession: str, doc: str) -> str:
        return self._get(f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession.replace('-', '')}/{doc}").text

    def facts(self, cik: int) -> dict:
        return self._get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json").json()
