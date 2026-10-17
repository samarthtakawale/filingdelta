from filingdelta.export import rows


class FakeEdgar:
    def filings(self, cik):
        return [{"accession": "N", "doc": "n"}, {"accession": "O", "doc": "o"}]

    def html(self, cik, accession, doc):
        paras = [f"Filler risk paragraph number {i} about supply chains." for i in range(6)]
        if accession == "N":
            paras.append("We received a subpoena from regulators.")
        body = "".join(f"<p>{p}</p>" for p in paras)
        return f"<div>Item 1A. Risk Factors</div>{body}<div>Item 1B. Unresolved Staff Comments</div>"


def test_export_rows_flag_the_new_paragraph():
    out = list(rows(FakeEdgar(), 1))
    assert len(out) == 1 and out[0][3] == "added" and out[0][5] == "litigation"