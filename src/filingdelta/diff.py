"""Parse a 10-K down to its Item 1A paragraphs, then diff two years of it."""
import math
import re
import zlib

from bs4 import BeautifulSoup


def paragraphs(html: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    leaves = [b for b in soup.find_all(["p", "div"]) if not b.find(["p", "div"])]
    return [t for t in (" ".join(b.get_text(" ").split()) for b in leaves) if t]


def risk_factors(paras: list[str]) -> list[str]:
    """Item 1A body. The table of contents also says 'Item 1A', so take the longest span."""
    best: list[str] = []
    for s, p in enumerate(paras):
        if re.match(r"item\s*1a\.?\s*[-—:]?\s*risk factors", p, re.I):
            end = next((j for j in range(s + 1, len(paras)) if re.match(r"item\s*(1b|1c|2)\b", paras[j], re.I)), len(paras))
            if end - s - 1 > len(best):
                best = paras[s + 1:end]
    return best


def hash_embed(texts: list[str], dim: int = 512) -> list[list[float]]:
    """shortcut: hashed unigram+bigram bag of words, offline and deterministic.
    Swap in a dense Hugging Face encoder with the same signature for real use."""
    out = []
    for t in texts:
        w = re.findall(r"\w+", t.lower())
        v = [0.0] * dim
        for tok in w + [a + " " + b for a, b in zip(w, w[1:])]:
            v[zlib.crc32(tok.encode()) % dim] += 1
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        out.append([x / n for x in v])
    return out


def _change(kind: str, old: str = "", new: str = "", sim: float = 0.0) -> dict:
    return {"kind": kind, "old": old, "new": new, "sim": round(sim, 3), "label": ""}


def diff(old: list[str], new: list[str], embed=hash_embed, thr: float = 0.5) -> list[dict]:
    """Exact matches are unchanged and dropped. The rest are paired greedily by similarity:
    paired = modified, unpaired new = added, unpaired old = removed.
    shortcut: all-pairs over the unmatched paragraphs only; use an ANN index past ~2k paragraphs."""
    norm = lambda s: " ".join(s.split())
    old_set, new_set = {norm(p) for p in old}, {norm(p) for p in new}
    un = [p for p in new if norm(p) not in old_set]
    uo = [p for p in old if norm(p) not in new_set]
    en, eo = embed(un), embed(uo)
    pairs = sorted(((sum(a * b for a, b in zip(x, y)), i, j) for i, x in enumerate(en) for j, y in enumerate(eo)), reverse=True)
    used_n, used_o, out = set(), set(), []
    for s, i, j in pairs:
        if s < thr:
            break
        if i not in used_n and j not in used_o:
            used_n.add(i)
            used_o.add(j)
            out.append(_change("modified", uo[j], un[i], s))
    out += [_change("added", new=p) for i, p in enumerate(un) if i not in used_n]
    out += [_change("removed", old=p) for j, p in enumerate(uo) if j not in used_o]
    return out
