"""The LangGraph pipeline: diff -> classify -> claims -> verify -> brief -> critic -> (human review).
Only material changes reach the later stages, and a contradicted claim pauses the run before 'review'.
State is plain dicts so any checkpointer can serialize it."""
import re
from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from .analyze import claims, classify, verify
from .diff import diff, hash_embed


class State(TypedDict, total=False):
    old: list
    new: list
    facts: dict
    changes: list
    material: list
    evidence: dict
    claims: list
    verdicts: list
    brief: list
    rejected: list
    needs_review: bool


def grounded(line: str, evidence: dict) -> bool:
    """A brief line must end in a citation that points at real evidence."""
    m = re.search(r"\[(E\d+)\]$", line)
    return bool(m) and m.group(1) in evidence


def build_graph(classifier=classify, embed=hash_embed):
    def diff_node(s):
        return {"changes": diff(s["old"], s["new"], embed)}

    def classify_node(s):
        ch = [{**c, "label": classifier(c)} for c in s["changes"]]
        return {"changes": ch, "material": [c for c in ch if c["label"] != "cosmetic"]}

    def claims_node(s):
        ev = {f"E{i}": {"label": c["label"], "text": c["new"] or c["old"]} for i, c in enumerate(s["material"])}
        return {"evidence": ev, "claims": [{**cl, "evidence": f"E{i}"} for i, c in enumerate(s["material"]) for cl in claims(c["new"])]}

    def verify_node(s):
        return {"verdicts": [{**c, **verify(c, s["facts"])} for c in s["claims"]]}

    def brief_node(s):
        # The synthesizer seam: deterministic template today, an LLM later. The critic polices either.
        lines = [f"[{e['label']}] {e['text'][:160]} [{k}]" for k, e in s["evidence"].items()]
        for v in s["verdicts"]:
            ref = f"XBRL {v['concept']}={v['actual']:,.0f}" if v["concept"] else "no matching XBRL fact"
            lines.append(f"{v['metric']} {v['year']} claimed ${v['value']:,.0f}: {v['status']} ({ref}) [{v['evidence']}]")
        return {"brief": lines}

    def critic_node(s):
        return {
            "brief": [l for l in s["brief"] if grounded(l, s["evidence"])],
            "rejected": [l for l in s["brief"] if not grounded(l, s["evidence"])],
            "needs_review": any(v["status"] == "contradicted" for v in s["verdicts"]),
        }

    g = StateGraph(State)
    for name, fn in [("diff", diff_node), ("classify", classify_node), ("claims", claims_node), ("verify", verify_node),
                     ("brief", brief_node), ("critic", critic_node), ("review", lambda s: {})]:
        g.add_node(name, fn)
    g.add_edge(START, "diff")
    g.add_edge("diff", "classify")
    g.add_conditional_edges("classify", lambda s: "claims" if s["material"] else END)
    g.add_edge("claims", "verify")
    g.add_edge("verify", "brief")
    g.add_edge("brief", "critic")
    g.add_conditional_edges("critic", lambda s: "review" if s["needs_review"] else END)
    g.add_edge("review", END)
    return g.compile(checkpointer=MemorySaver(), interrupt_before=["review"])


def analyze(old: list, new: list, facts: dict, graph=None, thread: str = "run") -> dict:
    """Runs to completion, or to the review pause: check graph.get_state(cfg).next, resume with graph.invoke(None, cfg)."""
    graph = graph or build_graph()
    return graph.invoke({"old": old, "new": new, "facts": facts}, {"configurable": {"thread_id": thread}})
