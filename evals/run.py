"""Eval harness: runs the pipeline on labeled fixtures, prints metrics, exits 1 if any metric
falls below evals/thresholds.json. Wired into CI, so a regression blocks the merge.

    python -m evals.run
"""
import json
import pathlib
import sys

from filingdelta.analyze import classify
from filingdelta.diff import hash_embed
from filingdelta.graph import analyze, build_graph

HERE = pathlib.Path(__file__).parent
NOT_MATERIAL = ("cosmetic", "unchanged")


def evaluate(cases: list[dict], classifier=classify, embed=hash_embed) -> dict:
    tp = pred = gold = labels_ok = claims_ok = claims_n = 0
    for case in cases:
        facts = {(k.split("|")[0], int(k.split("|")[1])): v for k, v in case["facts"].items()}
        state = analyze(case["old"], case["new"], facts, build_graph(classifier, embed), thread=case["id"])
        changes = state["changes"]
        pred += sum(c["label"] not in NOT_MATERIAL for c in changes)
        for g in case["gold_changes"]:
            hit = next((c for c in changes if g["contains"] in c["new"] + c["old"]), None)
            got = hit["label"] if hit else "unchanged"
            if g["label"] not in NOT_MATERIAL:
                gold += 1
                tp += got not in NOT_MATERIAL
                labels_ok += got == g["label"]
        verdicts = {(v["metric"], v["year"]): v["status"] for v in state.get("verdicts", [])}
        for g in case["gold_claims"]:
            claims_n += 1
            claims_ok += verdicts.get((g["metric"], g["year"])) == g["status"]
    return {
        "material_precision": tp / pred if pred else 1.0,
        "material_recall": tp / gold if gold else 1.0,
        "label_accuracy": labels_ok / gold if gold else 1.0,
        "claim_accuracy": claims_ok / claims_n if claims_n else 1.0,
    }


def main() -> int:
    metrics = evaluate(json.loads((HERE / "fixtures/cases.json").read_text()))
    floors = json.loads((HERE / "thresholds.json").read_text())
    failed = [k for k, floor in floors.items() if metrics[k] < floor]
    for k, v in metrics.items():
        print(f"{k:20s} {v:.2f}  (floor {floors[k]:.2f}) {'FAIL' if k in failed else 'ok'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
