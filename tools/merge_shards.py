#!/usr/bin/env python3
"""Merge sharded run.py evaluations into one result file.

Shards are slices `--start k*w --max_samples w` of the same seed-deterministic
shuffle, so they are disjoint; this is asserted on the question text.

    python tools/merge_shards.py OUT_TAG SHARD_TAG_1 SHARD_TAG_2 ... [--results_dir results]
"""
import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_tag")
    ap.add_argument("shard_tags", nargs="+")
    ap.add_argument("--results_dir", default="results")
    a = ap.parse_args()

    preds, summary, seen = [], None, set()
    for t in a.shard_tags:
        p = os.path.join(a.results_dir, f"{t}.json")
        if not os.path.exists(p):
            raise SystemExit(f"missing shard {p}")
        d = json.load(open(p))
        if summary is None:
            summary = dict(d.get("summary", {}))
        for pr in d["predictions"]:
            if pr["question"] in seen:
                raise SystemExit(f"shard {t} repeats a question; the --start offsets overlap")
            seen.add(pr["question"])
            preds.append(pr)

    n = len(preds)
    correct = sum(1 for p in preds if p.get("correct"))
    summary.update(run_id=a.out_tag, n=n, correct=correct, accuracy=correct / n,
                   merged_from=a.shard_tags)
    for k in ("total_time_sec", "sec_per_sample_mean"):
        summary.pop(k, None)
    out = os.path.join(a.results_dir, f"{a.out_tag}.json")
    json.dump({"summary": summary, "predictions": preds}, open(out, "w"), ensure_ascii=False)
    print(f"merged {len(a.shard_tags)} shards -> {out}  n={n} acc={correct / n:.4f}")


if __name__ == "__main__":
    main()
