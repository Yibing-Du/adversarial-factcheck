# Copyright (c) 2021, Yamagishi Laboratory, National Institute of Informatics
# Author: Canasai Kruengkrai (canasai@nii.ac.jp)
# All rights reserved.

import argparse
import bisect
import jsonlines
from collections import defaultdict
from transformers.data.processors.utils import DataProcessor


def get_best_evidence(in_file, pred_sent_file, min_score):
    print(in_file, pred_sent_file)
    lines_0 = list(line for line in DataProcessor._read_tsv(in_file))
    lines_1 = list(
        float(line.strip()) for line in open(pred_sent_file, "r", encoding="utf-8-sig")
    )
    assert len(lines_0) == len(lines_1)

    best_evidence = defaultdict(lambda: [])
    for (line_0, line_1) in zip(lines_0, lines_1):
        assert len(line_0) == 5
        claim_id, claim, doc_id, sent_id, sent_text = line_0
        score = line_1
        claim_id, sent_id, score = int(claim_id), int(sent_id), float(score)
        if score > min_score:
            bisect.insort(best_evidence[claim_id], (-score, doc_id, sent_id))

    for claim_id in best_evidence:
        for i, (score, doc_id, sent_id) in enumerate(best_evidence[claim_id]):
            best_evidence[claim_id][i] = (doc_id, sent_id, -score)

    return best_evidence


def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file", type=str, required=True)
    parser.add_argument("--pred_sent_file", type=str, required=True)
    parser.add_argument("--pred_doc_file", type=str, required=True)
    parser.add_argument("--out_file", type=str, required=True)
    parser.add_argument("--max_evidence_per_claim", type=int, default=5)
    parser.add_argument("--min_score", type=float, default=0.0)
    return parser.parse_args()


def main():
    args = build_args()
    best_evidence = get_best_evidence(args.in_file, args.pred_sent_file, args.min_score)

    print(f"Save to {args.out_file}")
    with jsonlines.open(args.pred_doc_file) as fin, jsonlines.open(
        args.out_file, "w"
    ) as out:
        for line in fin:
            claim_id = line["id"]
            if "noun_phrases" in line:
                del line["noun_phrases"]
            if "wiki_results" in line:
                del line["wiki_results"]
            line["predicted_evidence"] = best_evidence[claim_id][
                : args.max_evidence_per_claim
            ]
            out.write(line)


if __name__ == "__main__":
    main()
