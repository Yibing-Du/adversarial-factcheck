# Copyright (c) 2021, Yamagishi Laboratory, National Institute of Informatics
# Author: Canasai Kruengkrai (canasai@nii.ac.jp)
# All rights reserved.

import argparse
import csv
import jsonlines
import numpy as np


def get_predictions(pred_sent_file, pred_claim_file):
    lines_0 = list(line for line in jsonlines.open(pred_sent_file))
    lines_1 = list(
        csv.reader(open(pred_claim_file, "r", encoding="utf-8-sig"), delimiter=" ")
    )
    assert len(lines_0) == len(lines_1)

    labels = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
    predictions = {}
    for line_0, line_1 in zip(lines_0, lines_1):
        scores = np.asarray(list(map(float, line_1)))
        label_idx = np.argmax(scores)
        label = labels[label_idx]
        claim_id = int(line_0["id"])
        evidence = []
        for doc_id, sent_id, score in line_0["predicted_evidence"]:
            evidence.append((doc_id, sent_id))
        predictions[claim_id] = {"label": label, "evidence": evidence}
    return predictions


def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--pred_sent_file", type=str, required=True)
    parser.add_argument("--pred_claim_file", type=str, required=True)
    parser.add_argument("--out_file", type=str, required=True)
    return parser.parse_args()


def main():
    args = build_args()
    predictions = get_predictions(args.pred_sent_file, args.pred_claim_file)

    print(f"Save to {args.out_file}")
    with jsonlines.open(args.data_file) as fin, jsonlines.open(
        args.out_file, "w"
    ) as out:
        for line in fin:
            claim_id = line["id"]
            out.write(
                {
                    "id": claim_id,
                    "predicted_label": predictions[claim_id]["label"],
                    "predicted_evidence": predictions[claim_id]["evidence"],
                }
            )


if __name__ == "__main__":
    main()
