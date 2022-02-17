# Copyright (c) 2021, Yamagishi Laboratory, National Institute of Informatics
# Author: Canasai Kruengkrai (canasai@nii.ac.jp)
# All rights reserved.

import argparse
import jsonlines
import io
from fever_scorer import fever_score


def main(pred_file, gold_file):
    gold = [line for line in jsonlines.open(args.gold_file)]

    pred = []
    for i, line in enumerate(jsonlines.open(args.pred_file)):
        assert line["id"] == gold[i]["id"]
        pred_i = {}
        pred_i["id"] = gold[i]["id"]
        pred_i["predicted_label"] = gold[i]["label"]  # use gold label
        pred_i["predicted_evidence"] = [e[:2] for e in line["predicted_evidence"]]
        pred.append(pred_i)

    assert len(pred) == len(gold)
    _, _, prec, rec, f1 = fever_score(pred, gold)
    res = "\n".join(
        [
            f"Evidence precision: {prec*100.0:.4}",
            f"Evidence recall:    {rec*100.0:.4}",
            f"Evidence F1:        {f1*100.0:.4}",
        ]
    )
    print(res)
    print(f"Save to {args.out_file}")
    with io.open(args.out_file, "w", encoding="utf-8", errors="ignore") as out:
        out.write(res + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold_file", type=str, required=True)
    parser.add_argument("--pred_file", type=str, required=True)
    parser.add_argument("--out_file", type=str, required=True)
    args = parser.parse_args()
    main(args.pred_file, args.gold_file)
