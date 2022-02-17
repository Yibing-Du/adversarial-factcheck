# Copyright (c) 2021, Yamagishi Laboratory, National Institute of Informatics
# Author: Canasai Kruengkrai (canasai@nii.ac.jp)
# All rights reserved.

import argparse
import jsonlines
import io
from collections import Counter
from fever_scorer import fever_score
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix


def main(pred_file, gold_file):
    gold = [line for line in jsonlines.open(args.gold_file)]
    pred = [line for line in jsonlines.open(args.pred_file)]
    assert len(pred) == len(gold)
    pred_labels = [line["predicted_label"] for line in pred]
    gold_labels = [line["label"] for line in gold]
    score, label_acc, prec, rec, f1 = fever_score(pred, gold)
    res = "\n".join(
        [
            "           [N      R     S     ]",
            f"Precision: {precision_score(gold_labels, pred_labels, average=None).round(4)*100.0}",
            f"Recall:    {recall_score(gold_labels, pred_labels, average=None).round(4)*100.0}",
            f"F1:        {f1_score(gold_labels, pred_labels, average=None).round(4)*100.0}",
            "",
            "Confusion Matrix:",
            f"{confusion_matrix(gold_labels, pred_labels)}",
            "",
            f"{Counter(pred_labels)}",
            "",
            f"Evidence precision: {prec*100.0:.4}",
            f"Evidence recall:    {rec*100.0:.4}",
            f"Evidence F1:        {f1*100.0:.4}",
            f"Label accuracy:     {label_acc*100.0:.4}",
            f"FEVER score:        {score*100.0:.4}",
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
