# Copyright (c) 2021, Yamagishi Laboratory, National Institute of Informatics
# Author: Canasai Kruengkrai (canasai@nii.ac.jp)
# All rights reserved.

import argparse
import io
import jsonlines
import bisect
from collections import defaultdict
from tqdm import tqdm

PAD_SENT = ["[PAD]", -1, "[PAD]"]  # doc_id, sent_id, sent_text


def pad_to_max(sent_list, max_evidence_per_claim):
    if len(sent_list) < max_evidence_per_claim:
        sent_list += [PAD_SENT] * (max_evidence_per_claim - len(sent_list))


def get_all_sentences(corpus, pred_evidence, max_evidence_per_claim):
    sent_list = []
    for doc_id, sent_id, score in pred_evidence:
        if doc_id not in corpus:
            continue
        doc = {i: s for (i, s) in corpus[doc_id]["lines"]}
        sent_list.append([doc_id, sent_id, doc[sent_id]])
    pad_to_max(sent_list, max_evidence_per_claim)
    return sent_list[:max_evidence_per_claim]


def get_train_sentences(
    corpus,
    evidence,
    pred_evidence,
    label,
    max_evidence_per_claim,
):
    gold_evidence_set = set()
    for evidence_set in evidence:
        for _, _, doc_id, sent_id in evidence_set:
            if doc_id is not None and doc_id in corpus:
                gold_evidence_set.add((doc_id, sent_id))

    train_evidence = []
    pos_sents = defaultdict(lambda: set())
    for doc_id, sent_id in sorted(gold_evidence_set):
        doc = {i: s for (i, s) in corpus[doc_id]["lines"]}
        bisect.insort(
            train_evidence, (-1, doc_id, sent_id, doc[sent_id])
        )  # assign negative 1 for gold evidence
        pos_sents[doc_id].add(sent_id)

    for doc_id, sent_id, score in pred_evidence:
        if doc_id not in corpus:
            continue
        if doc_id in pos_sents and sent_id in pos_sents[doc_id]:
            continue
        doc = {i: s for (i, s) in corpus[doc_id]["lines"]}
        if sent_id not in doc:
            continue
        bisect.insort(train_evidence, (-float(score), doc_id, sent_id, doc[sent_id]))

    sent_list = []
    for score, doc_id, sent_id, sent_text in train_evidence:
        sent_list.append([doc_id, sent_id, sent_text])
    pad_to_max(sent_list, max_evidence_per_claim)

    for doc_id, sent_id, sent_text in sent_list[:max_evidence_per_claim]:
        selection_label = (
            1 if doc_id in pos_sents and sent_id in pos_sents[doc_id] else 0
        )
        yield [doc_id, sent_id, sent_text, label, selection_label]


def build_examples(args, corpus, line):
    claim_id = line["id"]
    claim_text = line["claim"]
    evidence = line.get("evidence", [])
    pred_evidence = line["predicted_evidence"]
    examples = []

    if args.training:
        label = line["label"][0]
        examples.append([claim_id, claim_text] + PAD_SENT + [label, 0])

        for evidence_sent in get_train_sentences(
            corpus,
            evidence,
            pred_evidence,
            label,
            args.max_evidence_per_claim,
        ):
            examples.append([claim_id, claim_text] + evidence_sent)
    else:
        examples.append([claim_id, claim_text] + PAD_SENT)
        for evidence_sent in get_all_sentences(
            corpus, pred_evidence, args.max_evidence_per_claim
        ):
            examples.append([claim_id, claim_text] + evidence_sent)

    return examples


def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=str, required=True)
    parser.add_argument("--in_file", type=str, required=True)
    parser.add_argument("--out_file", type=str, required=True)
    parser.add_argument("--training", action="store_true")
    parser.add_argument("--max_evidence_per_claim", type=int, default=5)
    return parser.parse_args()


def main():
    args = build_args()
    corpus = {doc["doc_id"]: doc for doc in jsonlines.open(args.corpus)}
    lines = [line for line in jsonlines.open(args.in_file)]
    out_examples = []

    for line in tqdm(lines, total=len(lines), desc="Building examples"):
        out_examples.extend(build_examples(args, corpus, line))

    print(f"Save to {args.out_file}")
    with io.open(args.out_file, "w", encoding="utf-8", errors="ignore") as out:
        for e in out_examples:
            e = list(map(str, e))
            out.write("\t".join(e) + "\n")


if __name__ == "__main__":
    main()
