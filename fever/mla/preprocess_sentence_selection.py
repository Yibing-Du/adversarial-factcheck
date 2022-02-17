# Copyright (c) 2021, Yamagishi Laboratory, National Institute of Informatics
# Author: Canasai Kruengkrai (canasai@nii.ac.jp)
# All rights reserved.

import argparse
import jsonlines
import random
import io
from tqdm import tqdm
from collections import defaultdict


def is_disambiguation_page(doc_id):
    return "-LRB-disambiguation-RRB-" in doc_id


def sample_sentences(claim_id, corpus, doc_id, pos_sent_ids=set(), num_samples=1):
    sents = []
    for sent_id, sent_text in corpus[doc_id]["lines"]:
        if sent_id in pos_sent_ids:
            continue
        sents.append((doc_id, sent_id, sent_text))

    if num_samples is None:
        return sents
    else:
        return random.sample(sents, min(len(sents), num_samples))


def get_all_sentences(claim_id, corpus, pred_docs):
    for doc_id in pred_docs:
        if doc_id not in corpus or is_disambiguation_page(doc_id):
            continue
        for evidence in sample_sentences(claim_id, corpus, doc_id, num_samples=None):
            yield evidence


def get_train_sentences(
    claim_id,
    corpus,
    evidence,
    pred_docs,
    neg_ratio,
    neg_per_pred_doc,
):
    pos_sents = defaultdict(lambda: set())
    for evidence_set in evidence:
        for _, _, doc_id, sent_id in evidence_set:
            if doc_id not in corpus or is_disambiguation_page(doc_id):
                continue
            pos_sents[doc_id].add(sent_id)

    for doc_id, pos_sent_ids in pos_sents.items():
        doc = {i: s for (i, s) in corpus[doc_id]["lines"]}
        for sent_id in pos_sent_ids:
            yield (doc_id, sent_id, doc[sent_id], 1)

        # Sample negative sentences from gold docs
        for evidence in sample_sentences(
            claim_id,
            corpus,
            doc_id,
            pos_sent_ids,
            num_samples=neg_ratio * len(pos_sent_ids),
        ):
            yield evidence + (0,)

    # Sample negative sentences from pred docs
    for doc_id in pred_docs:
        if (
            doc_id not in corpus
            or doc_id in pos_sents
            or is_disambiguation_page(doc_id)
        ):
            continue
        for evidence in sample_sentences(
            claim_id, corpus, doc_id, num_samples=neg_per_pred_doc
        ):
            yield evidence + (0,)


def build_examples(args, corpus, line):
    if args.training and line["verifiable"] == "NOT VERIFIABLE":
        return []

    claim_id = line["id"]
    claim = line["claim"]
    evidence = line.get("evidence", [])
    pred_docs = line["predicted_pages"]
    examples = []
    if args.training:
        for doc_id, sent_id, sent_text, label in get_train_sentences(
            claim_id,
            corpus,
            evidence,
            pred_docs,
            args.neg_ratio,
            args.neg_per_pred_doc,
        ):
            examples.append([claim_id, claim, doc_id, sent_id, sent_text, label])
    else:
        for doc_id, sent_id, sent_text in get_all_sentences(
            claim_id, corpus, pred_docs
        ):
            examples.append([claim_id, claim, doc_id, sent_id, sent_text])
    return examples


def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=str, required=True)
    parser.add_argument("--in_file", type=str, required=True)
    parser.add_argument("--out_file", type=str, required=True)
    parser.add_argument("--neg_ratio", type=int, default=2)
    parser.add_argument("--neg_per_pred_doc", type=int, default=2)
    parser.add_argument("--training", action="store_true")
    parser.add_argument("--seed", type=int, default=3435)
    return parser.parse_args()


def main():
    args = build_args()

    random.seed(args.seed)

    corpus = {doc["doc_id"]: doc for doc in jsonlines.open(args.corpus)}

    lines = [line for line in jsonlines.open(args.in_file)]

    out_examples = []
    for line in tqdm(lines, total=len(lines), desc="Building examples"):
        out_examples.extend(build_examples(args, corpus, line))

    print(f"Save to {args.out_file}")
    with io.open(args.out_file, "w", encoding="utf-8", errors="ignore") as out:
        for ex in out_examples:
            ex = list(map(str, ex))
            out.write("\t".join(ex) + "\n")


if __name__ == "__main__":
    main()
