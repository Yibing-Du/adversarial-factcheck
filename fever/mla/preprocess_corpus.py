# Copyright (c) 2021, Yamagishi Laboratory, National Institute of Informatics
# Author: Canasai Kruengkrai (canasai@nii.ac.jp)
# All rights reserved.

import argparse
import jsonlines
import re
import unicodedata
from functools import partial
from multiprocessing import Pool, cpu_count
from multiprocessing.util import Finalize
from tqdm import tqdm
from fever_doc_db import FeverDocDB

PROCESS_DB = None


def get_sentences(lines):
    sents = []
    lines = re.split(r"\n(?=\d+)", lines)
    for line in lines:
        line = line.split("\t")
        if len(line) < 2:
            continue
        sent_id = int(line[0])
        sent_text = unicodedata.normalize("NFD", line[1].strip())
        if not len(sent_text):
            continue
        sents.append([sent_id, sent_text])
    return sents


def get_documents(args, line):
    global PROCESS_DB
    evidence = line.get("evidence", [])
    pred_docs = line["predicted_pages"]
    gold_docs = [e[2] for es in evidence for e in es if e[2] is not None]
    docs = []
    for doc_id in set(gold_docs + pred_docs):
        lines = PROCESS_DB.get_doc_lines(doc_id)
        if lines:
            docs.append([doc_id, get_sentences(lines)])  # keep unnormalized doc_id
    return docs


def init(db_class, db_opts):
    global PROCESS_DB
    PROCESS_DB = db_class(**db_opts)
    Finalize(PROCESS_DB, PROCESS_DB.close, exitpriority=100)


def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_file", type=str, required=True)
    parser.add_argument("--in_file", type=str, required=True)
    parser.add_argument("--out_file", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=8)
    return parser.parse_args()


def main():
    args = build_args()

    lines = [line for line in jsonlines.open(args.in_file)]

    threads = min(args.num_workers, cpu_count())
    workers = Pool(
        threads,
        initializer=init,
        initargs=(FeverDocDB, {"db_path": args.db_file}),
    )

    _get_documents = partial(get_documents, args)

    out_docs = []
    for docs in tqdm(
        workers.imap_unordered(_get_documents, lines),
        total=len(lines),
        desc="Getting documents",
    ):
        out_docs.extend(docs)

    workers.close()
    workers.join()

    out_docs = {docs[0]: docs[1] for docs in out_docs}
    print(f"Save to {args.out_file}")
    with jsonlines.open(args.out_file, "w") as out:
        for k, v in sorted(out_docs.items(), key=lambda x: x[0]):
            out.write({"doc_id": k, "lines": v})


if __name__ == "__main__":
    main()
