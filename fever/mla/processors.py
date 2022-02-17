# Copyright (c) 2021, Yamagishi Laboratory, National Institute of Informatics
# Author: Canasai Kruengkrai (canasai@nii.ac.jp)
# All rights reserved.

import io
import numpy as np
import unicodedata
import re
from dataclasses import dataclass
from functools import partial
from multiprocessing import Pool, cpu_count
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    mean_squared_error,
)
from tqdm import tqdm
from transformers.data.processors.utils import DataProcessor
from typing import List, Optional, Union

tokenizer = None


@dataclass
class InputExample:
    guid: str
    text_a: str
    text_b: Optional[str] = None
    label: Optional[str] = None
    selection_label: Optional[str] = None
    index: Optional[int] = None


@dataclass(frozen=True)
class InputFeatures:
    input_ids: List[int]
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None
    label: Optional[Union[int, float]] = None
    selection_label: Optional[Union[int, float]] = None
    index: Optional[int] = None


def save_predictions(task, preds, out_file):
    output_mode = fc_output_modes[task]

    def label_from_pred(pred):
        if output_mode == "classification":
            if task == "sentence-selection":
                return str(round(pred[1], 5))  # score for ranking
            elif task == "claim-verification":
                return " ".join([str(round(score, 5)) for score in pred])
            else:
                raise KeyError(task)
        elif output_mode == "regression":
            return str(round(pred[0], 5))
        raise KeyError(output_mode)

    output = "\n".join([label_from_pred(pred) for pred in preds])
    with io.open(out_file, "w", encoding="utf8", errors="ignore") as out:
        out.write(output + "\n")


def convert_example_to_features(
    example,
    max_length,
    label_map,
    output_mode,
):
    if max_length is None:
        max_length = tokenizer.max_len

    def label_from_example(example: InputExample):
        if example.label is None:
            return None
        if output_mode == "classification":
            return label_map[example.label]
        elif output_mode == "regression":
            return float(example.label)
        raise KeyError(output_mode)

    inputs = tokenizer.encode_plus(
        example.text_a,
        example.text_b,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        truncation_strategy="only_second",
    )
    label = label_from_example(example)
    return InputFeatures(
        **inputs,
        label=label,
        selection_label=example.selection_label,
        index=example.index,
    )


def convert_example_to_features_init(tokenizer_for_convert):
    global tokenizer
    tokenizer = tokenizer_for_convert


def convert_examples_to_features(
    examples,
    tokenizer,
    max_length=None,
    task=None,
    label_list=None,
    output_mode=None,
    threads=8,
):
    if task is not None:
        processor = fc_processors[task]()
        if label_list is None:
            label_list = processor.get_labels()
        if output_mode is None:
            output_mode = fc_output_modes[task]

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []

    threads = min(threads, cpu_count())
    with Pool(
        threads, initializer=convert_example_to_features_init, initargs=(tokenizer,)
    ) as p:
        annotate_ = partial(
            convert_example_to_features,
            max_length=max_length,
            label_map=label_map,
            output_mode=output_mode,
        )
        features = list(
            tqdm(
                p.imap(annotate_, examples, chunksize=32),
                total=len(examples),
            )
        )
    return features


def compute_metrics(task, preds, labels):
    assert len(preds) == len(
        labels
    ), f"Predictions and labels have mismatched lengths {len(preds)} and {len(labels)}"

    output_mode = fc_output_modes[task]
    if output_mode == "classification":
        assert preds.shape[1] == fc_num_labels[task]
        preds = np.argmax(preds, axis=1)
    elif output_mode == "regression":
        preds = np.squeeze(preds)

    if output_mode == "classification":
        return {
            "acc": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds, average="macro"),
            "precision": precision_score(labels, preds, average="macro"),
            "recall": recall_score(labels, preds, average="macro"),
        }
    elif output_mode == "regression":
        return {"mse": mean_squared_error(labels, preds)}
    else:
        raise KeyError(task)


def process_claim(text):
    text = unicodedata.normalize("NFD", text)
    text = re.sub(r" \-LSB\-.*?\-RSB\-", "", text)
    text = re.sub(r"\-LRB\- \-RRB\- ", "", text)
    text = re.sub(" -LRB-", " ( ", text)
    text = re.sub("-RRB-", " )", text)
    text = re.sub("--", "-", text)
    text = re.sub("``", '"', text)
    text = re.sub("''", '"', text)
    return text


def process_title(text):
    text = unicodedata.normalize("NFD", text)
    text = re.sub("_", " ", text)
    text = re.sub(" -LRB-", " ( ", text)
    text = re.sub("-RRB-", " )", text)
    text = re.sub("-COLON-", ":", text)
    return text


def process_sentence(text):
    text = unicodedata.normalize("NFD", text)
    text = re.sub(" -LSB-.*-RSB-", " ", text)
    text = re.sub(" -LRB- -RRB- ", " ", text)
    text = re.sub("-LRB-", "(", text)
    text = re.sub("-RRB-", ")", text)
    text = re.sub("-COLON-", ":", text)
    text = re.sub("_", " ", text)
    text = re.sub(r"\( *\,? *\)", "", text)
    text = re.sub(r"\( *[;,]", "(", text)
    text = re.sub("--", "-", text)
    text = re.sub("``", '"', text)
    text = re.sub("''", '"', text)
    return text


class SentenceSelectionProcessor(DataProcessor):
    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def get_dummy_label(self):
        return "0"

    def get_length(self, file_path):
        return sum(1 for line in open(file_path, "r", encoding="utf-8-sig"))

    def get_examples(self, file_path, set_type, training=True, use_title=True):
        examples = []
        for (i, line) in enumerate(self._read_tsv(file_path)):
            guid = f"{set_type}-{i}"
            index = int(line[0])
            text_a = process_claim(line[1])
            text_b = None
            if 1 == 1: # int(line[3]) != -1:  # not claim-only line
                title = process_title(line[2])
                sentence = process_sentence(line[4])
                text_b = f"{title} : {sentence}" if use_title else sentence
            label = line[5] if training else self.get_dummy_label()
            selection_label = int(line[6]) if training and len(line) > 6 else None
            examples.append(
                InputExample(
                    guid=guid,
                    text_a=text_a,
                    text_b=text_b,
                    label=label,
                    selection_label=selection_label,
                    index=index,
                )
            )
        return examples


class ClaimVerificationProcessor(SentenceSelectionProcessor):
    def get_labels(self):
        """See base class."""
        return ["S", "R", "N"]  # SUPPORTS, REFUTES, NOT ENOUGH INFO

    def get_dummy_label(self):
        return "N"


fc_processors = {
    "sentence-selection": SentenceSelectionProcessor,
    "claim-verification": ClaimVerificationProcessor,
}

fc_num_labels = {
    "sentence-selection": 2,
    "claim-verification": 3,
}

fc_output_modes = {
    "sentence-selection": "classification",
    "claim-verification": "classification",
}
