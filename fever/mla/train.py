# Copyright (c) 2021, Yamagishi Laboratory, National Institute of Informatics
# Author: Canasai Kruengkrai (canasai@nii.ac.jp)
# All rights reserved.

import argparse
import datetime
import torch
import numpy as np
import pytorch_lightning as pl
from argparse import Namespace
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.core.decorators import auto_move_data
from pytorch_lightning.utilities import rank_zero_info
from lightning_base import BaseTransformer, generic_train
from modeling_base import BaseModel
from modeling_verification import VerificationModel, VerificationJointModel
from processors import (
    fc_processors,
    fc_num_labels,
    fc_output_modes,
    compute_metrics,
    convert_examples_to_features,
    save_predictions,
)

MODEL_NAMES_MAPPING = {
    "base": BaseModel,
    "verification": VerificationModel,
    "verification-joint": VerificationJointModel,
}


class FactCheckerTransformer(BaseTransformer):
    def __init__(self, hparams, **kwargs):
        if type(hparams) == dict:
            hparams = Namespace(**hparams)
        hparams.fc_output_mode = fc_output_modes[hparams.task]
        num_labels = fc_num_labels[hparams.task]
        rank_zero_info(f"Model: {hparams.model_name}")
        model = MODEL_NAMES_MAPPING[hparams.model_name](hparams, num_labels)
        super().__init__(
            hparams,
            num_labels=num_labels,
            model=model,
            config=None if model is None else model.config,
        )

    @auto_move_data
    def forward(self, **inputs):
        return self.model(**inputs)

    def create_features(self, set_type, file_path):
        rank_zero_info(f"Creating features from '{file_path}'")
        hparams = self.hparams
        processor = fc_processors[hparams.task]()
        examples = processor.get_examples(
            file_path, set_type, self.training, hparams.use_title
        )
        num_examples = processor.get_length(file_path)
        features = convert_examples_to_features(
            examples,
            self.tokenizer,
            max_length=hparams.max_seq_length,
            task=hparams.task,
            threads=hparams.num_workers,
        )

        def empty_tensor_1():
            return torch.empty(num_examples, dtype=torch.long)

        def empty_tensor_2():
            return torch.empty((num_examples, hparams.max_seq_length), dtype=torch.long)

        input_ids = empty_tensor_2()
        attention_mask = empty_tensor_2()
        token_type_ids = empty_tensor_2()
        if hparams.fc_output_mode == "classification":
            labels = empty_tensor_1()
        elif hparams.fc_output_mode == "regression":
            labels = empty_tensor_1().float()
        indices = empty_tensor_1()
        selection_labels = None
        if self.training and "joint" in hparams.model_name:
            selection_labels = empty_tensor_1()

        for i, feature in enumerate(features):
            input_ids[i] = torch.tensor(feature.input_ids)
            attention_mask[i] = torch.tensor(feature.attention_mask)
            if feature.token_type_ids is not None:
                token_type_ids[i] = torch.tensor(feature.token_type_ids)
            labels[i] = torch.tensor(feature.label)
            indices[i] = torch.tensor(feature.index)
            if selection_labels is not None and feature.selection_label is not None:
                selection_labels[i] = torch.tensor(feature.selection_label)

        feature_list = [input_ids, attention_mask, token_type_ids, indices, labels]
        if selection_labels is not None:
            feature_list = feature_list + [selection_labels]

        if "base" not in self.hparams.model_name:
            rank_zero_info(f"Reshaping features for '{self.hparams.model_name}'")
            feature_list = reshape_features(
                feature_list, self.hparams.num_evidence, self.hparams.max_seq_length
            )
        return feature_list

    def prepare_data(self):
        if self.training:
            for set_type in ["train", "dev", "test"]:
                feature_file = self._feature_file(set_type)
                if not feature_file.exists() or self.hparams.overwrite_cache:
                    file_path = Path(self.hparams.data_dir) / f"{set_type}.tsv"
                    if not file_path.exists():
                        continue
                    feature_list = self.create_features(set_type, file_path)
                    rank_zero_info(f"Saving features to '{feature_file}'")
                    torch.save(feature_list, feature_file)

    def init_parameters(self):
        base_name = self.config.model_type  # e.g., bert, roberta, ...
        no_init = [base_name] + self.hparams.no_init
        for n, p in self.model.named_parameters():
            if not any(ni in n for ni in no_init):
                rank_zero_info(f"Initialize '{n}'")
                if "bias" not in n:
                    p.data.normal_(mean=0.0, std=self.config.initializer_range)
                else:
                    p.data.zero_()

    def get_dataloader(self, mode, batch_size):
        feature_file = self._feature_file(mode)
        if not feature_file.exists():
            return None

        rank_zero_info(f"Loading features from '{feature_file}'")
        feature_list = torch.load(feature_file)
        if self.hparams.class_weighting and mode == "train":
            labels = feature_list[4]
            assert labels.dim() == 1
            classes, samples_per_class = torch.unique(labels, return_counts=True)
            assert len(classes) == self.model.num_labels
            weights = len(labels) / (len(classes) * samples_per_class.float())
            self.class_weights = weights / weights.sum()
            rank_zero_info(f"Class weights: {self.class_weights}")
        return DataLoader(
            TensorDataset(*feature_list),
            batch_size=batch_size,
            shuffle=True if mode == "train" and self.training else False,
        )

    def build_inputs(self, batch):
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[4]}
        if len(batch) == 6:
            inputs["selection_labels"] = batch[5]
        if self.config.model_type not in {"distilbert", "bart"}:
            inputs["token_type_ids"] = (
                batch[2]
                if self.config.model_type in ["bert", "xlnet", "albert"]
                else None
            )

        if self.training and hasattr(self, "class_weights"):
            inputs["class_weights"] = self.class_weights
        return inputs

    def training_step(self, batch, batch_idx):
        inputs = self.build_inputs(batch)
        outputs = self(**inputs)
        loss = outputs[0]
        self.log_dict({"train_loss": loss, "lr": self.lr_scheduler.get_last_lr()[-1]})
        return loss

    def validation_step(self, batch, batch_idx):
        inputs = self.build_inputs(batch)
        outputs = self(**inputs)
        loss, logits = outputs[:2]
        preds = (
            torch.softmax(logits, dim=-1)
            if self.hparams.fc_output_mode == "classification"
            else logits
        )
        return {
            "loss": loss.detach().cpu(),
            "preds": preds.detach().cpu().numpy(),
            "labels": inputs["labels"].detach().cpu().numpy(),
        }

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def _eval_end(self, outputs, mode="val"):
        avg_loss = (
            torch.stack([x["loss"] for x in outputs]).mean().detach().cpu().item()
        )
        labels = np.concatenate([x["labels"] for x in outputs], axis=0)
        preds = np.concatenate([x["preds"] for x in outputs], axis=0)
        results = {
            **{"loss": avg_loss},
            **compute_metrics(self.hparams.task, preds, labels),
        }
        log_dict = {f"{mode}_{k}": torch.tensor(v) for k, v in results.items()}
        return log_dict, preds, labels

    def validation_epoch_end(self, outputs):
        log_dict, _, _ = self._eval_end(outputs)
        self.log_dict(log_dict)

    def test_epoch_end(self, outputs):
        log_dict, preds, labels = self._eval_end(outputs, mode="test")
        if "out_file" in self.hparams:
            save_predictions(self.hparams.task, preds, self.hparams.out_file)
            self.log_dict({"msg": f"Save predictions to '{self.hparams.out_file}'"})
        else:
            self.log_dict(log_dict)

    @staticmethod
    def add_model_specific_args(parser):
        BaseTransformer.add_model_specific_args(parser)
        parser.add_argument("--task", type=str, required=True)
        parser.add_argument("--overwrite_cache", action="store_true")
        parser.add_argument("--save_all_checkpoints", action="store_true")
        parser.add_argument("--max_seq_length", type=int, default=512)
        parser.add_argument("--num_evidence", type=int, default=5)
        parser.add_argument("--use_title", action="store_true")
        parser.add_argument("--aggregate_mode", type=str, default="attn")
        parser.add_argument("--word_attn", action="store_true")
        parser.add_argument("--sent_attn", action="store_true")
        parser.add_argument("--lambda_joint", type=float, default=1.0)
        parser.add_argument(
            "--attn_bias_type",
            default="none",
            choices=["none", "key_only", "value_only", "both", "dot"],
        )
        parser.add_argument("--no_init", nargs="+", default=[])
        parser.add_argument("--freeze_params", nargs="+", default=[])
        parser.add_argument("--classifier_dropout_prob", type=float, default=0.1)
        parser.add_argument("--class_weighting", action="store_true")
        return parser


def reshape_features(feature_list, num_evidence=5, max_seq_length=128):
    # feature_list = input_ids, attention_mask, token_type_ids, indices, labels, [selection_labels]
    assert len(feature_list) >= 5
    num_evidence_plus = num_evidence + 1
    assert len(feature_list[0]) % (num_evidence_plus) == 0
    num_examples = len(feature_list[0]) // (num_evidence_plus)

    # input_ids, attention_mask, token_type_ids
    for i in range(0, 3):
        feature_list[i] = feature_list[i].view(-1, num_evidence_plus, max_seq_length)
        assert feature_list[i].size(0) == num_examples

    # incdices, labels
    for i in range(3, 5):
        feature_list[i] = torch.unique(
            feature_list[i].view(-1, num_evidence_plus), dim=1
        )
        assert feature_list[i].size(0) == num_examples and feature_list[i].size(1) == 1
        feature_list[i] = feature_list[i].view(-1)

    # selection_labels
    if len(feature_list) == 6:
        feature_list[5] = feature_list[5].view(-1, num_evidence_plus)
        assert feature_list[5].size(0) == num_examples
    return feature_list


def build_args():
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = FactCheckerTransformer.add_model_specific_args(parser)
    return parser.parse_args()


def main():
    t_start = datetime.datetime.now()

    args = build_args()

    if args.seed > 0:
        pl.seed_everything(args.seed)

    model = FactCheckerTransformer(args)

    ckpt_dirpath = Path(args.default_root_dir) / "checkpoints"
    ckpt_dirpath.mkdir(parents=True, exist_ok=True)

    monitor, mode, ckpt_filename = None, "min", "{epoch}"
    dev_filepath = Path(args.data_dir) / "dev.jsonl"
    if dev_filepath.exists():
        monitor, mode = "val_acc", "max"
        ckpt_filename = "{epoch}-{" + monitor + ":.4f}"

    callbacks = []
    callbacks.append(
        ModelCheckpoint(
            dirpath=ckpt_dirpath,
            filename=ckpt_filename,
            monitor=monitor,
            mode=mode,
            save_top_k=-1 if args.save_all_checkpoints else None,
        )
    )

    if monitor is not None:
        callbacks.append(
            EarlyStopping(monitor=monitor, mode=mode, patience=args.patience)
        )

    trainer = generic_train(model, args, callbacks)

    if args.do_predict:
        trainer.test()

    t_delta = datetime.datetime.now() - t_start
    rank_zero_info(f"\nTraining took '{t_delta}'")


if __name__ == "__main__":
    main()
