# Copyright (c) 2021, Yamagishi Laboratory, National Institute of Informatics
# Author: Canasai Kruengkrai (canasai@nii.ac.jp)
# All rights reserved.
#
# Adapted from https://github.com/huggingface/transformers/blob/master/examples/legacy/pytorch-lightning/lightning_base.py # noqa: E501

import math
import warnings
import pytorch_lightning as pl
from pathlib import Path
from pytorch_lightning.utilities import rank_zero_info
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from transformers.optimization import (
    AdamW,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)
from adafactor import Adafactor

warnings.filterwarnings("ignore")

arg_to_scheduler = {
    "linear": get_linear_schedule_with_warmup,
    "cosine": get_cosine_schedule_with_warmup,
}


class BaseTransformer(pl.LightningModule):
    def __init__(
        self,
        hparams,
        num_labels=None,
        config=None,
        tokenizer=None,
        model=None,
        **config_kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(hparams)
        if config is None:
            self.config = AutoConfig.from_pretrained(
                self.hparams.pretrained_model_name,
                **({"num_labels": num_labels} if num_labels is not None else {}),
            )
        else:
            self.config = config

        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.hparams.pretrained_model_name, use_fast=True
            )
        else:
            self.tokenizer = tokenizer

        if model is None:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.hparams.pretrained_model_name, config=self.config
            )
        else:
            self.model = model

    def _feature_file(self, mode):
        return (
            Path(self.hparams.data_dir)
            / f"cached_{mode}_{self.hparams.pretrained_model_name}_{self.hparams.max_seq_length}"
        )

    def init_parameters(self):
        raise NotImplementedError("You must implement this for your task")

    def get_dataloader(self, mode, batch_size):
        raise NotImplementedError("You must implement this for your task")

    def setup(self, stage):
        if self.training and stage == "fit":
            self.init_parameters()
            self.train_loader = self.get_dataloader(
                "train",
                self.hparams.train_batch_size,
            )
            num_devices = max(1, self.hparams.gpus)
            effective_batch_size = (
                self.hparams.train_batch_size
                * self.hparams.accumulate_grad_batches
                * num_devices
            )
            dataset_size = len(self.train_loader.dataset)
            self.total_steps = (
                dataset_size / effective_batch_size
            ) * self.hparams.max_epochs

        else:
            self.total_steps = 0

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.get_dataloader("dev", self.hparams.eval_batch_size)

    def test_dataloader(self):
        return self.get_dataloader("test", self.hparams.eval_batch_size)

    def get_lr_scheduler(self):
        get_schedule_func = arg_to_scheduler[self.hparams.lr_scheduler]
        if self.hparams.warmup_ratio > 0 and self.hparams.warmup_steps == 0:
            assert self.hparams.warmup_ratio > 0 and self.hparams.warmup_ratio <= 1
            self.hparams.warmup_steps = math.ceil(
                self.total_steps * self.hparams.warmup_ratio
            )
            rank_zero_info(f"warmup_ratio = {self.hparams.warmup_ratio:.2f}")
            rank_zero_info(f"warmup_steps = {self.hparams.warmup_steps:.2f}")
            rank_zero_info(f"total_steps = {self.total_steps:.2f}")

        self.lr_scheduler = get_schedule_func(
            self.opt,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.total_steps,
        )
        scheduler = {"scheduler": self.lr_scheduler, "interval": "step", "frequency": 1}
        return scheduler

    def configure_optimizers(self):
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        if self.hparams.adafactor:
            optimizer = Adafactor(
                optimizer_grouped_parameters,
                lr=self.hparams.learning_rate,
                scale_parameter=False,
                relative_step=False,
            )
        else:
            optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=self.hparams.learning_rate,
                eps=self.hparams.adam_epsilon,
            )
        self.opt = optimizer
        scheduler = self.get_lr_scheduler()
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument(
            "--pretrained_model_name", type=str, default="bert-base-uncased"
        )
        parser.add_argument("--model_name", type=str, default="base")
        parser.add_argument("--data_dir", type=str, required=True)
        parser.add_argument("--learning_rate", type=float, default=5e-5)
        parser.add_argument("--lr_scheduler", type=str, default="linear")
        parser.add_argument("--weight_decay", type=float, default=0.0)
        parser.add_argument("--adam_epsilon", type=float, default=1e-6)
        parser.add_argument("--warmup_ratio", type=float, default=0.0)
        parser.add_argument("--warmup_steps", type=int, default=0)
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--train_batch_size", type=int, default=32)
        parser.add_argument("--eval_batch_size", type=int, default=32)
        parser.add_argument("--adafactor", action="store_true")
        parser.add_argument("--seed", type=int, default=3435)
        parser.add_argument("--patience", type=int, default=2)
        parser.add_argument("--do_predict", action="store_true")


def generic_train(model, args, callbacks):
    train_params = {}

    if args.gpus > 1:
        train_params["distributed_backend"] = "ddp"

    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=callbacks,
        **train_params,
    )

    trainer.fit(model)

    return trainer
