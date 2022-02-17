# Copyright (c) 2021, Yamagishi Laboratory, National Institute of Informatics
# Author: Canasai Kruengkrai (canasai@nii.ac.jp)
# All rights reserved.

import argparse
import datetime
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from train import FactCheckerTransformer


def build_args():
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument("--checkpoint_file", type=str, required=True)
    parser.add_argument("--in_file", type=str, required=True)
    parser.add_argument("--out_file", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    return args


def main():
    t_start = datetime.datetime.now()
    args = build_args()

    model = FactCheckerTransformer.load_from_checkpoint(
        checkpoint_path=args.checkpoint_file
    )
    model.hparams.out_file = args.out_file
    model.eval()
    model.freeze()

    params = {}
    params["precision"] = model.hparams.precision

    trainer = pl.Trainer.from_argparse_args(
        args, logger=False, checkpoint_callback=False, **params
    )

    test_file_path = Path(args.in_file)
    if not test_file_path.exists():
        raise RuntimeError(f"Cannot find '{test_file_path}'")
    feature_list = model.create_features("test", test_file_path)
    test_dataloader = DataLoader(
        TensorDataset(*feature_list),
        batch_size=args.batch_size,
        shuffle=False,
    )

    trainer.test(model, test_dataloader)

    t_delta = datetime.datetime.now() - t_start
    rank_zero_info(f"\nPrediction took '{t_delta}'")


if __name__ == "__main__":
    main()
