import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--file')
args = parser.parse_args()

prediction = pd.read_json(args.file, lines=True)
golden = pd.read_json("data/golden_dev.json", lines=True)
golden["id"] = golden["id"].astype(int)
compare = prediction.merge(golden, on="id", how="left")
print(compare.shape)

pred = compare["predicted_label"].tolist()
true = compare["label"].tolist()

conf_mat = confusion_matrix(true, pred, labels=["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"])
print(conf_mat)
conf_mat_list = conf_mat.tolist()
if sum(conf_mat_list[0]) != 0:
    sup_acc = conf_mat_list[0][0] / sum(conf_mat_list[0])
    print(sup_acc)
if sum(conf_mat_list[1]) != 0:
    ref_acc = conf_mat_list[1][1] / sum(conf_mat_list[1])
    print(ref_acc)
if sum(conf_mat_list[2]) != 0:
    nei_acc = conf_mat_list[2][2] / sum(conf_mat_list[2])
    print(nei_acc)
