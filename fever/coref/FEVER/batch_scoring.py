import numpy as np
import pandas as pd
# from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--filename')
args = parser.parse_args()

# prediction = pd.read_json("./output/dev.json", lines=True)
prediction = pd.read_json("output/" + args.filename + ".jsonl", lines=True)
# prediction = pd.read_json("./output_levels/temp.json", lines=True)
golden = pd.read_json("/dfs/user/yibingdu/fever-resources/KernelGAT/data/golden_dev.json", lines=True)
compare = prediction.merge(golden, on="id", how="left")
pred = compare["predicted_label"].tolist()
true = compare["label"].tolist()

conf_mat = confusion_matrix(true, pred, labels=["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"])
print(conf_mat)
print("Total:", sum([sum(c) for c in conf_mat]))
conf_mat_list = conf_mat.tolist()
if sum(conf_mat_list[0]) != 0:
    sup_acc = conf_mat_list[0][0] / sum(conf_mat_list[0])
    print("SUPPORTS accuracy: ", sup_acc)
if sum(conf_mat_list[1]) != 0:
    ref_acc = conf_mat_list[1][1] / sum(conf_mat_list[1])
    print("REFUTES accuracy: ", ref_acc)
if sum(conf_mat_list[2]) != 0:
    nei_acc = conf_mat_list[2][2] / sum(conf_mat_list[2])
    print("NOT ENOUGH INFO accuracy: ", nei_acc)
