import numpy as np
import pandas as pd
# from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--base')
parser.add_argument('--new')
args = parser.parse_args()

# prediction = pd.read_json("output/" + args.new + ".jsonl", lines=True)
prediction = pd.read_json(args.new, lines=True)
"""
for i, row in prediction.iterrows():
    if row["id"] != row["id"]:
        # print(i)
        prediction.at[i, 'id'] = i
# prediction[prediction["id"] != prediction["id"]]["id"] = prediction[prediction["id"] != prediction["id"]].index
# print(prediction[prediction["id"] != prediction["id"]].shape[0])
prediction["id"] = prediction["id"].astype(str).apply(lambda x: x[1:]).astype(float).astype(int)
# print(prediction.tail())
"""
golden = pd.read_json("/dfs/user/yibingdu/fever-resources/KernelGAT/data/golden_dev.json", lines=True)
# print(golden.head())
golden["id"] = golden["id"].astype(int)
compare = prediction.merge(golden, on="id", how="left")
print(compare.shape)
print(compare.head())

pred = compare["predicted_label"].tolist()
true = compare["label"].tolist()

# print(pred[:5])
# print(true[:5])

conf_mat = confusion_matrix(true, pred, labels=["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"])
print(conf_mat)
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

# original = pd.read_json("output/" + args.base + ".jsonl", lines=True)
original = pd.read_json(args.base, lines=True)
change = prediction.merge(original, on="id", how="left")
curr = change["predicted_label_x"].tolist()
old = change["predicted_label_y"].tolist()
conf_mat = confusion_matrix(old, curr, labels=["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"])
print("Old and current predictions")
print(conf_mat)
