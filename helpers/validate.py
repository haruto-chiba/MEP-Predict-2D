import os
import glob
from natsort import natsorted
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
import pandas as pd


ROOTPATH = f"../../../../../mnt/share/Takase/Result_DeepLearning2D"
SUBJECT = "Aoki"
MUSCLE = 1
outputsFolder = os.path.join(f"outputs_{SUBJECT}", f"muscle{MUSCLE}")

targetFiles = glob.glob(
    os.path.join(ROOTPATH, outputsFolder, "*Binary_Standardized_*/*/*/*.csv")
)
targetFiles = natsorted(targetFiles)

min_loss = 10**8

for path in targetFiles:
    score = float(path.split("/")[-1].split("=")[-1][:-4])
    if score < min_loss:
        min_loss_path = path
        min_loss = score

testTargetFile = glob.glob(
    "/".join(min_loss_path.replace("outputs", "test").split("/")[:-1]) + "/*"
)[0]

print(testTargetFile)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


result = pd.read_csv(testTargetFile)
outputs = result["output"].values
targets = result["target"].values


probs = [sigmoid(o) for o in outputs]
outputs = [1.0 if sigmoid(o) >= 0.2 else 0.0 for o in outputs]

outputs = np.array(outputs)
targets = np.array(targets)

accuracy = (outputs == targets).mean()
print("accuracy", accuracy)
tp = np.sum((outputs == 1) & (targets == 1))
fp = np.sum((outputs == 1) & (targets == 0))
tn = np.sum((outputs == 0) & (targets == 0))
fn = np.sum((outputs == 0) & (targets == 1))

print("TP:", tp, "FP:", fp, "FN:", fn, "TN:", tn)

# precision, recall, F1
precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

print("precision:", precision)
print("recall:", recall)
print("f1:", f1)

auc = roc_auc_score(targets, probs)
ap = average_precision_score(targets, probs)

print("ROC-AUC:", auc)
print("PR-AUC:", ap)
