#!/usr/bin/env python
# coding: utf-8


import numpy as np
data1 = np.load("eval_results_ENC003left.npz")
all_preds1 = data1["all_preds"]
all_labels1 = data1["all_labels"]
y_score1 = data1["y_score"]


data2 = np.load("eval_results_ENC003left_len500.npz")
all_preds2 = data2["all_preds"]
all_labels2 = data2["all_labels"]
y_score2 = data2["y_score"]


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
fpr1, tpr1, _ = roc_curve(all_labels1, y_score1)
roc_auc1 = auc(fpr1, tpr1)

fpr2, tpr2, _ = roc_curve(all_labels2, y_score2)
roc_auc2 = auc(fpr2, tpr2)

plt.figure(figsize=(6,6))
plt.plot(fpr1, tpr1, color='blue', lw=2, label=f'100bp ROC (AUC = {roc_auc1:.4f})')
plt.plot(fpr2, tpr2, color='red', lw=2, label=f'500bp ROC (AUC = {roc_auc2:.4f})')
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')  # 随机猜测
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()


from sklearn.metrics import precision_recall_curve, average_precision_score
precision1, recall1, _ = precision_recall_curve(all_labels1, y_score1)
ap1 = average_precision_score(all_labels1, y_score1)

precision2, recall2, _ = precision_recall_curve(all_labels2, y_score2)
ap2 = average_precision_score(all_labels2, y_score2)

plt.figure(figsize=(6,6))
plt.plot(recall1, precision1, color='blue', lw=2, label=f'100bp (AP = {ap1:.4f})')
plt.plot(recall2, precision2, color='red', lw=2, label=f'500bp (AP = {ap2:.4f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve Comparison')
plt.legend(loc="lower left")
plt.grid(True)
plt.show()





