import imp
from locale import DAY_1
import torch
from vit_pytorch import ViT
from models import Model
from random import shuffle
import random
import pandas as pd
import numpy as np
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
import os


def evaluations_matrix(gs, pred):
    fpr, tpr, thresholds = metrics.roc_curve(gs, pred, pos_label=1)
    auroc = metrics.auc(fpr, tpr)
    auprc = metrics.average_precision_score(gs, pred)
    return auroc, auprc

def evaluations_matrix_binary(gs, pred):
    """
    Params
    ------
    gs: gold standards
    pred: predicted binary labels

    Yields
    ------
    sensitivity
    specificity
    precision
    F1-score
    auprc_baseline: P/N in all samples
    """
    conf_mat = confusion_matrix(gs, pred)
    tn, fp, fn, tp = conf_mat.ravel()
    auprc_baseline = (fn + tp) / (tn + fp + fn + tp)
    sensitivity = tp / (tp + fn)  # recall
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp)
    accuracy = (tn + tp) / (tn + fp + fn + tp)
    F1 = 2 * (sensitivity * precision) / (sensitivity + precision)
    return conf_mat, sensitivity, specificity, precision, accuracy, F1, auprc_baseline


device = torch.device("cuda:0")

model = Model(image_size=(1, 3000),
            patch_size=(1, 100), 
            num_classes=5,
            dim=1024,
            depth=6,
            channels=3,
            heads=16,
            mlp_dim=2048,
            dropout=0., 
            emb_dropout=0.).to(device)

BATCH_SIZE = 128

data_file = pd.read_table('../data_file.txt', header=None)
anno_file = pd.read_table('../anno_file.txt', header=None)

r=random.random
random.seed(5)
a = list(range(len(data_file)))
shuffle(a, random=r)

test_id = a[180:]
path1 = '../../sleep_edf_npy/'

filename = './sleep_edf_weight.pt'
checkpoint = torch.load(filename)
model.load_state_dict(checkpoint['ViT'])

all_test_eva = open('./eva_test.txt', 'w')
labels = ['Wake', 'N1', 'N2', 'N3', 'REM'] 
all_test_eva.write(
        'id\tlabel\tAUROC\tAUPRC\tSensitivity\tSpecificity\tPrecision\tAccuracy\tF1\tauprc_baseline\ttotal_length\n')

if not os.path.exists('specs'):
    os.mkdir('specs')

conf_matrix = np.zeros((5, 5))

for id in test_id:
    x_data = np.load(path1 + data_file.iloc[id, 0] + '.npy')
    y_data = np.load(path1 + anno_file.iloc[id, 0] + '.npy')
    sleepStart = np.min(np.argwhere(y_data != 0))
    sleepEnd = len(y_data) - np.min(np.argwhere(y_data[::-1] != 0))
    W_reserve = 120
    if sleepStart > W_reserve and len(y_data) > sleepEnd + W_reserve:
        x_data = x_data[sleepStart-W_reserve:sleepEnd+W_reserve, :, :]
        y_data = y_data[sleepStart-W_reserve:sleepEnd+W_reserve]

    pad = 1000
    d1 = len(y_data)
    x_prev = x_data[:d1-2]
    x_behind = x_data[2:]
    x_new = np.concatenate((x_prev[:, :, 3000-pad:], x_data[1:d1-1], x_behind[:, :, :pad]), axis=2)

    y_new = y_data[1: d1-1]

    x_new = torch.from_numpy(x_new).to(device)
    x_new = x_new.to(torch.float32)

    with torch.no_grad():
        model.eval()
        output_fi = model(x_new)

    _, pred_y = output_fi.max(dim=1)
    output_fi = output_fi.cpu().detach().numpy().T
    pred_y = pred_y.cpu().numpy()

    d1 = len(y_new)
    # pred_y = pred_y[:d1]

    np.save('specs/pred'+str(id), pred_y)
    np.save('specs/label'+str(id), y_new)

    output_bi = np.zeros((5, d1))
    label_bi = np.zeros((5, d1))
    for i in range(d1):
        label_bi[y_new[i], i] = 1
        output_bi[pred_y[i], i] = 1
    for i in [0,1,2,3,4]:  # Stage
        gs_i =label_bi[i,:]
        pred_i = output_fi[i,:] 
        pred_bi = output_bi[i,:]
        auroc,auprc = evaluations_matrix(gs_i, pred_i)
        _, sensitivity, specificity, precision, accuracy, F1, auprc_baseline = evaluations_matrix_binary(gs_i, pred_bi)
        all_test_eva.write('%s\t%s\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%d\n' % (
            data_file.iloc[id, 0], labels[i], auroc, auprc, sensitivity, specificity, precision, accuracy, F1, auprc_baseline,
            len(y_new)))

    for p, t in zip(pred_y, y_new):
            conf_matrix[p, t] += 1
    
print("conf_matrix", conf_matrix)
TP = np.zeros((5,))
FP = np.zeros((5,))
TN = np.zeros((5,))
FN = np.zeros((5,))
SUM = np.sum(conf_matrix)
for i in range(5):
    TP[i] = conf_matrix[i, i]
    FP[i] = np.sum(conf_matrix, axis=1)[i] - TP[i]
    TN[i] = SUM + TP[i] - np.sum(conf_matrix, axis=1)[i] - np.sum(conf_matrix, axis=0)[i]
    FN[i] = np.sum(conf_matrix, axis=0)[i] - TP[i]
accuracy = (TP + TN) / SUM
specificity = TN / (TN + FP)
sensitivity = TP / (TP + FN)
precision = TP / (TP + FP)
F1 = 2 * precision * sensitivity / (precision + sensitivity)
print("accuracy: ", accuracy)
print("specificity: ", specificity)
print("sensitivity: ", sensitivity)
print('precision: ', precision)
print('F1: ', F1)
