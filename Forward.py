from cProfile import label
import torch
import numpy as np
import models
from torch.nn.functional import binary_cross_entropy

result_file = "Wres120_15_35_2_2_5_depth6_gradLr.txt"

def train(optimizer, criterion1,criterion2, train_loader, model, device):
    model.train()
    running_loss, correct = 0, 0
    conf_matrix = np.zeros((5, 5))
    batch = 0

    for idx, (inputs, labels) in enumerate(train_loader):
        batch += 1
        optimizer.zero_grad()
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        #######
        loss1 = criterion1(outputs, labels)
        loss2 = criterion2(outputs, labels)
        loss = loss1 + 3 * loss2
        #######
        running_loss += loss.item()
        loss.backward()
        optimizer.step()

        _, prediction = outputs.max(dim=1)
        correct += (prediction == labels).sum().item()

        for p, t in zip(prediction, labels):
            conf_matrix[p, t] += 1

    conf_matrix = np.transpose(conf_matrix)

    return 


# def test(criterion1, criterion2, test_loader, model1, model2, device):
def test(criterion1, criterion2, test_loader, model, device):
    model.eval()
    test_loss, correct = 0, 0
    conf_matrix = np.zeros((5, 5))
    with torch.no_grad():
        batch = 0
        for idx, (inputs, labels) in enumerate(test_loader):
            batch += 1
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            #######
            loss1 = criterion1(outputs, labels)
            loss2 = criterion2(outputs, labels)
            loss = loss1 + 3 * loss2
            #######
            test_loss += loss.item()

            _, prediction = outputs.max(dim=1)
            correct += (prediction == labels).sum().item()

            for p, t in zip(prediction, labels):
                conf_matrix[p, t] += 1

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
    f1 = 2 * precision * sensitivity / (precision + sensitivity)
    overall = np.sum(TP) / SUM

    with open(result_file, 'a+') as f:
        f.write("accuracy: " + str(accuracy) + "\n")
        f.write("specificity: " + str(specificity) + "\n")
        f.write("sensitivity: " + str(sensitivity) + "\n")
        f.write("precision: " + str(precision) + "\n")
        f.write("f1 score: " + str(f1) + "\n")
        f.write("Overall Accuracy: " + str(overall) + '\n')
        f.write("@@@@@@@@@@@@@@@@@@@@@@@@@@@\n")

    conf_matrix = np.transpose(conf_matrix)
    return test_loss

