from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import pandas as pd
from preProcess import preProcess, readFiles
from neuralNetwork import neuralNetwork1, buildModel, transform_output_nominal_class_into_one_hot_encoding
from naive import naiveBayes
from dt import decisionTree, decisionTreeK
import math as mt
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def confusion_matrix(y_true, y_pred, num_classes):
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    print(y_true, y_pred)
    count = 0
    for i in range(len(y_true)):
        true_class = int(float(y_true[i])) - 1
        pred_class = int(float(y_pred[i])) - 1

        if(true_class<0):
            true_class = 0
        if(pred_class<0):
            pred_class = 0
        
        if int(float(true_class)) < num_classes and int(float(pred_class))   < num_classes:
            matrix[int(float(true_class)), int(float(pred_class))] += 1
        else:
            print(true_class, pred_class)
            count = count + 1
    print(count)
    print()
    print()
    return matrix

def plot_confusion_matrix(matrix, class_names):
    plt.figure(figsize=(len(class_names), len(class_names)+1))
    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(j, i, str(matrix[i, j]), ha="center", va="center", color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.show()

def linearRegression(X_train, y_train, X_test, dttest,model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_predLR = y_pred
    for i in range(len(y_pred)):
        y_predLR[i] = int(y_pred[i])
    return accuracy_score(dttest, y_predLR)

def kfold_cross_validation(X, y, classes, y_train, dttest, n_splits=5, epochs=10):
    num_samples = len(X)
    fold_size = num_samples // n_splits

    num_classes = len(classes)

    model = buildModel(X.shape[1], num_classes)
    accByModel = []
    accuracyBymodelTrain  = []
    dt = DecisionTreeClassifier()
    nB = GaussianNB()
    LR = LinearRegression()
    mConf = []
    class_mapping = {class_name: index for index, class_name in enumerate(classes)}
    
    for modelI in range(4):
        accuracies = []
        for i in range(n_splits):
            start = i * fold_size
            end = (i + 1) * fold_size if i < n_splits - 1 else num_samples

            X_train = pd.concat([X.iloc[:start], X.iloc[end:]]).reset_index(drop=True)
            dttrain = pd.concat([y.iloc[:start], y.iloc[end:]]).reset_index(drop=True)
            X_test = X.iloc[start:end].reset_index(drop=True)
            dttestK = y.iloc[start:end].reset_index(drop=True)
            
            dttest_mapped = [class_mapping.get(class_name, -1) for class_name in dttestK]

            accuracy = 0
            if(modelI == 0):
                accuracy = decisionTreeK(X_train, dttrain, X_test, dttest_mapped, dt)
            elif(modelI == 2):
                accuracy = linearRegression(X_train, dttrain, X_test, dttest_mapped, LR)
            elif(modelI == 1):
                accuracy = naiveBayes(X_train, X_test, dttrain, dttest_mapped, nB)
            else:
                accuracy = neuralNetwork1(X_train, dttrain, X_test, dttestK, classes, model)
            accuracies.append(accuracy)
        accuracyBymodelTrain.append(accuracies)
            
        accuracyTest = 0
        dttest_mapped = [class_mapping.get(class_name, -1) for class_name in dttest]
        if(modelI == 0):
            y_pred = dt.predict(y_train)
            accuracyTest = accuracy_score(dttest_mapped, y_pred)
            mConf.append(confusion_matrix(dttest_mapped, y_pred, len(classes)))
        elif(modelI == 2):
            y_pred = LR.predict(y_train)
            y_predLR = y_pred
            for i in range(len(y_pred)):
                y_predLR[i] = int(y_pred[i])
            accuracyTest = accuracy_score(dttest_mapped, y_pred)
            mConf.append(confusion_matrix(dttest_mapped, y_pred, len(classes)))
        elif(modelI == 1):
            y_pred = nB.predict(y_train)
            accuracyTest = accuracy_score(dttest_mapped, y_pred)
            mConf.append(confusion_matrix(dttest_mapped, y_pred, len(classes)))
        else:
            y_traink = y_train
            for i in y_traink.columns:
                y_traink[i] = y_traink[i].astype('int32')
            y_pred = model.predict(y_traink)
            dttest_encoded = transform_output_nominal_class_into_one_hot_encoding(dttest, classes)
            loss, accuracyTest = model.evaluate(y_traink, dttest_encoded, verbose=0)
        accByModel.append(accuracyTest)
        
    return accByModel, accuracyBymodelTrain, mConf


def main():

    dirlist = ["education", "survey"]
    filesList = ["grades.csv", "LonelinessScale.csv"]

    dataframe = readFiles(dirlist, filesList)
    dic = dict()
    for seed in [590]:
        train, test = preProcess(dataframe, seed)
        classes = ["0", "2", "3", "4.0"]
        dttrain, dttest, accDT = decisionTree(train, test)

        #print(test, dttest)

        accNN, accuracyBymodelTrain ,mconf = kfold_cross_validation(train, dttrain, classes, test, dttest, n_splits=5)
        #accNN1 = neuralNetwork(train, dttrain, test, dttest, classes)
        #accNB = naiveBayes(train, test, dttrain, dttest)
        #print(seed/999*100)

        print("Acc dos modelos")
        print("Decision Tree:", accNN[0], "Naive Bayes:", accNN[2], "Neural Network:", accNN[1])


        print(accuracyBymodelTrain)

        for i, conf_matrix in enumerate(mconf):
            print("acc final", accNN[i])
            print("Confusion Matrix for Model", i)
            plot_confusion_matrix(conf_matrix, classes)
            print("acc media",np.mean(accuracyBymodelTrain[i]))
            print("desvio padrao",np.std(accuracyBymodelTrain[i]))
            print()
        print("acc final", accNN[3])
        #print(accNB)
        #dic[seed] = mt.pow(accDT * accNN * accNB, 1/3)
    # bestToModel = (0,0)
    # for i in dic:
    #     if(dic[i] > bestToModel[1]):
    #         bestToModel = (i,dic[i])
            #bestToModel[j][0] = i

    # print(bestToModel)

if __name__ == "__main__":
    main()

