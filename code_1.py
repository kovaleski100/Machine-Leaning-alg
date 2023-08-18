import pandas as pd
from preProcess import preProcess, dropsColumns, readFiles
from neuralNetwork import neuralNetwork
from naive import naiveBayes
from dt import decisionTree
import math as mt

def main():

    dirlist = ["education", "survey"]
    filesList = ["grades.csv", "LonelinessScale.csv"]

    dataframe = readFiles(dirlist, filesList)
    dic = dict()
    for seed in [590]:
        train, test = preProcess(dataframe, seed)
        classes = ["0", "2", "3", "4.0"]
        dttrain, dttest, accDT = decisionTree(train, test)
        accNN = neuralNetwork(train, dttrain, test, dttest, classes)
        accNB = naiveBayes(train, test, dttrain, dttest)
        print(seed/999*100)

        print("acc dos modelos")
        print(accDT)
        print(accNN)
        print(accNB)
        #dic[seed] = mt.pow(accDT * accNN * accNB, 1/3)
    # bestToModel = (0,0)
    # for i in dic:
    #     if(dic[i] > bestToModel[1]):
    #         bestToModel = (i,dic[i])
            #bestToModel[j][0] = i

    # print(bestToModel)

if __name__ == "__main__":
    main()

