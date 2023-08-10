from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

colluns = ["type",
           "1. I feel in tune with the people around me",
           "2. I lack companionship",
           "3. There is no one I can turn to",
           "4. I do not feel alone",
           "5. I feel part of a group of friends",
           "6. I have a lot in common with the people around me",
           "7. I am no longer close to anyone",
           "8. My interests and ideas are not shared by those around me",
           "9. I am an outgoing person",
           "10. There are people I feel close to",
           "11. I feel left out",
           "12. My social relationships are superficial",
           "13. No one really knows me well",
           "14. I feel isolated from others",
           "15. I can find companionship when I want it",
           "16. There are people who really understand me",
           "17. I am unhappy being so withdrawn",
           "18. People are around me but not with me",
           "19. There are people I can talk to",
           "20. There are people I can turn to"
]

replys = {
    "Often": 1,
    "Rarely": 2,
    "Sometimes": 3,
    "Never": 4
}

state = {
    "pre": 1,
    "post": 2
}


def models(train, test):
    X_train, X_test, y_train, y_test = train_test_split(train, test, test_size=0.2, random_state=42)
    pca = PCA(n_components=2)  # Defina o número de componentes principais desejados
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    DT = DecisionTreeClassifier()
    DT.fit(X_train_pca, y_train)

    # Fazer previsões usando os dados de teste reduzidos pelo PCA
    y_pred = DT.predict(X_test_pca)

    # Calcular a acurácia
    accuracy = accuracy_score(y_test, y_pred)
    print("Acurácia após PCA:", accuracy)


def changeValueType(dataframe):
    for key in state:
        dataframe.loc[dataframe[colluns[0]] == key, colluns[0]] = state[key]
    return dataframe

def changeValueQ1(dataframe, field):
    for key in replys:
        dataframe.loc[dataframe[field] == key,field] = replys[key]
    return dataframe

def preProcess(dataframe):
    dataframe = changeValueType(dataframe)
    for i in range(1, len(colluns)):
        dataframe = changeValueQ1(dataframe, colluns[i])
    types = dataframe.groupby(colluns[0])
    x = types.get_group(state["pre"])
    y = types.get_group(state["post"])
    #dataframe.loc[dataframe["1. I feel in tune with the people around me"]]
    return x, y
def main():
    dirlist = ["education", "survey"]
    filesList = ["grades.csv", "LonelinessScale.csv"]

    dataframeList = []

    for dirname, _, filenames in os.walk(os.path.abspath(os.getcwd())):
        for filename in filenames:
            dir = dirname.split("/")
            if dir[-1] in dirlist and filename in filesList:
                filepath = os.path.join(dirname, filename)
                dataframeList.append(pd.read_csv(filepath))

    tabelaco = pd.merge(dataframeList[0], dataframeList[1], on="uid", how="inner")
    dataframe = pd.DataFrame(tabelaco)
    train, test = preProcess(dataframe)

    
    models(train, test)

if __name__ == "__main__":
    main()

