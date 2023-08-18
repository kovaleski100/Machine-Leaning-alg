from sklearn.model_selection import train_test_split
import os
import pandas as pd

drops = ["uid", " gpa all",  " cs 65"]

replys = {
    "Often": "1",
    "Rarely": "2",
    "Sometimes": "3",
    "Never": "4"
}

state = {
    "pre": "1",
    "post": "2"
}

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

def readFiles(dirlist, filesList):
    dataframeList = []
    for dirname, _, filenames in os.walk(os.path.abspath(os.getcwd())):
        for filename in filenames:
            dir = dirname.split("/")
            if dir[-1] in dirlist and filename in filesList:
                filepath = os.path.join(dirname, filename)
                dataframeList.append(pd.read_csv(filepath))
    tabelaco = pd.merge(dataframeList[0], dataframeList[1], on="uid", how="inner")
    dataframe = pd.DataFrame(tabelaco)
    dataframe = dropsColumns(dataframe)
    return dataframe


def dropsColumns(dataframe):
    for i in drops:
        dataframe = dataframe.drop([i], axis=1)
    return dataframe

def convertGrades(dataframe, field):
    dataframe[field] = dataframe[field].astype(str)
    dataframe.loc[dataframe[field] == '4', field] = '4'
    dataframe.loc[(dataframe[field] >= '3') & (dataframe[field] < '4'), field] = '3'
    dataframe.loc[(dataframe[field] >= '2') & (dataframe[field] < '3'), field] = '2'
    dataframe.loc[dataframe[field] < '2', field] = '0'
    return dataframe

def changeValueType(dataframe):
    for key in state:
        dataframe.loc[dataframe[colluns[0]] == key, colluns[0]] = state[key]
    return dataframe

def changeValueQ1(dataframe, field):
    for key in replys:
        dataframe.loc[dataframe[field] == key,field] = replys[key]
    return dataframe

def preProcess(dataframe, seed):
    dataframe = changeValueType(dataframe)
    for i in range(1, len(colluns)):
        dataframe = changeValueQ1(dataframe, colluns[i])
    dataframe = convertGrades(dataframe, " gpa 13s")
    X_train,  y_train, = train_test_split(dataframe, test_size=0.2, random_state=seed)
    return X_train, y_train