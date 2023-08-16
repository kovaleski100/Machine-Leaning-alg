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
import tensorflow as tf

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

def buildModel(shape, num_classes):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(21, activation='relu', input_shape=(shape,)),
        tf.keras.layers.Dense(63, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  metrics=['accuracy'])
    return model

# one-hot encoding function
def transform_output_nominal_class_into_one_hot_encoding(dataset, classes):
    # create two classes based on the single class
    #print(dataset)
    one_hot_encoded_data = pd.get_dummies(dataset)
    for i in classes:
        if i not in one_hot_encoded_data.columns:
            one_hot_encoded_data[i] = '0'
        else:
            one_hot_encoded_data[i] = one_hot_encoded_data[i].replace(['True'], '1')
            one_hot_encoded_data[i] = one_hot_encoded_data[i].replace(['False'], '0')
    
        one_hot_encoded_data[i] = one_hot_encoded_data[i].astype('int32')
    dataset = one_hot_encoded_data
    print(one_hot_encoded_data)
    return dataset

def neuralNetwork(X_train, dttrain, y_train, dttest, classes):
#    print(y_train)

    # for label in y_train:
    #    print(label)

    for i in X_train.columns:
        X_train[i] = X_train[i].astype('int32')
    for i in y_train.columns:
        y_train[i] = y_train[i].astype('int32')
    #print(X_train.describe())
    #print(X_train.shape[1])
    
    num_classes = len(classes)
    model = buildModel(X_train.shape[1], num_classes)
    
    # Convertendo os rótulos em one-hot encoding
    dttrain = transform_output_nominal_class_into_one_hot_encoding(dttrain, classes)
    #print(dttrain)

    #print(dttest.info())
    print(dttest)
    dttest = transform_output_nominal_class_into_one_hot_encoding(dttest, classes)
    #y_train_one_hot = tf.keras.utils.to_categorical(X_train, num_classes=num_classes)
    #dttest_one_hot = tf.keras.utils.to_categorical(dttrain, num_classes=num_classes)
    
    model.fit(X_train, dttrain, epochs=10)
    
    loss, accuracy = model.evaluate(X_train, dttrain)
    print("Training set Accuracy: {:5.2f}%".format(accuracy * 100))
    
    #print(X_train, dttrain)
    print(y_train.shape[1], dttest.shape[1])
    loss, accuracy = model.evaluate(y_train, dttest)
    print("Testing set Accuracy: {:5.2f}%".format(accuracy * 100))





def decisionTree(X_train, y_train):
    #X_train, X_test, y_train, y_test = train_test_split(train, test, test_size=0.2, random_state=42)
    #pca = PCA(n_components=2)  # Defina o número de componentes principais desejados
    #X_train_pca = pca.fit_transform(X_train)
    #X_test_pca = pca.transform(X_test)
    DT = DecisionTreeClassifier()
    dataframeGPAtrain = X_train[" gpa 13s"]
    del X_train[" gpa 13s"]
    #print(y_train)
    DT.fit(X_train, dataframeGPAtrain)

    dataframeGPAtest = y_train[" gpa 13s"]
    del y_train[" gpa 13s"]
    
    #print(y_train)
    #print(dataframeGPA)
    # Fazer previsões usando os dados de teste reduzidos pelo PCA
    y_pred = DT.predict(y_train)

    # Calcular a acurácia
    accuracy = accuracy_score(dataframeGPAtest, y_pred)
    print("Acurácia:", accuracy)
    return dataframeGPAtrain, dataframeGPAtest

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

def preProcess(dataframe):
    dataframe = changeValueType(dataframe)
    for i in range(1, len(colluns)):
        dataframe = changeValueQ1(dataframe, colluns[i])
    dataframe = convertGrades(dataframe, " gpa 13s")
    # types = dataframe.groupby(colluns[0])
    # x = types.get_group(state["pre"])
    # y = types.get_group(state["post"])

    #dataframe.loc[dataframe["1. I feel in tune with the people around me"]]
    #return x, y
    X_train,  y_train, = train_test_split(dataframe, test_size=0.2)
    return X_train, y_train

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
    for i in drops:
        dataframe = dataframe.drop([i], axis=1)

    train, test = preProcess(dataframe)
    #decisionTree(train, test)
    dttrain, dttest = decisionTree(train, test)
    classes = ["0", "2", "3", "4.0"]
    neuralNetwork(train, dttrain, test, dttest, classes)

if __name__ == "__main__":
    main()

