from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def decisionTree(X_train, y_train):
    DT = DecisionTreeClassifier()
    dataframeGPAtrain = X_train[" gpa 13s"]
    del X_train[" gpa 13s"]
    DT.fit(X_train, dataframeGPAtrain)

    dataframeGPAtest = y_train[" gpa 13s"]
    del y_train[" gpa 13s"]
    
    y_pred = DT.predict(y_train)

    accuracy = accuracy_score(dataframeGPAtest, y_pred)
    #print("Acurácia:", accuracy)
    return dataframeGPAtrain, dataframeGPAtest, accuracy