from sklearn.metrics import accuracy_score

def naiveBayes(X_train, y_train, dttrain, dttest, nB):
    nB.fit(X_train, dttrain)
    pred = nB.predict(y_train)
    acc = accuracy_score(dttest, pred)
    #print(acc)
    return acc