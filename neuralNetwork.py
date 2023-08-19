import tensorflow as tf
import pandas as pd

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

def transform_output_nominal_class_into_one_hot_encoding(dataset, classes):
    one_hot_encoded_data = pd.get_dummies(dataset)
    for i in classes:
        if i not in one_hot_encoded_data.columns:
            one_hot_encoded_data[i] = '0'
        else:
            one_hot_encoded_data[i] = one_hot_encoded_data[i].replace(['True'], '1')
            one_hot_encoded_data[i] = one_hot_encoded_data[i].replace(['False'], '0')
    
        one_hot_encoded_data[i] = one_hot_encoded_data[i].astype('int32')
    dataset = one_hot_encoded_data
    return dataset

def neuralNetwork(X_train, dttrain, y_train, dttest, classes):
    for i in X_train.columns:
        X_train[i] = X_train[i].astype('int32')
    for i in y_train.columns:
        y_train[i] = y_train[i].astype('int32')
    
    num_classes = len(classes)
    model = buildModel(X_train.shape[1], num_classes)
    
    dttrain = transform_output_nominal_class_into_one_hot_encoding(dttrain, classes)
    dttest = transform_output_nominal_class_into_one_hot_encoding(dttest, classes)

    model.fit(X_train, dttrain, epochs=10, verbose=0)
    
    loss, accuracyTrain = model.evaluate(X_train, dttrain, verbose=0)
    loss, accuracyTest = model.evaluate(y_train, dttest, verbose=0)
    return accuracyTest

def neuralNetwork1(X_train, dttrain, y_train, dttest, classes, model):
    for i in X_train.columns:
        X_train[i] = X_train[i].astype('int32')
    for i in y_train.columns:
        y_train[i] = y_train[i].astype('int32')

    dttrain = transform_output_nominal_class_into_one_hot_encoding(dttrain, classes)
    dttest = transform_output_nominal_class_into_one_hot_encoding(dttest, classes)

    model.fit(X_train, dttrain, epochs=10, verbose=0)
    
    loss, accuracyTrain = model.evaluate(X_train, dttrain, verbose=0)
    loss, accuracyTest = model.evaluate(y_train, dttest, verbose=0)
    return accuracyTest