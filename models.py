import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, confusion_matrix, \
    precision_recall_fscore_support, roc_auc_score
from tensorflow.keras import Sequential
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.layers import Dense, Conv1D, Flatten, Conv2D, MaxPooling1D,MaxPooling2D
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import numpy as np
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt


tf.random.set_seed(9999)

epochs_number = 10  
test_set_size = 0.1 
oversampling_flag = 0 
oversampling_percentage = 0.2 

# Definition of functions
def read_data():
    rawData = pd.read_csv('preprocessedR.csv')
    
    y = rawData[['FLAG']]
    X = rawData.drop(['FLAG', 'CONS_NO'], axis=1)

    print('Normal Consumers:                    ', y[y['FLAG'] == 0].count()[0])
    print('Consumers with Fraud:                ', y[y['FLAG'] == 1].count()[0])
    print('Total Consumers:                     ', y.shape[0])
    print("Classification assuming no fraud:     %.2f" % (y[y['FLAG'] == 0].count()[0] / y.shape[0] * 100), "%")

    # columns reindexing according to dates
    X.columns = pd.to_datetime(X.columns)
    X = X.reindex(X.columns, axis=1)

    # Splitting the dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y['FLAG'], test_size=test_set_size, random_state=0)
    print("Test set assuming no fraud:           %.2f" % (y_test[y_test == 0].count() / y_test.shape[0] * 100), "%\n")

    # Oversampling of minority class to encounter the imbalanced learning
    if oversampling_flag == 1:
        over = SMOTE(sampling_strategy=oversampling_percentage, random_state=0)
        X_train, y_train = over.fit_resample(X_train, y_train)
        print("Oversampling statistics in training set: ")
        print('Normal Consumers:                    ', y_train[y_train == 0].count())
        print('Consumers with Fraud:                ', y_train[y_train == 1].count())
        print("Total Consumers                      ", X_train.shape[0])

    return X_train, X_test, y_train, y_test

def results(y_test, prediction, model_name):
    print("Accuracy", 100 * accuracy_score(y_test, prediction))
    print("RMSE:", mean_squared_error(y_test, prediction, squared=False))
    print("MAE:", mean_absolute_error(y_test, prediction))
    print("F1:", 100 * precision_recall_fscore_support(y_test, prediction)[2])
    print("AUC:", 100 * roc_auc_score(y_test, prediction))
    conf_matrix = confusion_matrix(y_test, prediction)
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    fig.dpi = 400
    ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
    
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix ' + model_name, fontsize=18)
    plt.savefig('Confusion_Matrix '+ model_name + '.png')
  
def logistic_regression(X_train, X_test, y_train, y_test):
    print('Logistic Regression:')    
    model = LogisticRegression(C=1000, max_iter=1000, n_jobs=-1, solver='newton-cg')
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    results(y_test, prediction, "Logistic_Regression")


def decision_tree(X_train, X_test, y_train, y_test):
    print('Decision Tree:')
    model = DecisionTreeClassifier(random_state=0)
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    results(y_test, prediction, "Decision_Tree")


def random_forest(X_train, X_test, y_train, y_test):
    print('Random Forest:')
    model = RandomForestClassifier(n_estimators=100, min_samples_leaf=1, max_features='auto',  # max_depth=10,
                                   random_state=0, n_jobs=-1)
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    results(y_test, prediction, "Random_Forest")

def artificial_neural_network_(X_train, X_test, y_train, y_test):
    print('Artificial Neural Network:')

    # Model creation
    model = Sequential()
    model.add(Dense(1000, input_dim=1034, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer='adam',
                  metrics=['accuracy'])

    # model.fit(X_train, y_train, validation_split=0, epochs=i, shuffle=True, verbose=0)
    model.fit(X_train, y_train, validation_split=0, epochs=epochs_number, shuffle=True, verbose=1)
    temp_prediction = model.predict(X_test)
    prediction=(model.predict(X_test) > 0.5).astype("int32")
    model.summary()
    results(y_test, prediction, "ANN")


def convolutional_neural_network_1D(X_train, X_test, y_train, y_test):
    print('1D - Convolutional Neural Network:')

    # Transforming the dataset into tensors
    X_train = X_train.to_numpy().reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.to_numpy().reshape(X_test.shape[0], X_test.shape[1], 1)

    # Model creation
    model = Sequential()
    model.add(Conv1D(100, kernel_size=7, input_shape=(1034, 1), activation='relu'))
    model.add(MaxPooling1D(pool_size=7))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer='adam',
                  metrics=['accuracy'])

    # model.fit(X_train, y_train, epochs=1, validation_split=0.1, shuffle=False, verbose=1)
    model.fit(X_train, y_train, epochs=epochs_number, validation_split=0, shuffle=False, verbose=1)
    # prediction = model.predict_step(X_test)
    prediction = (model.predict(X_test) > 0.5).astype("int32")
    model.summary()
    results(y_test, prediction, "CNN 1D")


def convolutional_neural_network_2D(X_train, X_test, y_train, y_test):
    print('2D - Convolutional Neural Network:')

    # Transforming every row of the train set into a 2D array and then into a tensor
    n_array_X_train = X_train.to_numpy()
    n_array_X_train_extended = np.hstack((n_array_X_train, np.zeros(
        (n_array_X_train.shape[0], 2))))  # adding two empty columns in order to make the number of columns
    # an exact multiple of 7
    week = []
    for i in range(n_array_X_train_extended.shape[0]):
        a = np.reshape(n_array_X_train_extended[i], (-1, 7, 1))
        week.append(a)
    X_train_reshaped = np.array(week)

    # Transforming every row of the train set into a 2D array and then into a tensor
    n_array_X_test = X_test.to_numpy()  # X_test to 2D - array
    n_array_X_train_extended = np.hstack((n_array_X_test, np.zeros((n_array_X_test.shape[0], 2))))
    week2 = []
    for i in range(n_array_X_train_extended.shape[0]):
        b = np.reshape(n_array_X_train_extended[i], (-1, 7, 1))
        week2.append(b)
    X_test_reshaped = np.array(week2)

    input_shape = (1, 148, 7, 1)  # input shape of the tensor

    # Model creation
    model = Sequential()
    model.add(Conv2D(kernel_size=(7,2), filters=32, input_shape=input_shape[1:], activation='relu',
                     data_format='channels_last'))
    model.add(MaxPooling2D((7,2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(X_train_reshaped, y_train, validation_split=0.1, epochs=epochs_number, shuffle=False, verbose=1)

    prediction = (model.predict(X_test_reshaped) > 0.5).astype("int32")    
    model.summary()
    results(y_test, prediction, "CNN 2D")

# Main 
X_train, X_test, y_train, y_test = read_data()


# random_forest(X_train, X_test, y_train, y_test)
# logistic_regression(X_train, X_test, y_train, y_test)
# decision_tree(X_train, X_test, y_train, y_test)
# artificial_neural_network_(X_train, X_test, y_train, y_test)
# convolutional_neural_network_1D(X_train, X_test, y_train, y_test)
# convolutional_neural_network_2D(X_train, X_test, y_train, y_test)
