from datetime import datetime
from tensorflow import keras
from keras.models import Sequential
from keras import Input
from keras.layers import Dense
import pandas as pd
import numpy as np
from numpy import mean
from numpy import std
import sklearn
from sklearn.model_selection import RepeatedKFold


def get_dataset(x_path, y_path):
    # Read pickle dataset
    X = pd.read_pickle(x_path)
    y = pd.read_pickle(y_path)
    X.astype(np.float32)
    y.astype(np.float32)
    y = y[['T30', 'P40', 'Wf']] # Reduce the number of outputs
    return X, y


def get_model(n_input, n_output):
    # Structure of the neural network
    model = Sequential(name="Healthy-signal-estimator")  # Model
    model.add(Input(shape=(n_input,), name='Input-Layer'))  # Input Layer
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform', name='Hidden-Layer'))  # Hidden Layer
    model.add(Dense(n_output, name='Output-Layer'))  # Output Layer

    # Trying to replicate what the paper proposed
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=5e-4,
        decay_steps=1000,
        decay_rate=1e-3)
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

    epochs = 50
    learning_rate = 0.1
    decay_rate = learning_rate / epochs
    momentum = 0.8
    sgd = keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)

    # Compile the keras model
    model.compile(optimizer='adam', loss='mae')
    return model


def evaluate_model(X, y):
    results = list()
    models = list()
    n_inputs, n_outputs = X.shape[1], y.shape[1]
    # 10-fold cross validation
    cv = RepeatedKFold(n_splits=10, n_repeats=1, random_state=1)
    for train_ix, test_ix in cv.split(X):
        X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
        y_train, y_test = y.iloc[train_ix], y.iloc[test_ix]
        # Define the model
        model = get_model(n_inputs, n_outputs)
        X_train = X_train.astype(np.float32)
        y_train = y_train.astype(np.float32)
        X_test = X_test.astype(np.float32)
        y_test = y_test.astype(np.float32)
        model.fit(X_train, y_train, epochs=150)
        # Evaluate the model
        mae = model.evaluate(X_test, y_test)
        print('>%.3f' % mae)
        results.append(mae)
        models.append(model)
    return results, models


X, y = get_dataset('dataset/train_X_merged.pkl', 'dataset/train_Y_merged.pkl')
results, models = evaluate_model(X, y)
# summarize performance
print('MAE: %.3f (%.3f)' % (mean(results), std(results)))
min_mae = min(results)
print(str(min_mae))
min_index = results.index(min_mae)
model_to_save = models[min_index]
# Save the model
# datetime object containing current date and time
now = datetime.now()
dt_string = now.strftime("%d_%m_%Y-%H_%M_%S")
model_to_save.save('model/model_' + dt_string + '_mae_' + str(min_mae))