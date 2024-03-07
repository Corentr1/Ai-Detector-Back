import numpy as np
from typing import Tuple
from google.cloud import storage
from keras import Model, Sequential, layers, regularizers, optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from ai_detector.params import *

def plot_history(history, title='', axs=None, exp_name=""):
    if axs is not None:
        ax1, ax2 = axs
    else:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    if len(exp_name) > 0 and exp_name[0] != '_':
        exp_name = '_' + exp_name
    ax1.plot(history.history['loss'], label = 'train' + exp_name)
    ax1.plot(history.history['val_loss'], label = 'val' + exp_name)
    ax1.set_ylim(0., 2.2)
    ax1.set_title('loss')
    ax1.legend()

    ax2.plot(history.history['accuracy'], label='train accuracy'  + exp_name)
    ax2.plot(history.history['val_accuracy'], label='val accuracy'  + exp_name)
    ax2.set_ylim(0.25, 1.)
    ax2.set_title('Accuracy')
    ax2.legend()
    plt.show()
    return (ax1, ax2)

# def save_model_to_GCS():

#     storage_filename = "####"
#     local_filename = "####"

#     client = storage.Client()
#     bucket = client.bucket(BUCKET_NAME)
#     blob = bucket.blob(storage_filename)
#     blob.upload_from_filename(local_filename)

#     print("Model saved to GCS")

def initialize_model(input_shape: tuple) -> Model:
    """
    Initialize the Neural Network with random weights
    """

    assert input_shape==(32, 32, 3), "input shape should be (32, 32, 3)"

    model = Sequential()
    model.add(layers.Conv2D(16, (1,1), input_shape=input_shape, padding='same', activation="relu"))
    model.add(layers.MaxPool2D(pool_size=(2,2)))
    model.add(layers.Conv2D(16, (2,2), padding="same", activation="relu"))
    model.add(layers.MaxPool2D(pool_size=(2,2)))
    model.add(layers.Conv2D(16, (2,2), activation="relu"))
    model.add(layers.MaxPool2D(pool_size=(2,2)))
    model.add(layers.Conv2D(16, (2,2), activation="relu"))
    model.add(layers.Flatten())
    model.add(layers.Dense(50, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1, activation='sigmoid'))

    return model


def compile_model(model: Model, learning_rate=0.0005) -> Model:
    """
    Compile the Neural Network
    """
    model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy']
              )

    return model

def train_model(
        model: Model,
        X: np.ndarray,
        y: np.ndarray,
        batch_size=64,
        patience=20,
        validation_data=None, # overrides validation_split
        validation_split=0.3,
        epochs=100
    ) -> Tuple[Model, dict]:
    """
    Fit the model and return a tuple (fitted_model, history)
    """
    filepath = "###"

    es = EarlyStopping(
        patience=patience,
        restore_best_weights=True,
        verbose=0)

    cp = ModelCheckpoint(
        filepath,
        monitor='val_loss',
        verbose=0,
        save_best_only=True,
        mode='auto',
        period=1
    )

    history = model.fit(
        X,
        y,
        validation_split = validation_split,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[es],#, cp],
        verbose=1)

    return model, history

# set to True to save model to GCS
SAVE_TO_GCS = False

if __name__ == "__main__":

    # get the data from GCS
    storage_client = storage.Client()

    # create train data
    X = []
    y = []
    blobbe_fake = storage_client.list_blobs(BUCKET_NAME_FAKE,
                                            prefix="IF-CC1M",
                                            max_results=5)
    for blob in blobbe_fake:
        string_out = blob.download_as_bytes()
        array_tensor = tf.convert_to_tensor(string_out)
        good_array = tf.io.decode_image(array_tensor)
        X.append(good_array)
        y.append(1)


    blobbe_real = storage_client.list_blobs(BUCKET_NAME_REAL,
                                            prefix="extracted",
                                            max_results=5)
    for blob in blobbe_real:
        string_out = blob.download_as_bytes()
        array_tensor = tf.convert_to_tensor(string_out)
        good_array = tf.io.decode_image(array_tensor)
        X.append(good_array)
        y.append(0)

    X = np.array(X)
    y = np.array(y)
    print('Finished')
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

    # initialize the model
    model = initialize_model(input_shape=(32,32,3))

    # compile the model
    model = compile_model(model, learning_rate=0.0005)

    # train the model
    # model, history = train_model(
    #     model,
    #     X_train,
    #     y_train,
    #     batch_size=50,
    #     patience=5,
    #     validation_data=[X_val, y_val], # overrides validation_split
    #     #validation_split=0.3,
    #     epochs=40)

    # print("model trained")
    # plot_history(history)

    # if SAVE_TO_GCS:
    #     # save the best model to GCS
    #     save_model_to_GCS()
