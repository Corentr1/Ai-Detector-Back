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

    assert input_shape==(256, 256, 3), "input shape should be (32, 32, 3)"

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
        validation_data=None,
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
        callbacks=[es, cp],
        verbose=1)

    return model, history

def load_and_preprocess_image(file_path, new_height, new_width):
    # Load the image
    image = tf.io.read_file(file_path)
    # Decode the image
    image = tf.image.decode_jpeg(image, channels=3)
    # Resize the image
    image = tf.image.resize(image, [new_height, new_width])
    # Preprocess the image (e.g., normalize the pixel values)
    image = tf.image.per_image_standardization(image)
    return image

def blob_to_url(blob):
    return blob.download_url

# set to True to save model to GCS
SAVE_TO_GCS = False

if __name__ == "__main__":

    # get the data from GCS
    storage_client = storage.Client()

    #size images
    new_height = 256
    new_width =256

    fake_files = tf.data.Dataset.list_files(f"gs://{BUCKET_NAME_FAKE_SAMPLE}/*")
    real_files = tf.data.Dataset.list_files(f"gs://{BUCKET_NAME_REAL_SAMPLE}/*")

    # Map the function over the dataset
    fake_files = fake_files.map(lambda x: load_and_preprocess_image(x, new_height, new_width))
    real_files = real_files.map(lambda x: load_and_preprocess_image(x, new_height, new_width))

    # Create a dataset containing both the real and fake images with their corresponding labels
    labels = tf.data.Dataset.from_tensor_slices(tf.constant([1]*len(fake_files) + [0]*len(real_files)))
    files = fake_files.concatenate(real_files)
    dataset = tf.data.Dataset.zip((files, labels))

    dataset = dataset.shuffle(buffer_size=200) # shuffle the dataset
    dataset = dataset.batch(50)
    dataset = dataset.prefetch(1)

   # split the dataset into training and validation sets
    dataset_size = dataset.reduce(0, lambda x, _: x+1).numpy()
    train_size = int(0.7 * dataset_size)
    val_size = dataset_size - train_size
    train_dataset = dataset.take(train_size)
    val_dataset = dataset.skip(train_size).take(val_size)

    # initialize the model
    model = initialize_model(input_shape=(256, 256, 3))

    # compile the model
    model = compile_model(model, learning_rate=0.0005)

    # train the model
    X_train, y_train = train_dataset.unbatch()
    X_val, y_val = val_dataset.unbatch()
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_val = np.array(X_val)
    y_val = np.array(y_val)
    model, history = train_model(
        model,
        X_train,
        y_train,
        batch_size=50,
        patience=5,
        validation_data=(X_val, y_val),
        epochs=40)






    print("model trained")
    plot_history(history)

    # if SAVE_TO_GCS:
    #     # save the best model to GCS
    #     save_model_to_GCS()
