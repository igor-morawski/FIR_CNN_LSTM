import glob
import pandas as pd
import numpy as np
import argparse
import os
import math
import random
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import train_test_split
import hashlib
from keras.utils import to_categorical
import tensorflow as tf

from keras.models import Sequential
from keras.layers.wrappers import TimeDistributed
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.recurrent import LSTM
from keras.layers.pooling import GlobalAveragePooling1D, GlobalMaxPooling1D
# from keras.layers.merge import Average
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.utils import plot_model, np_utils
from keras import optimizers
from keras.layers import BatchNormalization

import matplotlib.pyplot as plt

frames = 30
batch_size = 100
epochs = 10
data_type = np.float32

mean_ = 24.20601406
scale_ = 1.49176938
var_ = 0.28328413

labels_regex = dict([
    (r'walk.*', 0),
    (r'sitdown', 1),
    (r'standup', 2),
    (r'falling.*', 3),
    (r'^(sit|lie|stand)$', 4),
])

labels = list(set(labels_regex.values()))
labels_num = len(labels)

# dataset


def load_annotation(dir_name, drop_actor=None):
    # check if annotation already processed
    options = str((frames, tuple(labels_regex), drop_actor))
    options = hashlib.md5(options.encode()).hexdigest()
    fn = os.path.join('.', 'cached', str(options)+".pkl")
    if os.path.isfile(fn):
        print("Reading cached annotation file...")
        return pd.read_pickle(fn)

    pattern = os.path.join(dir_name, 'annotation', '*_human*.csv')
    annotation_generator = glob.iglob(pattern)
    annotation = pd.concat([pd.read_csv(fn, header=None)
                            for fn in annotation_generator], ignore_index=True)

    if drop_actor:
        annotation = annotation[annotation[4] != drop_actor]

    for label in labels_regex:
        annotation = annotation.replace(regex=label, value=labels_regex[label])
    # d.loc[d[3] == "new"]

    def process_annotation(annotation, window_size):
        annotation_processed = pd.DataFrame(columns=annotation.keys())
        # tmp = pd.DataFrame([[0, 0,  0, 0, 0]], columns=annotation.keys())

        def ranges(start, end, window_size):
            ranges = []
            iterator = 0
            while (start+window_size-1+iterator <= end):
                ranges.append([start+iterator, start+window_size-1+iterator])
                iterator += 1
            return ranges

        for row in annotation.itertuples():
            start = row[2]
            end = row[3]
            range_list = ranges(start, end, window_size)
            tmp = []
            for window in range_list:
                tmp.append(pd.DataFrame(
                    [[row[1], window[0], window[1], row[4], row[5]]], columns=annotation.keys()))
            if range_list:
                tmp = pd.concat(tmp, ignore_index=True)
                annotation_processed = pd.concat(
                    [annotation_processed, tmp], ignore_index=True)

        return annotation_processed

    print("Processing annotation file...")
    annotation = process_annotation(annotation, window_size=frames)
    print("Saving annotation file...")
    annotation.to_pickle(fn)

    return annotation


def split_data(annotation, train_ratio, random=42, balanced=False, samples_min=None, shuffle=True):
    X_train, X_test, y_train, y_test = [], [], [], []
    if balanced:
        print("Balancing dataset...")
        lenghts = []
        for label in labels:
            lenghts.append(len(annotation[annotation[3] == label]))
        print("\nBefore balancing:")
        table = pd.DataFrame([["length "]+[length for length in lenghts]],
                             columns=["label "]+[label for label in labels])
        print(table.to_string(index=False)+"\n")
        if samples_min:
            None
        print("Balancing not yet implemented")
    for label in labels:
        data_label = annotation[annotation[3] == label]
        X_label = data_label[[0, 1, 2]]
        y_label = data_label[3]
        X_train_label, X_test_label, y_train_label, y_test_label = \
            train_test_split(X_label, y_label, test_size=1-train_ratio,
                             train_size=train_ratio, random_state=random)
        X_train.append(X_train_label)
        X_test.append(X_test_label)
        y_train.append(y_train_label)
        y_test.append(y_test_label)
    X_train = pd.concat(X_train, ignore_index=True)
    X_test = pd.concat(X_test, ignore_index=True)
    y_train = pd.concat(y_train, ignore_index=True)
    y_test = pd.concat(y_test, ignore_index=True)
    if shuffle:
        idx_train = np.random.permutation(X_train.index)
        idx_test = np.random.permutation(X_test.index)
        X_train, y_train = X_train.reindex(
            idx_train), y_train.reindex(idx_train)
        X_test, y_test = X_test.reindex(idx_test), y_test.reindex(idx_test)

    return X_train, X_test, y_train, y_test


def expand_last_batch(x_set, y_set, batch_size):
    add_number = batch_size - len(x_set) % batch_size
    for _ in range(add_number):
        idx = random.randint(0, len(x_set))
        x_set = pd.concat([x_set, x_set[idx:idx+1]], ignore_index=True)
        y_set = pd.concat([y_set, y_set[idx:idx+1]], ignore_index=True)
    return x_set, y_set


def read_FIR_sequence(pd_row, augmentation=True):
    fn = pd_row[1]
    start = pd_row[2]
    end = pd_row[3]
    pickle = pd.read_pickle(os.path.join(
        '.', 'cached', fn[:-4]+".pkl"))[start:end+1]
    sequence = pickle[pickle.columns[2:]]
    result = sequence.values.astype(data_type).reshape(end-start+1, 16, 16)
    if augmentation:
        # rotation
        if random.randint(0, 1) == 1:
            result = np.rot90(result, k=random.randint(1, 3), axes=(1, 2))
        # mirror horizontaly, use flipud and transposing,
        # using fliplr(result) would flip the time direction
        if random.randint(0, 1) == 1:
            result = np.flipud(result.T).T
        # mirror vertically
        if random.randint(0, 1) == 1:
            result = np.fliplr(result.T).T
    return result


class FIRSequence(Sequence):

    def __init__(self, x_set, y_set, batch_size, shuffle=True, augmentation=True):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.current = 0
        self.len = math.ceil(len(self.x) / self.batch_size)
        self.shuffle = shuffle
        self.augmentation = augmentation

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def steps(self):
        return int((len(self.x) - 1) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) *
                         self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) *
                         self.batch_size]
        x, y = np.array([read_FIR_sequence(row, augmentation=self.augmentation) for row in batch_x.itertuples(
        )]).reshape([batch_size, frames, 16, 16, 1]), to_categorical(batch_y, num_classes=labels_num)
        x -= mean_
        x /= scale_
        return x, y
        '''
         # x, y = np.array([read_FIR_sequence(row, augmentation=self.augmentation) for row in batch_x.itertuples(
        # )]).reshape([batch_size, frames, 16, 16, 1]), to_categorical(batch_y, num_classes=labels_num)
        return np.random.uniform(size=(100, 10, 16, 16, 1)), y
        '''

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.current = 0
        if self.shuffle:
            idx = np.random.permutation(self.x.index)
            self.x, self.y = self.x.reindex(idx), self.y.reindex(idx)

    def __iter__(self):
        self.current = 0
        return self

    def __next__(self):
        if (self.current >= math.ceil(len(self.x) / self.batch_size) - 1):
            self.on_epoch_end()
            return self[self.current]
        else:
            self.current += 1
            return self[self.current]


def build_model():
    model = Sequential()

    model.add(TimeDistributed(Conv2D(32, (3, 3), padding='same'),
                              input_shape=(frames, 16, 16, 1)))
    model.add(BatchNormalization())
    model.add(TimeDistributed(Activation('relu')))
    model.add(TimeDistributed(Conv2D(32, (3, 3))))
    model.add(TimeDistributed(Activation('relu')))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
    model.add(TimeDistributed(Dropout(0.25)))

    model.add(TimeDistributed(Flatten()))
    model.add(TimeDistributed(Dense(512)))

    model.add(TimeDistributed(Dense(35, name="first_dense_rgb")))

    model.add(LSTM(20, return_sequences=True, name="lstm_layer_rgb"))

    model.add(TimeDistributed(Dense(labels_num),
                              name="time_distr_dense_one_rgb"))
    model.add(GlobalAveragePooling1D(name="global_avg_rgb"))

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    plot_model(model, to_file='model/cnn_lstm.png')
    return model


def plot_history(history):
    # Plot the history of accuracy
    plt.plot(history.history['acc'], "o-", label="accuracy")
    plt.plot(history.history['val_acc'], "o-", label="val_acc")
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(loc="lower right")
    plt.savefig("model/model_accuracy.png")

    # Plot the history of loss
    plt.plot(history.history['loss'], "o-", label="loss",)
    plt.plot(history.history['val_loss'], "o-", label="val_loss")
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='lower right')
    plt.savefig("model/model_loss.png")

    return


if __name__ == "__main__":
    print("You need to implement balancing and augmentation... bitch")
    print("Initializing...")

    # Parse arguments
    parser = argparse.ArgumentParser(
        description="FIR action recognition by cnn and lstm.")
    parser.add_argument("--dataset", type=str, default='dataset')
    parser.add_argument("--human_cross", type=int, default=None)
    args = parser.parse_args()

    dataset = args.dataset
    human_cross = args.human_cross
    actor = "human" + str(human_cross)

    import tools.dataset
    tools.dataset.isCached(cache=True)

    import tools.dir_structure
    tools.dir_structure.dir_init()

    annotation = load_annotation(dataset)
    X_train, X_test, y_train, y_test = split_data(
        annotation, train_ratio=0.8, balanced=True)

    X_train, y_train = expand_last_batch(X_train, y_train, batch_size)
    X_test, y_test = expand_last_batch(X_test, y_test, batch_size)

    '''
    training_samples = FIRSequence(
        X_train, y_train, batch_size, augmentation=False)
    test_samples = FIRSequence(X_test, y_test, batch_size, augmentation=False)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    for generator in [test_samples, training_samples]:
        for idx in range(generator.steps()):
            x = generator[idx][0]
            data = np.stack([sample.flatten() for sample in x]).reshape(-1, 1)
            scaler.partial_fit(data)
    print(scaler.mean_)
    print(scaler.var_)
    '''

    training_samples = FIRSequence(X_train, y_train, batch_size)
    test_samples = FIRSequence(X_test, y_test, batch_size)
    model = build_model()
    # model.summary()
    #history = model.fit(x=training_samples[0][0], y=training_samples[0][1], batch_size=batch_size, epochs=epochs, verbose=1, callbacks=None, validation_split=0.0, validation_data=test_samples[0], shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None)
    history = model.fit_generator(generator=training_samples, steps_per_epoch=training_samples.steps(
    ), epochs=epochs, verbose=1, validation_data=test_samples[0],
        validation_steps=test_samples.steps())
    plot_history(history)
    # Save model and weights
    json_string = model.to_json()
    open('model/cnn_lstm.json', 'w').write(json_string)
    model.save_weights('model/cnn_lstm.hdf5')
    print("Saved model")

    '''
    history = model.fit_generator(generator=training_samples, steps_per_epoch=training_samples.steps(
    ), epochs=epochs, verbose=1, validation_data=test_samples,
        validation_steps=test_samples.steps())

    plot_history(history)
    print("Trained model")
    # Save model and weights
    json_string=model.to_json()
    open('model/cnn_lstm.json', 'w').write(json_string)
    model.save_weights('model/cnn_lstm.hdf5')
    print("Saved model")

    # Evaluate model
    score=model.evaluate_generator(training_samples, test_samples)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # Clear session
    from keras.backend import tensorflow_backend as backend
    backend.clear_session()
    '''
