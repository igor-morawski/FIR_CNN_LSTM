from tools import dataset
from tools.dataset import Dataset
from tools import prepare

import os
import argparse

from glob import glob
import collections
import re
import random
SEED = 5 # set to None to use the current system time
random.seed(a=SEED)

import numpy as np
import matplotlib.pyplot as plt

import tensorflow
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, TimeDistributed, Dense, Dropout, Flatten, \
    Activation, Conv2D, MaxPooling2D, LSTM, GlobalAveragePooling1D, average
from tensorflow.keras.backend import clear_session
from tensorflow.keras.optimizers import SGD

# LABELS_REGEX = dataset.LABELS_REGEX
LABELS_REGEX = dataset.PAPER_LABELS_REGEX
CLASSES_N = len(LABELS_REGEX)

keras.backend.set_image_data_format('channels_last')


def build_model(model_dir, optimizer="adam"):
    spatial_input = Input(shape=(None, 16, 16, 1))
    spatial_conv1 = TimeDistributed(Conv2D(32, (3, 3), padding='same', activation='relu'))(spatial_input)
    spatial_pooled1 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(spatial_conv1)
    spatial_conv2 = TimeDistributed(Conv2D(32, (3, 3), activation='relu'))(spatial_pooled1)
    spatial_dropout1 = TimeDistributed(Dropout(0.25))(spatial_conv2)
    spatial_flatten1 = TimeDistributed(Flatten())(spatial_dropout1)
    spatial_dense1 = TimeDistributed(Dense(512))(spatial_flatten1)
    spatial_dense2 = TimeDistributed(Dense(35))(spatial_dense1)
    spatial_LSTM1 = LSTM(CLASSES_N, return_sequences=True)(spatial_dense2)
    spatial_glob_avg = GlobalAveragePooling1D()(spatial_LSTM1)
    spatial_stream = spatial_glob_avg
    # spatial_stream = Model(spatial_input, spatial_glob_avg)

    temporal_input = Input(shape=(None, 16, 16, 2))
    temporal_conv1 = TimeDistributed(Conv2D(32, (3, 3), padding='same', activation='relu'))(temporal_input)
    temporal_pooled1 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(temporal_conv1)
    temporal_conv2 = TimeDistributed(Conv2D(32, (3, 3), activation='relu'))(temporal_pooled1)
    temporal_dropout1 = TimeDistributed(Dropout(0.25))(temporal_conv2)
    temporal_flatten1 = TimeDistributed(Flatten())(temporal_dropout1)
    temporal_dense1 = TimeDistributed(Dense(512))(temporal_flatten1)
    temporal_dense2 = TimeDistributed(Dense(35))(temporal_dense1)
    temporal_LSTM1 = LSTM(CLASSES_N, return_sequences=True)(temporal_dense2)
    temporal_glob_avg = GlobalAveragePooling1D()(temporal_LSTM1)
    temporal_stream = temporal_glob_avg
    # temporal_stream = Model(temporal_input, temporal_glob_avg)

    merged = average([spatial_stream, temporal_stream])
    model = Model([spatial_input, temporal_input], merged)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    prepare.ensure_dir_exists(model_dir)
    keras.utils.plot_model(model, os.path.join(model_dir, 'model.png'))
    return model

def plot_history(history, model_dir):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(os.path.join(model_dir, "model_accuracy.png"))

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(os.path.join(model_dir, "model_loss.png"))
    return

def to_categorical(y):
    return tensorflow.keras.utils.to_categorical(y, CLASSES_N)


class DataGenerator(keras.utils.Sequence):
    '''
    FIR data batch generator for Keras

    Parameters
    ----------
    data: list
        list of [fn, y] where fn is file location and y is a label

    '''
    def __init__(self, data, batch_size, shuffle: bool = True):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        if self.shuffle:
            random.shuffle(self.data)

    def __len__(self):
        return int(np.floor(len(self.data) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.data)

    def __getitem__(self, index):
        indices = list(
            range(index * self.batch_size, (index + 1) * self.batch_size))
        return self.__load_data(indices)

    def __load_data(self, indices):
        samples = []
        temperature_length_max = 0
        flow_length_max = 0
        for idx in indices:
            [temperature_fn, flow_fn], y = self.data[idx]
            temperature = np.load(temperature_fn)
            temperature = temperature[..., np.newaxis]
            flow = np.load(flow_fn)
            if temperature.shape[0] > temperature_length_max:
                temperature_length_max = temperature.shape[0]
            if flow.shape[0] > flow_length_max:
                flow_length_max = flow.shape[0]
            samples.append([[temperature, flow], y])

        # zero-pad
        TEMPERATURE, FLOW = [], []
        Y = []
        for sample in samples:
            [temperature, flow], y = sample
            temperature = self.__pad_to_length(temperature,
                                               temperature_length_max)
            flow = self.__pad_to_length(flow, flow_length_max)
            TEMPERATURE.append(temperature)
            FLOW.append(flow)
            Y.append(y)
        TEMPERATURE, FLOW, Y = np.array(TEMPERATURE), np.array(FLOW), np.array(
            Y)
        return ([TEMPERATURE, FLOW], Y)

    def __pad_to_length(self, sequence, length):
        if sequence.shape[0] == length:
            return sequence
        trailing = np.zeros([length - sequence.shape[0], *sequence.shape[1:]],
                            sequence.dtype)
        return np.vstack([sequence, trailing])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir',
                        type=str,
                        default=os.path.join("..", "dataset"),
                        help='Path to folder containing the FIR dataset.')
    parser.add_argument('--model_dir',
                        type=str,
                        default="/" + os.path.join("tmps", "model"),
                        help='Where to save the trained model.')
    parser.add_argument(
        '--temperature_dir',
        type=str,
        default="/" + os.path.join("tmps", "cache", "temperature"),
        help='Where to save the cached sequences (temperature).')
    parser.add_argument(
        '--flow_dir',
        type=str,
        default="/" + os.path.join("tmps", "cache", "optical_flow"),
        help='Where to save the cached sequences (optical flow).')
    parser.add_argument('--epochs',
                        type=int,
                        default=50,
                        help='How many epochs to run before ending.')
    parser.add_argument('--learning_rate',
                        type=float,
                        default=0.01,
                        help='How large a learning rate to use when training.')
    parser.add_argument(
        '--validation_size',
        type=float,
        default=0.1,
        help='Between 0.0 and 1.0, the proportion of the dataset \
            to include in the validation split.')
    # ! ADD: {}_batch_size for [train, validation, test]
    # ! ADD: {} -1 for batch_size = sample_num
    parser.add_argument('--batch_size',
                        type=int,
                        default=100,
                        help='How many images to train on at a time.')
    parser.add_argument("--download",
                        action="store_true",
                        help='Download the dataset.')
    parser.add_argument("--prepare",
                        action="store_true",
                        help='Prepare the dataset.')
    FLAGS, unparsed = parser.parse_known_args()

    if FLAGS.download:
        dataset.download("..")

    data_normalized = Dataset(FLAGS.dataset_dir, minmax_normalized=True)

    if FLAGS.prepare:
        prepare.sequences_by_actor(data_normalized, FLAGS.temperature_dir)
        prepare.optical_flow(data_normalized, FLAGS.flow_dir)

    temperature_files = glob(os.path.join(FLAGS.temperature_dir, "**",
                                          "*.npy"))
    flow_files = glob(os.path.join(FLAGS.flow_dir, "**", "*.npy"))

    def files_same(a, b):
        return collections.Counter([os.path.split(item)[1]
                                    for item in a]) == collections.Counter(
                                        [os.path.split(item)[1] for item in b])

    if not files_same(temperature_files, flow_files):
        raise ValueError(
            "The number and naming of the samples in temporal and spatial \
                streams should be the same.")

    if (FLAGS.validation_size > 1) or (FLAGS.validation_size < 0):
        raise ValueError("Validation size should be between 0.0 and 1.0")

    # relative_path, y = data_fn_y[i]
    data_fn_y = []
    for path in temperature_files:
        sample_actor, sample_basename = path.split(os.path.sep)[-2:]
        relative_path = os.path.join(sample_actor, sample_basename)
        y = None
        for pattern in LABELS_REGEX:
            if re.search(pattern + "_", sample_basename):
                y = LABELS_REGEX[pattern]
        data_fn_y.append([relative_path, y])

    # LOOCV
    for actor in dataset.ACTORS:
        testing_actor = actor
        training_actors = list(dataset.ACTORS)
        training_actors.remove(testing_actor)

        train_val_fns_y = []
        testing_fns_y = []
        for sample in data_fn_y:
            fn, y = sample
            sample_actor, sample_basename = fn.split(os.path.sep)
            if sample_actor == testing_actor:
                testing_fns_y.append([fn, y])
            else:
                train_val_fns_y.append([fn, y])

        '''
        # regular split
        split = int(len(training_fns_y) * FLAGS.validation_size)
        validation_fns_y, training_fns_y = train_val_fns_y[:split], train_val_fns_y[split:]
        '''

        # balanced split
        validation_fns_y, training_fns_y = [], []
        train_val_fns_y_classes = []
        for key in LABELS_REGEX.keys():
            tmp_class = []
            random.shuffle(train_val_fns_y)
            for sample in train_val_fns_y:
                fn, y = sample
                if (y == LABELS_REGEX[key]):
                    tmp_class.append(sample)
            split = int(len(tmp_class) * FLAGS.validation_size)
            validation_fns_y.extend(tmp_class[:split])
            training_fns_y.extend(tmp_class[split:])
        

        # add back the prefix
        # [temperature_fn, flow_fn], y = *_data
        def add_prefixes(list_fns_y, temperature_prefix, flow_prefix):
            list_data = []
            for sample in list_fns_y:
                fn, y = sample
                list_data.append([[
                    os.path.join(temperature_prefix, fn),
                    os.path.join(flow_prefix, fn)
                ],
                                  to_categorical(y)])
            return list_data

        # [temperature_fn, flow_fn], y = *_data
        testing_data = add_prefixes(testing_fns_y, FLAGS.temperature_dir,
                                    FLAGS.flow_dir)
        training_data = add_prefixes(training_fns_y, FLAGS.temperature_dir,
                                     FLAGS.flow_dir)
        validation_data = add_prefixes(validation_fns_y, FLAGS.temperature_dir,
                                       FLAGS.flow_dir)

        training_batches = DataGenerator(training_data,
                                       FLAGS.batch_size,
                                       shuffle=True)
        validation_batches = DataGenerator(validation_data,
                                         FLAGS.batch_size,
                                         shuffle=True)
        testing_batches = DataGenerator(testing_data,
                                      FLAGS.batch_size,
                                      shuffle=True)

        print("[INFO] \n")
        print("Training: {} samples -> {} batches".format(
            len(training_data), len(training_batches)))
        print("Validation: {} samples -> {} batches".format(
            len(validation_data), len(validation_batches)))
        print("Testing: {} samples -> {} batches".format(
            len(testing_data), len(testing_batches)))
        
        optimizer = SGD(lr=1e-5)
        model = build_model(FLAGS.model_dir, optimizer)
        model.summary()

        history = model.fit_generator(training_batches, epochs=FLAGS.epochs, validation_data=validation_batches)
        plot_history(history, FLAGS.model_dir)
        
        clear_session()
        break
