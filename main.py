from tools import dataset
from tools.dataset import Dataset
from tools import prepare
from tools import augmentation as augment

import os
import argparse

from glob import glob
import collections
import re
import random
SEED = None # set to None to use the current system time
random.seed(a=SEED)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score

import tensorflow
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, TimeDistributed, Dense, Dropout,\
    Flatten, Activation, Conv2D, MaxPooling2D, GlobalAveragePooling1D,\
    BatchNormalization, Masking, multiply, GlobalMaxPooling1D, Reshape,\
    GRU, average, Lambda, Average, Maximum, Concatenate

from tools.flow import farneback
from tensorflow.keras.backend import clear_session
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN, ModelCheckpoint, ReduceLROnPlateau

# LABELS_REGEX = dataset.LABELS_REGEX #7 labels
LABELS_REGEX = dataset.PAPER_LABELS_REGEX #5 labels
CLASSES_N = len(LABELS_REGEX)

KERAS_EPSILON = tensorflow.keras.backend.epsilon()

keras.backend.set_image_data_format('channels_last')


def build_model(model_dir, optimizer="adam"):
    spatial_input = Input(shape=(None, 16, 16, 1))
    temporal_input = Input(shape=(None, 16, 16, 2))

    #spatial stream
    spatial_conv1 = TimeDistributed(Conv2D(16, (3, 3), padding='same', activation='relu', name='spatial_conv1'))(spatial_input)
    spatial_bn_layer = TimeDistributed(BatchNormalization(name='spatial_bn_layer'))(spatial_conv1)
    spatial_maxpool1 = TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='spatial_maxpool1'))(spatial_bn_layer)
    spatial_conv2 = TimeDistributed(Conv2D(32, (3, 3), padding='same', activation='relu', name='spatial_conv2'))(spatial_maxpool1)
    spatial_maxpool2 = TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='spatial_maxpool2'))(spatial_conv2)
    spatial_conv3 = TimeDistributed(Conv2D(64, (3, 3), padding='same', activation='relu', name='spatial_conv3'))(spatial_maxpool2)
    spatial_maxpool3 = TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='spatial_maxpool3'))(spatial_conv3)
    spatial_conv4 = TimeDistributed(Conv2D(128, (3, 3), padding='same', activation='relu', name='spatial_conv4'))(spatial_maxpool3)
    spatial_maxpool4 = TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='spatial_maxpool4'))(spatial_conv4)
    spatial_flattened = TimeDistributed(Flatten(name='spatial_flattened'))(spatial_maxpool4)
    spatial_dense1 = TimeDistributed(Dense(512, name='spatial_dense1'))(spatial_flattened)
    spatial_dense2 = TimeDistributed(Dense(256, name='spatial_dense2'))(spatial_dense1)
    spatial_GRU = GRU(100, return_sequences=True, name='spatial_GRU')(spatial_dense2)
    spatial_GRU2 = GRU(100,  return_sequences=False, name='spatial_GRU2')(spatial_GRU)

    #handle numerical instability
    spatial_output = Lambda(lambda x: tensorflow.keras.backend.clip(x, KERAS_EPSILON, 1-KERAS_EPSILON))(spatial_GRU2)

    #temporal stream
    temporal_conv1 = TimeDistributed(Conv2D(16, (3, 3), padding='same', activation='relu', name='temporal_conv1'))(temporal_input)
    temporal_bn_layer = TimeDistributed(BatchNormalization(name='temporal_bn_layer'))(temporal_conv1)
    temporal_maxpool1 = TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='temporal_maxpool1'))(temporal_bn_layer)
    temporal_conv2 = TimeDistributed(Conv2D(32, (3, 3), padding='same', activation='relu', name='temporal_conv2'))(temporal_maxpool1)
    temporal_maxpool2 = TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='temporal_maxpool2'))(temporal_conv2)
    temporal_conv3 = TimeDistributed(Conv2D(64, (3, 3), padding='same', activation='relu', name='temporal_conv3'))(temporal_maxpool2)
    temporal_maxpool3 = TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='temporal_maxpool3'))(temporal_conv3)
    temporal_conv4 = TimeDistributed(Conv2D(128, (3, 3), padding='same', activation='relu', name='temporal_conv4'))(temporal_maxpool3)
    temporal_maxpool4 = TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='temporal_maxpool4'))(temporal_conv4)
    temporal_flattened = TimeDistributed(Flatten(name='temporal_flattened'))(temporal_maxpool4)
    temporal_dense1 = TimeDistributed(Dense(512, name='temporal_dense1'))(temporal_flattened)
    temporal_dense2 = TimeDistributed(Dense(256, name='temporal_dense2'))(temporal_dense1)
    temporal_GRU = GRU(100, return_sequences=True, name='temporal_GRU')(temporal_dense2)
    temporal_GRU2 = GRU(100, return_sequences=False, name='temporal_GRU2')(temporal_GRU)

    #handle numerical instability
    temporal_output = Lambda(lambda x: tensorflow.keras.backend.clip(x, KERAS_EPSILON, 1-KERAS_EPSILON))(temporal_GRU2)

    #merging
    concat = Concatenate(name='concat')([spatial_output, temporal_output])
    output = Dense(CLASSES_N, activation="softmax", name='output')(concat)
    #compiling
    model=Model([spatial_input, temporal_input], output)
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
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(os.path.join(model_dir, "model_accuracy.png"))
    plt.close()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(os.path.join(model_dir, "model_loss.png"))
    return

def to_categorical(y):
    return tensorflow.keras.utils.to_categorical(y, CLASSES_N)


class DataGenerator(keras.utils.Sequence):
    '''
    FIR data batch generator for Keras

    Parameters
    ----------
    data : list
        list of [fn, y] where fn is file location and y is a label

    Returns
    ----------
    [[temperature, flow], y] : list
        temperature : numpy array 
        flow : numpy array
        y : numpy array (one-hot encoded)

    '''
    def __init__(self, data, batch_size, shuffle: bool = True, augmentation: bool = False):
        self.data = data
        if (batch_size == -1):
            self.batch_size = len(data)
        else:
            self.batch_size = batch_size
        self.shuffle = shuffle
        if self.shuffle:
            random.shuffle(self.data)
        self.augmentation = augmentation

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
            if self.augmentation:
                k_rot = np.random.randint(0, 4)
                k_flip = np.random.randint(0, 3)
            [temperature_fn, flow_fn], y = self.data[idx]
            temperature = np.load(temperature_fn).astype(np.float32)
            if self.augmentation:
                temperature = augment.random_rotation(temperature, case=k_rot)
                temperature = augment.random_flip(temperature, case=k_flip)
            temperature = temperature[..., np.newaxis]
            flow = np.load(flow_fn)
            if self.augmentation:
                flow = farneback(np.squeeze(temperature))
                #flow = augment.random_rotation(flow, case=k_rot)
                #flow = augment.random_flip(flow, case=k_flip)
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
        return np.vstack([trailing, sequence])


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
                        default=1e-1,
                        help='How large a learning rate to use when training.')
    parser.add_argument(
        '--validation_size',
        type=float,
        default=0.1,
        help='Between 0.0 and 1.0, the proportion of the dataset \
            to include in the validation split.')
    parser.add_argument('--training_batch_size',
                        type=int,
                        default=128,
                        help='How many images to train on at a time.')
    parser.add_argument('--validation_batch_size',
                        type=int,
                        default=-1,
                        help='How many images to validate on at a time. -1 for batch_size = samples_n (more stable results).')
    parser.add_argument('--testing_batch_size',
                        type=int,
                        default=-1,
                        help='How many images to test on at a time. -1 for batch_size = samples_n (more stable results).')
    parser.add_argument("--download",
                        action="store_true",
                        help='Download the dataset.')
    parser.add_argument("--prepare",
                        action="store_true",
                        help='Prepare the dataset.')
    parser.add_argument("--testing_actor",
                        type=str,
                        default=None,
                        help='Choose testing actor, pattern: "human{}" [0-9]. Otherwise full cross validation is performed.')
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

    cnfs_mtx_dict = dict()
        
    # LOOCV
    for actor in dataset.ACTORS:
        if FLAGS.testing_actor:
            if actor != FLAGS.testing_actor:
                print("Skip")
                continue

        testing_actor = actor
        training_actors = list(dataset.ACTORS)
        training_actors.remove(testing_actor)

        model_fn_json = os.path.join(FLAGS.model_dir, "model_{}.json".format(actor))
        model_fn_hdf5 = os.path.join(FLAGS.model_dir, "model_{}.hdf5".format(actor))

        train_val_fns_y = []
        testing_fns_y = []
        for sample in data_fn_y:
            fn, y = sample
            sample_actor, sample_basename = fn.split(os.path.sep)
            if sample_actor == testing_actor:
                testing_fns_y.append([fn, y])
            else:
                train_val_fns_y.append([fn, y])

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
            print("{} samples in class {}".format(len(tmp_class), LABELS_REGEX[key]))
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
                                       FLAGS.training_batch_size,
                                       shuffle=True, augmentation=True)
        validation_batches = DataGenerator(validation_data,
                                         FLAGS.validation_batch_size,
                                         shuffle=True)
        testing_batches = DataGenerator(testing_data,
                                      FLAGS.testing_batch_size,
                                      shuffle=False)

        print("[INFO] \n")
        print("Actor: {}".format(actor))
        print("Training: {} samples -> {} batches".format(
            len(training_data), len(training_batches)))
        print("Validation: {} samples -> {} batches".format(
            len(validation_data), len(validation_batches)))
        print("Testing: {} samples -> {} batches".format(
            len(testing_data), len(testing_batches)))
            
        optimizer = optimizers.SGD(lr=FLAGS.learning_rate, clipnorm=0.5, momentum=0.5, nesterov=True) # best
        model = build_model(FLAGS.model_dir, optimizer)
        json_string = model.to_json()
        open(model_fn_json, 'w').write(json_string)
        model.summary()

        early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=1)
        # that shouldn't happen
        terminateNaN = TerminateOnNaN()
        saveBest = ModelCheckpoint(model_fn_hdf5, save_best_only=True)
        #reduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
        history = model.fit_generator(training_batches, epochs=FLAGS.epochs, validation_data=validation_batches, callbacks=[early_stopping, terminateNaN, saveBest])
        plot_history(history, FLAGS.model_dir)

        clear_session()
        # load json and create model
        json_file = open(model_fn_json, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = tensorflow.keras.models.model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(model_fn_hdf5)
        print("Loaded model from disk")

        predictions = loaded_model.predict_generator(testing_batches)
        y_pred = np.argmax(predictions, axis=-1)
        y_test = np.argmax(testing_batches[0][1], axis=-1)
        cnfs_mtx = confusion_matrix(y_test, y_pred)
        print(accuracy_score(y_test, y_pred))
        C = cnfs_mtx / cnfs_mtx.astype(np.float).sum(axis=1)

        cnfs_mtx_dict[actor] = cnfs_mtx

        print("[INFO] Model successfully trained, tested on {} ".format(actor))
        clear_session()


    cross_validation_cnfs_mtx = sum(cnfs_mtx_dict[item] for item in cnfs_mtx_dict)
    cross_validation_accuracy = cross_validation_cnfs_mtx.diagonal().sum()/cross_validation_cnfs_mtx.sum()

    metrics = dict()
    metrics["confusion_matrix"] = cross_validation_cnfs_mtx
    metrics["accuracy"] = cross_validation_accuracy
    np.save(os.path.join(FLAGS.model_dir, "metrics_dict.npy"), metrics)
    # metrics = np.load(os.path.join(FLAGS.model_dir, "metrics_dict.npy"), allow_pickle=True)[()]