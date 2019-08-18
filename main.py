from tools import dataset
from tools.dataset import Dataset
from tools import prepare

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
    Flatten, Activation, Conv2D, MaxPooling2D, LSTM, GlobalAveragePooling1D,\
    BatchNormalization, Masking, multiply, GlobalMaxPooling1D, Reshape,\
    GRU, average, Lambda

from tensorflow.keras.backend import clear_session
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN, ModelCheckpoint

# LABELS_REGEX = dataset.LABELS_REGEX
LABELS_REGEX = dataset.PAPER_LABELS_REGEX
CLASSES_N = len(LABELS_REGEX)

KERAS_EPSILON = tensorflow.keras.backend.epsilon()

keras.backend.set_image_data_format('channels_last')


def build_model(model_dir, optimizer="adam"):
    spatial_input = Input(shape=(None, 16, 16, 1))
    temporal_input = Input(shape=(None, 16, 16, 2))

    #spatial stream
    spatial_conv1 = TimeDistributed(Conv2D(16, (3, 3), padding='same', activation='relu'))(spatial_input)
    spatial_bn_layer = TimeDistributed(BatchNormalization())(spatial_conv1)
    spatial_maxpool1 = TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))(spatial_bn_layer)
    spatial_conv2 = TimeDistributed(Conv2D(32, (3, 3), padding='same', activation='relu'))(spatial_maxpool1)
    spatial_maxpool2 = TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))(spatial_conv2)
    spatial_conv3 = TimeDistributed(Conv2D(64, (3, 3), padding='same', activation='relu'))(spatial_maxpool2)
    spatial_maxpool3 = TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))(spatial_conv3)
    spatial_conv4 = TimeDistributed(Conv2D(128, (3, 3), padding='same', activation='relu'))(spatial_maxpool3)
    spatial_maxpool4 = TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))(spatial_conv4)
    spatial_flattened = TimeDistributed(Flatten())(spatial_maxpool4)
    spatial_dense1 = TimeDistributed(Dense(512))(spatial_flattened)
    spatial_dense2 = TimeDistributed(Dense(256))(spatial_dense1)
    spatial_LSTM = LSTM(CLASSES_N, return_sequences=False, activation='softmax')(spatial_dense2)
    #spatial_global_pool = GlobalAveragePooling1D()(spatial_LSTM)
    spatial_global_pool = spatial_LSTM
    #handle numerical instability
    spatial_output = Lambda(lambda x: tensorflow.keras.backend.clip(x, KERAS_EPSILON, 1-KERAS_EPSILON))(spatial_global_pool)

    #temporal stream
    temporal_conv1 = TimeDistributed(Conv2D(16, (3, 3), padding='same', activation='relu'))(temporal_input)
    temporal_bn_layer = TimeDistributed(BatchNormalization())(temporal_conv1)
    temporal_maxpool1 = TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))(temporal_bn_layer)
    temporal_conv2 = TimeDistributed(Conv2D(32, (3, 3), padding='same', activation='relu'))(temporal_maxpool1)
    temporal_maxpool2 = TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))(temporal_conv2)
    temporal_conv3 = TimeDistributed(Conv2D(64, (3, 3), padding='same', activation='relu'))(temporal_maxpool2)
    temporal_maxpool3 = TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))(temporal_conv3)
    temporal_conv4 = TimeDistributed(Conv2D(128, (3, 3), padding='same', activation='relu'))(temporal_maxpool3)
    temporal_maxpool4 = TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))(temporal_conv4)
    temporal_flattened = TimeDistributed(Flatten())(temporal_maxpool4)
    temporal_dense1 = TimeDistributed(Dense(512))(temporal_flattened)
    temporal_dense2 = TimeDistributed(Dense(256))(temporal_dense1)
    temporal_LSTM = GRU(CLASSES_N, return_sequences=True, activation='softmax')(temporal_dense2)
    temporal_LSTM2 = GRU(CLASSES_N, return_sequences=False, activation='softmax')(temporal_LSTM)
    #temporal_global_pool = GlobalAveragePooling1D()(temporal_LSTM2)
    #handle numerical instability
    temporal_output = Lambda(lambda x: tensorflow.keras.backend.clip(x, KERAS_EPSILON, 1-KERAS_EPSILON))(temporal_LSTM2)

    #merging
    output = temporal_output
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
    data: list
        list of [fn, y] where fn is file location and y is a label

    '''
    def __init__(self, data, batch_size, shuffle: bool = True):
        self.data = data
        if (batch_size == -1):
            self.batch_size = len(data)
        else:
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
            temperature = np.load(temperature_fn).astype(np.float32)
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
                        default=1e-5,
                        help='How large a learning rate to use when training.')
    parser.add_argument(
        '--validation_size',
        type=float,
        default=0.1,
        help='Between 0.0 and 1.0, the proportion of the dataset \
            to include in the validation split.')
    # ! ADD: {}_batch_size for [train, validation, test]
    # ! ADD: {} -1 for batch_size = sample_num
    parser.add_argument('--training_batch_size',
                        type=int,
                        default=100,
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

    model_fn_json = os.path.join(FLAGS.model_dir, "model.json")
    model_fn_hdf5 = os.path.join(FLAGS.model_dir, "model.hdf5")

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
        if actor != 'human2':
            print("Skip")
            continue
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
                                       shuffle=True)
        validation_batches = DataGenerator(validation_data,
                                         FLAGS.validation_batch_size,
                                         shuffle=True)
        testing_batches = DataGenerator(testing_data,
                                      FLAGS.testing_batch_size,
                                      shuffle=False)

        print("[INFO] \n")
        print("Training: {} samples -> {} batches".format(
            len(training_data), len(training_batches)))
        print("Validation: {} samples -> {} batches".format(
            len(validation_data), len(validation_batches)))
        print("Testing: {} samples -> {} batches".format(
            len(testing_data), len(testing_batches)))
        
        #optimizer = optimizers.SGD(lr=FLAGS.learning_rate, clipnorm=0.5, momentum=0.2)
        optimizer = optimizers.SGD(lr=FLAGS.learning_rate, clipnorm=0.5, momentum=0.5, nesterov=True) # best
        model = build_model(FLAGS.model_dir, optimizer)
        json_string = model.to_json()
        open(model_fn_json, 'w').write(json_string)
        model.summary()

        # ! later change to val_los!! amd add to model.fig_generator
        early_stopping = EarlyStopping(monitor='loss', patience=20, verbose=1)
        #that shouldn't happen anymore! (NaN)
        terminateNaN = TerminateOnNaN()
        saveBest = ModelCheckpoint(model_fn_hdf5, save_best_only=True)
        history = model.fit_generator(training_batches, epochs=FLAGS.epochs, validation_data=validation_batches, callbacks=[early_stopping, terminateNaN, saveBest])
        plot_history(history, FLAGS.model_dir)
        test = model.evaluate_generator(testing_batches)
        print('Test loss:', test[0])
        print('Test accuracy:', test[1])
        predictions = model.predict_generator(testing_batches)
        y_pred = np.argmax(predictions, axis=-1)
        y_test = np.argmax(testing_batches[0][1], axis=-1)
        cnfs_mtx = confusion_matrix(y_test, y_pred)
        clear_session()
        break

'''
# manual testing
# load json and create model
json_file = open(model_fn_json, 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = tensorflow.keras.models.model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(model_fn_hdf5)
print("Loaded model from disk")

predictions = model.predict_generator(testing_batches)
y_pred = np.argmax(predictions, axis=-1)
y_test = np.argmax(testing_batches[0][1], axis=-1)
cnfs_mtx = confusion_matrix(y_test, y_pred)
print(accuracy_score(y_test, y_pred))

walk = np.load(r"D:\tmps\cache\temperature\human1\walk_20170203_p5_dark1_126_142.npy")[..., np.newaxis]
sitdown = np.load(r"D:\tmps\cache\temperature\human1\sitdown_20170203_p6_dark2_128_148.npy")[..., np.newaxis]
standup = np.load(r"D:\tmps\cache\temperature\human1\standup_20170203_p8_light2_169_197.npy")[..., np.newaxis]
falling = np.load(r"D:\tmps\cache\temperature\human1\falling1_20170203_p14_light1_104_125.npy")[..., np.newaxis]
sit = np.load(r"D:\tmps\cache\temperature\human1\sit_20170203_p1_dark3_197_208.npy")[..., np.newaxis]
lie = np.load(r"D:\tmps\cache\temperature\human1\lie_20170203_p3_light4_175_199.npy")[..., np.newaxis]
stand = np.load(r"D:\tmps\cache\temperature\human1\stand_20170203_p3_light3_186_209.npy")[..., np.newaxis]
predictions = []
for action in [walk, sitdown, standup, falling, sit, lie, stand]:
    predictions.append(loaded_model.predict([action[np.newaxis], np.random.rand(np.prod([*action.shape[:-1], 2])).reshape([*action.shape[:-1], 2])[np.newaxis]]))

for action in predictions:
    print(action.argmax())


def __pad_to_length(sequence, length):
    if sequence.shape[0] == length:
        return sequence
    trailing = np.zeros([length - sequence.shape[0], *sequence.shape[1:]],
                        sequence.dtype)
    return np.vstack([sequence, trailing])

a = __pad_to_length(falling, 30)
'''