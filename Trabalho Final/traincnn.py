import numpy
import pickle
import tflearn
import tensorflow
from sklearn.model_selection import StratifiedKFold

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

X = pickle.load(open("Xconv.pkl", "rb"))
Y_ = pickle.load(open("Yconv.pkl", "rb"))

Y = numpy.eye(2)[Y_.astype(numpy.uint8).reshape(-1)]

cross_val = StratifiedKFold(n_splits=10)
cross_val.get_n_splits(X, Y_)

for i, (train_index, test_index) in enumerate(cross_val.split(X, Y_)):

    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]

    X_train = X_train.reshape([-1, 28, 28, 1])
    X_test = X_test.reshape([-1, 28, 28, 1])

    tensorflow.reset_default_graph()
    tflearn.init_graph(num_cores=4)

    # Building 'VGG Network'
    network = input_data(shape=[None, 28, 28, 1])

    network = conv_2d(network, 14, 3, activation='relu')
    network = conv_2d(network, 14, 3, activation='relu')
    network = max_pool_2d(network, 2, strides=2)

    network = conv_2d(network, 64, 3, activation='relu')
    network = conv_2d(network, 64, 3, activation='relu')
    network = max_pool_2d(network, 2, strides=2)

    network = conv_2d(network, 128, 3, activation='relu')
    network = conv_2d(network, 128, 3, activation='relu')
    network = conv_2d(network, 128, 3, activation='relu')
    network = max_pool_2d(network, 2, strides=2)

    network = conv_2d(network, 256, 3, activation='relu')
    network = conv_2d(network, 256, 3, activation='relu')
    network = conv_2d(network, 256, 3, activation='relu')
    network = max_pool_2d(network, 2, strides=2)

    network = fully_connected(network, 1024, activation='relu')
    network = dropout(network, 0.5)
    network = fully_connected(network, 1024, activation='relu')
    network = dropout(network, 0.5)
    network = fully_connected(network, 2, activation='softmax')

    network = regression(network, optimizer='rmsprop', loss='categorical_crossentropy', learning_rate=0.0001)

    # Training
    model = tflearn.DNN(network, tensorboard_verbose=0)
    # model.fit({'input': X_train}, {'target': Y_train}, n_epoch=20, validation_set=({'input': X_test}, {'target': Y_test}), snapshot_step=100, show_metric=True)
    model.fit(X_train, Y_train, n_epoch=2, snapshot_step=100, show_metric=True)
    model.save("models/galaxies-cnn-%02d.model" % (i))

    Y_pred = model.predict(X_test)
    Y_pred_classes = numpy.argmax(Y_pred, axis=1)
    Y_classes = numpy.argmax(Y_test, axis=1)
    print(numpy.sum(Y_pred_classes==Y_classes)/len(X_test))
