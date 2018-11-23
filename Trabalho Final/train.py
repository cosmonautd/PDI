import numpy
import tflearn
import tensorflow
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

X = numpy.genfromtxt("X2.csv", delimiter=",")
Y_ = numpy.genfromtxt("Y2.csv", delimiter=",")

Y = numpy.eye(2)[Y_.astype(numpy.uint8).reshape(-1)]

cross_val = StratifiedKFold(n_splits=10)
cross_val.get_n_splits(X, Y_)

for i, (train_index, test_index) in enumerate(cross_val.split(X, Y_)):

    # X_train, X_test = X[train_index], X[test_index]
    # Y_train, Y_test = Y[train_index], Y[test_index]

    # tensorflow.reset_default_graph()
    # tflearn.init_graph(num_cores=4)

    # net = tflearn.input_data(shape=[None, 40])
    # net = tflearn.fully_connected(net, 100)
    # net = tflearn.fully_connected(net, 100)
    # net = tflearn.dropout(net, 0.5)
    # net = tflearn.fully_connected(net, 2, activation="softmax")
    # net = tflearn.regression(net, optimizer="adam", loss="categorical_crossentropy")

    # model = tflearn.DNN(net)
    # model.fit(X_train, Y_train, n_epoch=20)
    # model.save("models/galaxies-%02d.model" % (i))

    # Y_pred = model.predict(X_test)
    # Y_pred_classes = numpy.argmax(Y_pred, axis=1)
    # Y_classes = numpy.argmax(Y_test, axis=1)
    # print(numpy.sum(Y_pred_classes==Y_classes)/len(X_test))


    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y_[train_index], Y_[test_index]

    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, Y_train)

    Y_pred = lda.predict(X_test)
    print(numpy.sum(Y_pred==Y_test)/len(X_test))