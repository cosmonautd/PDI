import os
import time
import numpy
import sklearn
import numexpr

def sigmoid(x):
    # return 1 / (1 + numpy.exp(-x))
    return numexpr.evaluate('1 / (1 + exp(-x))')

def tanh(x):
    # return numpy.tanh(x)
    return numexpr.evaluate('tanh(x)')

def gaussian_weights(p, neurons):
    mu = 0
    sigma = 1
    numpy.random.seed(0)
    W = sigma*numpy.random.randn(neurons, p+1) + mu
    return W

def lcg_weights(p, q, seed, a, b, c):
    W = list()
    W.append(seed)
    for i in range(q*(p+1)-1):
        r = (a*W[i] + b) % c
        if r not in W:
            W.append(r)
        else:
            S = list(set(list(range(c))) - set(W))
            S.sort()
            W.append(S[int(len(S)/2)])
    W = numpy.array(W)
    W = (W - numpy.mean(W))/numpy.std(W, ddof=1)
    W = W.reshape((q, p+1))
    return W

class ContourDescriptor():
    def __init__(self, mode, neurons, params):
        assert type(mode) == str
        assert type(params) == tuple
        if mode == "neighborhood":
            assert all(isinstance(x, int) for x in params)
            assert all(x % 2 == 0 for x in params)
            self.f = self.neighborhood
            self.neurons = neurons
            self.params = params
            self.W = list()
            for param in self.params:
                self.W.append(lcg_weights(param, self.neurons, self.neurons, param, self.neurons*(param+1), (self.neurons*(param+1))**2))
        elif mode == "contour_portion":
            assert all(isinstance(x, int) for x in params)
            assert all(x in [5,10,15,20,25] for x in params)
            self.f = self.contour_portion
            self.neurons = neurons
            self.params = params
            self.W = list()
            add_ = {5:0, 10:1, 15:2, 20:3, 25:4}
            for param in self.params:
                add__ = add_[param]
                self.W.append(lcg_weights(2, self.neurons, self.neurons + add__, 2, self.neurons*(2+1) + add__, (self.neurons*(2+1))**2))
        elif mode == "angle":
            assert all(isinstance(x, int) for x in params)
            assert all(x % 2 == 0 for x in params)
            self.f = self.angle
            self.neurons = neurons
            self.params = params
            self.W = list()
            for param in self.params:
                # self.W.append(lcg_weights(param+1, self.neurons, self.neurons, param+1, self.neurons*(param+2), (self.neurons*(param+2))**2))
                self.W.append(gaussian_weights(param+1, self.neurons))
        else:
            raise ValueError("Mode %s is unavailable" % mode)

    def generate_contour(self, image):
        raise NotImplementedError("Contour generation not yet implemented")
    
    def extract_contour_features(self, image=None, contour=None):
        if image is None and contour is None:
            raise ValueError("At least one image or contour must be provided")
        elif image is not None and contour is None:
            contour = self.generate_contour(image)
        return self.f(contour)
    
    def neighborhood(self, contour):
        # cy, cx = numpy.mean(contour, axis=0)
        c = numpy.mean(contour, axis=0)
        M = list()
        for i, r in enumerate(self.params):

            X = numpy.zeros((r*len(contour),2))
            counter = 0
            r = int(r/2)
            for j in range(len(contour)):
                for k in range(j-r, j+r+1, 1):
                    if j != k:
                        X[counter][0] = contour[k%len(contour)][0]
                        X[counter][1] = contour[k%len(contour)][1]
                        counter += 1
            d = numpy.array(contour)
            D = numpy.linalg.norm(d-c, axis=1)/len(contour)
            
            X = numpy.linalg.norm(X-c, axis=1)/len(contour)
            X = X.reshape((len(contour),2*r))
            # X = sklearn.preprocessing.scale(X, axis=0)
            X = X - numpy.mean(X, axis=0)
            X = X / numpy.std(X, axis=0, ddof=1)
            X = X.T
            bias_X = -1*numpy.ones(X.shape[1]).reshape(1, X.shape[1])
            X = numpy.concatenate((bias_X, X))
            Z = sigmoid(numpy.dot(self.W[i], X))
            bias_Z = -1*numpy.ones(Z.shape[1]).reshape(1, Z.shape[1])
            Z = numpy.concatenate((bias_Z, Z))
            M.append(numpy.dot((numpy.dot(D,Z.T)),(numpy.linalg.inv(numpy.dot(Z,Z.T)))))
            
        return numpy.concatenate(tuple([numpy.array(m).flatten() for m in M]), axis=None)
    
    def contour_portion(self, contour):
        c = numpy.mean(contour, axis=0)
        M = list()
        for i, r in enumerate(self.params):

            X = numpy.zeros((2*len(contour),2))
            D = numpy.zeros(len(contour))
            counter_x = 0
            counter_y = 0
            portion = int((r/100)*len(contour))
            for j, (py, px) in enumerate(contour):
                yy, xx = contour[(j+portion) % len(contour)] # counterclockwise
                X[counter_x][0] = py
                X[counter_x][1] = px
                counter_x += 1
                X[counter_x][0] = yy
                X[counter_x][1] = xx
                counter_x += 1
                D[counter_y] = numpy.sqrt((px - xx)**2 + (py - yy)**2)/len(contour)
                counter_y += 1
            X = numpy.linalg.norm(X-c, axis=1)/len(contour)
            X = X.reshape((len(contour),2))
            
            # X = sklearn.preprocessing.scale(X, axis=1)
            X = X - numpy.mean(X, axis=0)
            X = X / numpy.std(X, axis=0, ddof=1)
            X = X.T
            bias_X = -1*numpy.ones(X.shape[1]).reshape(1, X.shape[1])
            X = numpy.concatenate((bias_X, X))
            Z = sigmoid(numpy.dot(self.W[i], X))
            bias_Z = -1*numpy.ones(Z.shape[1]).reshape(1, Z.shape[1])
            Z = numpy.concatenate((bias_Z, Z))
            M.append(numpy.dot((numpy.dot(D,Z.T)),(numpy.linalg.inv(numpy.dot(Z,Z.T)))))

        return numpy.concatenate(tuple([numpy.array(m).flatten() for m in M]), axis=None)
        
    def angle(self, contour):
        cy, cx = numpy.mean(contour, axis=0)
        M = list()
        for i, r in enumerate(self.params):
            X = list()
            D = list()

            r = int(r/2)
            for j, (py, px) in enumerate(contour):
                x = list()
                for (yy, xx) in [contour[(k)%len(contour)] for k in range(j-r, j+r+1, 1)]:
                    x.append((numpy.degrees(numpy.arctan2((yy-cy), (xx-cx))) + 360) % 360)
                X.append(x)
                D.append(numpy.sqrt((px - cx)**2 + (py - cy)**2)/len(contour))
            
            X = numpy.array(X).T
            D = numpy.array(D)
            X = sklearn.preprocessing.scale(X, axis=1)
            bias_X = -1*numpy.ones(X.shape[1]).reshape(1, X.shape[1])
            X = numpy.concatenate((bias_X, X))
            Z = sigmoid(numpy.dot(self.W[i], X))
            bias_Z = -1*numpy.ones(Z.shape[1]).reshape(1, Z.shape[1])
            Z = numpy.concatenate((bias_Z, Z))
            M.append(numpy.dot((numpy.dot(D,Z.T)),(numpy.linalg.inv(numpy.dot(Z,Z.T)))))
        return numpy.concatenate(tuple([numpy.array(m).flatten() for m in M]), axis=None)

class StackedContourDescriptor():
    def __init__(self, descriptors):
        assert type(descriptors) == list
        assert all(isinstance(x, ContourDescriptor) for x in descriptors)
        self.descriptors = descriptors
    
    def generate_contour(self, image):
        raise NotImplementedError("Contour generation not yet implemented")
    
    def extract_contour_features(self, image=None, contour=None):
        if image is None and contour is None:
            raise ValueError("At least one image or contour must be provided")
        elif image is not None and contour is None:
            contour = self.generate_contour(image)
        return numpy.concatenate(tuple([d.extract_contour_features(contour=contour) for d in self.descriptors]), axis=None)