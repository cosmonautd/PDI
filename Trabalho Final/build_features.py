import numpy

X1 = numpy.genfromtxt("X.csv", delimiter=",")
Y1 = numpy.genfromtxt("Y.csv", delimiter=",")

X2 = numpy.genfromtxt("X2.csv", delimiter=",")
Y2 = numpy.genfromtxt("Y2.csv", delimiter=",")

X3 = numpy.hstack((X1,X2))
Y3 = numpy.hstack((Y1,Y2))

numpy.savetxt("X3.csv", X3, delimiter=",")
numpy.savetxt("Y3.csv", Y3, delimiter=",")