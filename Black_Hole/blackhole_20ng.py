import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.parasite_axes import host_subplot
from mpl_toolkits.axisartist.axislines import Axes
from scipy.sparse.csr import csr_matrix

from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn.metrics import hamming_loss


bunch = datasets.fetch_20newsgroups_vectorized(subset='all')
#print("bunch = ", bunch)
#print("type = ", type(bunch))
X, y = shuffle(bunch.data, bunch.target)
offset = int(X.shape[0] * 0.8)
X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]

print("X type  = ", type(X))
#X = X.toarray()
#print("X type  = ", type(X), X.shape)
import pandas as pd
X = pd.DataFrame.sparse.from_spmatrix(X)
print("x = ", X)
print("X type = ", type(X))
print("y = ", y, "y.shape = ", y.shape," y tpe = ", type(y))

import tensorflow as tf
Y = tf.keras.utils.to_categorical(
    y, num_classes=20,
)
print("y shape = ", Y.shape, " y type = ", type(Y))




