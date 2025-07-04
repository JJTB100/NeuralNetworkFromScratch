import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mnist import init_params, forward_propagation, gradient_descent

data = pd.read_csv('emnist-digits-train.csv')
data = np.array(data)
m, n = data.shape
np.random.shuffle(data) # shuffle before splitting into dev and training sets

data_train = data[:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.

W1, b1, W2, b2, W3, b3, iters = gradient_descent(X_train, Y_train, iterations=20, alpha=0.01, reset=True)

np.savez('emnist_digits_all_weights.npz', W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3, iterations=iters)
