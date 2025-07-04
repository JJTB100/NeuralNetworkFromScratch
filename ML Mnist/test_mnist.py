import pandas as pd
import numpy as np
from mnist import forward_propagation, get_predictions, get_accuracy
import matplotlib.pyplot as plt
from PIL import Image
# Load weights and biases from a single file
params = np.load("emnist_digits_all_weights.npz")
W1, b1 = params["W1"], params["b1"]
W2, b2 = params["W2"], params["b2"]
W3, b3 = params["W3"], params["b3"]



# Load test data
test_data = pd.read_csv('emnist-digits-test.csv')
test_data = np.array(test_data).T
Y_test = test_data[0]
X_test = test_data[1:] / 255.
# Forward pass on test data
_, _, _, _, _, A3_test, _, _ = forward_propagation(W1, b1, W2, b2, W3, b3, X_test, training=False)
test_predictions = get_predictions(A3_test)
test_accuracy = get_accuracy(test_predictions, Y_test)
print(f"Test accuracy = {test_accuracy*100:.4f}%")
