# With m training images, each being 28x28 pixels, 784 total.
# We can represent the images in a matrix X shape (m, 784)^T -> (784, m) each col is a image.
# NN to have 3 layers, input layer with 784 neurons, hidden with 10, and output layer with 10 neurons.

# Forward Propagation:
# 1. Input layer: A0 = X
# 2. Hidden layer: Z1 = W1.A0 + b1 where W1 is (10, 784) and b1 is a bias (10, 1)
# 3. Activation: A1 = ReLU(Z1)
# 4. Output layer: Z2 = W2.A1 + b2 where W2 is (10, 10) and b2 is a bias (10, 1)
# 5. Activation: A2 = softmax(Z2) <- makes sure the output is a probability distribution

# Backward Propagation:
# 1. Compute the loss: dZ2 = A2 - Y where Y is the true label (one-hot encoded)
# 2. Compute gradients: dW2 = dZ2.A1^T / m              
#                       db2 = sum(dZ2) / m
# 3. Backpropagate to hidden layer: dZ1 = W2^T.dZ2 * ReLU'(Z1)
# 4. Compute gradients for hidden layer: dW1 = dZ1.A0^T / m
#                                        db1 = sum(dZ1) / m
# 5. Update weights: W1 -= learning_rate * dW1
#                    W2 -= learning_rate * dW2
#                    b1 -= learning_rate * db1
#                    b2 -= learning_rate * db2   
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt 
NUMBER_OF_CLASSES = 10
HIDDEN_LAYER1_SIZE = 532
HIDDEN_LAYER2_SIZE = 200
def init_params():
    W1 = np.random.randn(HIDDEN_LAYER1_SIZE, 784) * np.sqrt(2. / 784)
    b1 = np.zeros((HIDDEN_LAYER1_SIZE, 1))
    print(b1.shape, W1.shape)

    W2 = np.random.randn(HIDDEN_LAYER2_SIZE, HIDDEN_LAYER1_SIZE) * np.sqrt(2. / HIDDEN_LAYER1_SIZE)
    b2 = np.zeros((HIDDEN_LAYER2_SIZE, 1))

    W3 = np.random.randn(NUMBER_OF_CLASSES, HIDDEN_LAYER2_SIZE) * np.sqrt(2. / HIDDEN_LAYER2_SIZE)
    b3 = np.zeros((NUMBER_OF_CLASSES, 1))
    
    iterations_done = 0
    return W1, b1, W2, b2, W3, b3, iterations_done



def ReLU(Z):
    # Standard ReLU activation
    A = np.maximum(0, Z)
    return A

def softmax(x):
    f = np.exp(x - np.max(x, axis=0, keepdims=True))  # shift values
    return f / f.sum(axis=0, keepdims=True)
    

def forward_propagation(W1, b1, W2, b2, W3, b3, X, training=True):
    Z1 = W1.dot(X) + b1
    A1 = leaky_ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = leaky_ReLU(Z2)
    Z3 = W3.dot(A2) + b3
    A3 = softmax(Z3)
    
    
    # Dropout for hidden layers
    dropout_rate1 = 0.2  # e.g., 20% dropout for first hidden layer
    dropout_rate2 = 0.2  # e.g., 20% dropout for second hidden layer
    if not training:
        dropout_rate1 = 0.0
        dropout_rate2 = 0.0
    D1 = (np.random.rand(*A1.shape) > dropout_rate1).astype(float)
    A1 *= D1
    A1 /= (1 - dropout_rate1)
    D2 = (np.random.rand(*A2.shape) > dropout_rate2).astype(float)
    A2 *= D2
    A2 /= (1 - dropout_rate2)
    # Save masks for backward pass
    forward_propagation.D1 = D1
    forward_propagation.D2 = D2

    return Z1, A1, Z2, A2, Z3, A3, D1, D2


def one_hot(Y):
    one_hot_Y = np.zeros((NUMBER_OF_CLASSES, Y.size))
    one_hot_Y[Y, np.arange(Y.size)] = 1
    return one_hot_Y

def deriv_ReLU(Z):
    return Z > 0
def leaky_ReLU(Z, alpha=0.01):
    return np.where(Z > 0, Z, alpha * Z)

def deriv_leaky_ReLU(Z, alpha=0.01):
    return np.where(Z > 0, 1, alpha)

def backward_prop(Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, X, Y, D1, D2):
    m = X.shape[1]
    dZ3 = A3 - one_hot(Y)
    dW3 = 1 / m * dZ3.dot(A2.T)
    db3 = 1 / m * np.sum(dZ3, axis=1, keepdims=True)

    # Backprop for second hidden layer (no dropout)
    dA2 = W3.T.dot(dZ3)
    dZ2 = dA2 * deriv_leaky_ReLU(Z2)
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)

    # Backprop for first hidden layer (no dropout)
    dA1 = W2.T.dot(dZ2)
    dZ1 = dA1 * deriv_leaky_ReLU(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)
    
    return dW1, db1, dW2, db2, dW3, db3
    
def update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2 
    W3 = W3 - alpha * dW3
    b3 = b3 - alpha * db3
    return W1, b1, W2, b2, W3, b3

def get_predictions(A3):
    return np.argmax(A3, 0)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

def create_batches(X, Y, batch_size):
    m = X.shape[1]
    permutation = np.random.permutation(m)
    X_shuffled = X[:, permutation]
    Y_shuffled = Y[permutation]

    batches = []
    for k in range(0, m, batch_size):
        X_batch = X_shuffled[:, k:k+batch_size]
        Y_batch = Y_shuffled[k:k+batch_size]
        batches.append((X_batch, Y_batch))
    return batches

def gradient_descent(X, Y, alpha, iterations, reset=False):
    if(not reset):
        try:
            params = np.load("mnist_all_weights.npz")
            W1, b1 = params["W1"], params["b1"]
            W2, b2 = params["W2"], params["b2"]
            W3, b3 = params["W3"], params["b3"]
            iterations_done = params["iterations"]
            print("Loaded existing parameters.")
        except FileNotFoundError:
            print("No existing parameters found. Initializing new parameters.")
            W1, b1, W2, b2, W3, b3, iterations_done = init_params()
    else:
        W1, b1, W2, b2, W3, b3, iterations_done = init_params()
    
    for i in range(iterations):
        batches = create_batches(X, Y, batch_size=64)
        for X_batch, Y_batch in batches:
            Z1, A1, Z2, A2, Z3, A3, D1, D2 = forward_propagation(W1, b1, W2, b2, W3, b3, X_batch)
            dW1, db1, dW2, db2, dW3, db3 = backward_prop(Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, X_batch, Y_batch, D1, D2)
            W1, b1, W2, b2, W3, b3 = update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha)
        
        if i % 1 == 0:
            _, _, _, _, _, A3_full, _, _ = forward_propagation(W1, b1, W2, b2, W3, b3, X)
            predictions = get_predictions(A3_full)
            acc = get_accuracy(predictions, Y)
            print(f"Epoch {i + iterations_done}: Training accuracy = {acc:.4f}")
    return W1, b1, W2, b2, W3, b3, iterations_done + iterations


