import numpy as np

from sigmoid import sigmoid

def softmax(x):
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x

def digit(x):
    ''' Select highest prob from x and return index'''
    return np.round(np.argmax(x)+1) # Add 1 here because of matlab/python indexing

def predict(Theta1, Theta2, X):
#PREDICT Predict the label of an input given a trained neural network
#   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
#   trained weights of a neural network (Theta1, Theta2)

# Useful values
    m = np.shape(X)[0]              #number of examples

# You need to return the following variables correctly 
    p = np.zeros(m);

# ====================== YOUR CODE HERE ======================
# Instructions: Complete the following code to make predictions using
#               your learned neural network. You should set p to a 
#               vector containing labels between 1 to num_labels.
#

    # Add bias to X
    ones = np.ones([m,1])
    X = np.c_[ones,X]

    for i in range(m):
        z_1 = np.dot(Theta1, X[i])
        a_1 = sigmoid(z_1)
    
        # Add bias to a_1
        a_1 = np.concatenate((np.array([1,]), a_1), axis=None)
        z_2 = np.dot(Theta2, a_1)
        a_2 = sigmoid(z_2)  
     
        y_hat = softmax(a_2)
        p[i] = digit(y_hat)

    return p

# =========================================================================
