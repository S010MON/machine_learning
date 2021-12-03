import numpy as np
from sigmoid import sigmoid
from sigmoidGradient import sigmoidGradient

def softmax(x):
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x

def digit(x):
    ''' Select highest prob from x and return index'''
    return np.round(np.argmax(x)+1) # Add 1 here because of matlab/python indexing

def one_hot(i, max):
    ''' return a vector of size max where the entry i is 1'''
    v = np.zeros([max,])
    v[i-1] = 1
    return v

def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_value):
#   NNCOSTFUNCTION Implements the neural network cost function for a two layer
#   neural network which performs classification
#   nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_value)
#   computes the cost and gradient of the neural network. The
#   parameters for the neural network are "unrolled" into the vector
#   nn_params and need to be converted back into the weight matrices. 
# 
#   The returned parameter grad should be a "unrolled" vector of the
#   partial derivatives of the neural network.
#

# Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
# for our 2 layer neural network
    tmp = nn_params.copy()
    Theta1 = np.reshape(tmp[0:hidden_layer_size * (input_layer_size + 1)],
                          (hidden_layer_size, (input_layer_size + 1)), order='F')
    Theta2 = np.reshape(tmp[(hidden_layer_size * (input_layer_size + 1)):len(tmp)],
                          (num_labels, (hidden_layer_size + 1)), order='F')

# Setup some useful variables
    m = np.shape(X)[0]

# Computation of the Cost function including regularisation
# Feedforward 
    a_2 = sigmoid(np.dot(np.hstack((np.ones((m, 1)), X)), np.transpose(Theta1)))
    a_3 = sigmoid(np.dot(np.hstack((np.ones((m, 1)), a_2)), np.transpose(Theta2)))

    # Cost function for Logistic Regression summed over all output nodes
    Cost = np.empty((num_labels, 1))
    for k in range(num_labels):
        # which examples fit this label
        y_binary=(y==k+1)
        # select all predictions for label k
        hk=a_3[:,k]
        # compute two parts of cost function for all examples for node k
        Cost[k][0] = np.sum(np.transpose(y_binary)*np.log(hk)) + np.sum(((1-np.transpose(y_binary))*np.log(1-hk)))

# Sum over all labels and average over examples
    J_no_regularisation = -1./m * sum(Cost)
# No regularization over intercept
    Theta1_no_intercept = Theta1[:, 1:]
    Theta2_no_intercept = Theta2[:, 1:]

# Sum all parameters squared
    RegSum1 = np.sum(np.sum(np.power(Theta1_no_intercept, 2)))
    RegSum2 = np.sum(np.sum(np.power(Theta2_no_intercept, 2)))
# Add regularisation term to final cost
    J = J_no_regularisation + (lambda_value/(2*m)) * (RegSum1+RegSum2)

# You need to return the following variables correctly 
    Theta1_grad = np.zeros(np.shape(Theta1))
    Theta2_grad = np.zeros(np.shape(Theta2))

# ====================== YOUR CODE HERE ======================
# Implement the backpropagation algorithm to compute the gradients
# Theta1_grad and Theta2_grad. You should return the partial derivatives of
# the cost function with respect to Theta1 and Theta2 in Theta1_grad and
# Theta2_grad, respectively. After implementing Part 2, you can check
# that your implementation is correct by running checkNNGradients
#
# Note: The vector y passed into the function is a vector of labels
#       containing values from 1..K. You need to map this vector into a 
#       binary vector of 1's and 0's to be used with the neural network
#       cost function.
#
# Hint: It is recommended implementing backpropagation using a for-loop
#       over the training examples if you are implementing it for the 
#       first time.
    
    ones = np.ones([m,1])
    X = np.c_[ones,X]
                   
    for i in range(m):
        
        a_0 = X[i]
        z_1 = np.dot(Theta1, a_0)       
        a_1 = sigmoid(z_1)
        a_1 = np.concatenate((np.array([1,]), a_1), axis=None)
        z_2 = np.dot(Theta2, a_1)
        a_2 = sigmoid(z_2)  
        y_hat = digit(softmax(a_2))
            
        d_2 = a_2 - one_hot(y[i], np.shape(a_2)[0])
        
        A = sigmoidGradient(z_1)
        B = np.dot(Theta2[:, 1:].T, d_2)
        d_1 = np.multiply(A, B)

        d_2 = d_2.reshape((np.shape(d_2)[0], 1))
        a_1 = a_1.reshape((np.shape(a_1)[0], 1))
        Theta2_grad += np.dot(d_2, a_1.T)
        
        d_1 = d_1.reshape((np.shape(d_1)[0], 1))
        a_0 = a_0.reshape((np.shape(a_0)[0], 1))
        
        print(str(np.shape(a_0.T)) + str(a_0.T))
        print(str(np.shape(d_1)) + str(d_1))
        
        Theta1_grad += np.dot(d_1, a_0.T)

    Theta1_grad = Theta1_grad/m
    Theta2_grad = Theta2_grad/m
    
    Theta1_grad[:,1:] += (lambda_value/m) * Theta1[:,1:]
    Theta2_grad[:,1:] += (lambda_value/m) * Theta2[:,1:]

# =========================================================================

# Unroll gradients
    Theta1_grad = np.reshape(Theta1_grad, Theta1_grad.size, order='F')
    Theta2_grad = np.reshape(Theta2_grad, Theta2_grad.size, order='F')
    grad = np.expand_dims(np.hstack((Theta1_grad, Theta2_grad)), axis=1)

    return J, grad

