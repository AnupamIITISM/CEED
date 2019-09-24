# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 07:39:59 2019

@author: drshr
"""
# importing all the required libraries
import numpy as np
import matplotlib .pyplot as plt
    
# function to initialize parameters to be uniformly distributed random numbers
# between 0.0 and 1.0
def randInitializeWeights(L_in, L_out):
    W = np.random.rand(L_out, 1 + L_in)
    return W

# function to calculate sigmoid of activity
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# function to calculate sigmoid gradient
def sigmoidGradient(z):
    return np.multiply(sigmoid(z), 1 - sigmoid(z))

# function to compute cost and gradients
def computeCost(X, y, Theta1, Theta2):
    m, n = X.shape
    
    J = 0
    Theta1_grad = np.zeros(Theta1.shape)
    Theta2_grad = np.zeros(Theta2.shape)
    
    # Forward Propagation:
    
    # input layer values (with bias unit)
    a1 = np.concatenate((np.ones((m, 1)), X), axis=1)
    # calculating activity of hidden layer
    z2 = a1 * Theta1.T
    a, b = z2.shape
    # calculating activation of hidden layer (with bias unit)
    a2 = np.concatenate((np.ones((a, 1)), sigmoid(z2)), axis=1)
    # calculating activity of output layer
    z3 = a2 * Theta2.T
    # calculating activation of output layer
    a3 = sigmoid(z3)
    # hypothesis
    h = a3
    
    # calculating mean squared error cost
    J = (1/(2 * m)) * np.sum(np.square(np.subtract(h, y)))
    
    # Backpropagation:
    
    # calculating gradients
    d3 = h - y
    d2 = np.multiply(d3 * Theta2,  sigmoidGradient(np.concatenate((np.ones((a, 1)), z2), axis=1)))
    c, d = d2.shape
    d2 = d2[:, [1, d-1]]
    
    delta1 = d2.T * a1
    delta2 = d3.T * a2
    
    Theta1_grad = delta1 / m
    Theta2_grad = delta2 / m
    
    return J, Theta1_grad, Theta2_grad

# function for gradient descent
def gradientDescent(x, y, Theta1, Theta2, alpha, num_iters):
    
    # initializing matrix to store cost history
    J_history = np.zeros((num_iters,1))
    
    # initializing matrix to store parameter/theta history
    nn_params_history = np.matrix(np.concatenate((Theta1.ravel(), Theta2.ravel()), axis = 0))
    
    for iter in range(0, num_iters):
        
        J, Theta1_grad, Theta2_grad = computeCost(x, y, Theta1, Theta2)
        
        #updating parameters/thetas
        Theta1 = np.subtract(Theta1, alpha * Theta1_grad)
        Theta2 = np.subtract(Theta2, alpha * Theta2_grad)
        
        J_history[iter] = J
        
        nn_params_history = np.concatenate((nn_params_history, np.concatenate((Theta1.ravel(), Theta2.ravel()), axis = 1)), axis = 0)
        
    return J_history, nn_params_history, Theta1, Theta2

def main():
    
    input_layer_size = 2
    hidden_layer_size = 2
    output_layer_size = 2
    
    # training data
    x = np.matrix([0.05, 0.1])
    y = np.matrix([0.01, 0.99])
        
    m, n = x.shape
        
    initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
    initial_Theta2 = randInitializeWeights(hidden_layer_size, output_layer_size)
    
    # no. of iterations
    iterations = 7000
    
    # learning rate
    alpha = 0.1
        
    J_history, nn_params_history, Theta1, Theta2 = gradientDescent(x, y, initial_Theta1, initial_Theta2, alpha, iterations)
    
    # plotting total cost vs iterations
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('total cost vs iterations')
    ax.set_xlabel(r'iterations')
    ax.set_ylabel(r'$J(\theta)$')
    ax.scatter(range(iterations), J_history, color='blue', s=10)
    fig.set_size_inches(8, 5)
    plt.savefig('total cost vs iterations')
    fig.show()
    
    # plotting each parameter/theta vs iterations
    for i in range(12):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel(r'iterations')
        ax.scatter(range(iterations+1), nn_params_history[:,i], color='blue', s=10)
        fig.set_size_inches(8, 5)
        if i == 0:
            ax.set_title(r'$\theta^1_1,_0$' + ' vs iterations')
            ax.set_ylabel(r'$\theta^1_1,_0$')
            plt.savefig('theta_1_1_0 vs iterations')
        elif i == 1:
            ax.set_title(r'$\theta^1_1,_1$' + ' vs iterations')
            ax.set_ylabel(r'$\theta^1_1,_1$')
            plt.savefig('theta_1_1_1 vs iterations')
        elif i == 2:
            ax.set_title(r'$\theta^1_1,_2$' + ' vs iterations')
            ax.set_ylabel(r'$\theta^1_1,_2$')
            plt.savefig('theta_1_1_2 vs iterations')
        elif i == 3:
            ax.set_title(r'$\theta^1_2,_0$' + ' vs iterations')
            ax.set_ylabel(r'$\theta^1_2,_0$')
            plt.savefig('theta_1_2_0 vs iterations')
        elif i == 4:
            ax.set_title(r'$\theta^1_2,_1$' + ' vs iterations')
            ax.set_ylabel(r'$\theta^1_2,_1$')
            plt.savefig('theta_1_2_1 vs iterations')
        elif i == 5:
            ax.set_title(r'$\theta^1_2,_2$' + ' vs iterations')
            ax.set_ylabel(r'$\theta^1_2,_2$')
            plt.savefig('theta_1_2_2 vs iterations')
        elif i == 6:
            ax.set_title(r'$\theta^2_1,_0$' + ' vs iterations')
            ax.set_ylabel(r'$\theta^2_1,_0$')
            plt.savefig('theta_2_1_0 vs iterations')
        elif i == 7:
            ax.set_title(r'$\theta^2_1,_1$' + ' vs iterations')
            ax.set_ylabel(r'$\theta^2_1,_1$')
            plt.savefig('theta_2_1_1 vs iterations')
        elif i == 8:
            ax.set_title(r'$\theta^2_1,_2$' + ' vs iterations')
            ax.set_ylabel(r'$\theta^2_1,_2$')
            plt.savefig('theta_2_1_2 vs iterations')
        elif i == 9:
            ax.set_title(r'$\theta^2_2,_0$' + ' vs iterations')
            ax.set_ylabel(r'$\theta^2_2,_0$')
            plt.savefig('theta_2_2_0 vs iterations')
        elif i == 10:
            ax.set_title(r'$\theta^2_2,_1$' + ' vs iterations')
            ax.set_ylabel(r'$\theta^2_2,_1$')
            plt.savefig('theta_2_2_1 vs iterations')
        elif i == 11:
            ax.set_title(r'$\theta^2_2,_2$' + ' vs iterations')
            ax.set_ylabel(r'$\theta^2_2,_2$')
            plt.savefig('theta_2_2_2 vs iterations')
        fig.show()
        
if __name__ == '__main__':
    main()
