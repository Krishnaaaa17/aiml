 
import numpy as np 
X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float) 
y = np.array(([92], [86], [89]), dtype=float) 
X = X / np.amax(X, axis=0) 
y = y / 100 
def sigmoid(x): 
    return 1 / (1 + np.exp(-x)) 
def derivatives_sigmoid(x): 
    return x * (1 - x) 
epoch = 1000 
learning_rate = 0.6 
inputlayer_neurons = 2 
hiddenlayer_neurons = 3 
output_neurons = 1 
wh = np.random.uniform(size = (inputlayer_neurons,hiddenlayer_neurons)) 
bh = np.random.uniform(size = (1, hiddenlayer_neurons)) 
wo = np.random.uniform(size = (hiddenlayer_neurons,output_neurons)) 
bo = np.random.uniform(size = (1, output_neurons)) 
for i in range(epoch): 
    net_h = np.dot(X, wh) + bh 
    sigma_h = sigmoid(net_h) 
    net_o = np.dot(sigma_h, wo) + bo 
    output = sigmoid(net_o) 
    deltaK = (y - output) * derivatives_sigmoid(output) 
    deltaH = deltaK.dot(wo.T) * derivatives_sigmoid(sigma_h) 
    wo = wo + sigma_h.T.dot(deltaK) * learning_rate 
    wh = wh + X.T.dot(deltaH) * learning_rate 
print ("Input: \n" + str(X)) 
print ("Actual Output: \n" + str(y)) 
print ("Predicted Output: \n", output)