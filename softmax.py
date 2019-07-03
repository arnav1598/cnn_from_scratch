import numpy as np

class Softmax:
    
    def __init__(self, input_len, nodes):
        self.weights = np.random.randn(input_len, nodes) / input_len
        self.biases = np.zeros(nodes)
    
    def forward(self, inp):
        # Performs a forward pass of the softmax layer using the given input.
        # Returns a 1d numpy array containing the respective probability values.
        
        self.last_input_shape = inp.shape
        
        inp = inp.flatten()
        self.last_input = inp
        
        input_len, nodes = self.weights.shape
        
        totals = np.dot(inp, self.weights) + self.biases
        self.last_totals = totals
        exp = np.exp(totals)
        return exp / np.sum(exp, axis = 0)
    
    def backprop(self, dl_dout, learn_rate):
        # Performs a backward pass of the softmax layer.
        # Returns the loss gradient for this layer's inputs.
        
        for i, gradient in enumerate(dl_dout):
            if gradient == 0:
                continue
            
            t_exp = np.exp(self.last_totals)
            
            s = np.sum(t_exp)
            
            dout_dt = -t_exp[i] * t_exp / (s ** 2)
            dout_dt[i] = t_exp[i] * (s - t_exp[i]) / (s ** 2)
            
            dt_dw = self.last_input
            dt_db =1
            dt_dinputs = self.weights
            
            dl_dt = gradient * dout_dt
            
            dl_dw = dt_dw[np.newaxis].T @ dl_dt[np.newaxis]
            dl_db = dl_dt * dt_db
            dl_dinputs = dt_dinputs @ dl_dt
            
            self.weights -= learn_rate * dl_dw
            self.biases -= learn_rate * dl_db
            
            return dl_dinputs.reshape(self.last_input_shape)