import numpy as np

class NeuralNetwork:
    def __init__(self, x, y):
        self.input      = x
        print (self.input.shape[0])
        self.weights1   = np.random.rand(784,784) 
        print (self.weights1)
        self.weights2   = np.random.rand(4,1)   
        self.y          = y
        self.output     = np.zeros(self.y.shape)

if __name__ == "__main__":
    X = np.array([[0,1,0],
                  [1,1,0],
                  [0,1,0],
                  [1,1,1]])
    
    y = np.array([[0],[1],[0],[1]])
    nn = NeuralNetwork(X,y)