from cgi import test
import sys
import numpy as np
import math
import time

# Global variables
# Make layers a global variable
layer = []

# Activation functions
# I could have used the activations library, but its a needlessly big one for my purposes    
# Sigmoid 
def sigmoid(x):
    return 1/(1 + math.exp(-x))

# Derivative of Sigmoid
def dSigmoid(x):
    return math.exp(-x)/((1+math.exp(-x))**2)

# ReLU
def relu(x):
    return max(0,x)

# Derivative of ReLU
def dRelu(x):
    return 1*(x>=0)

# Softmax
exp = np.vectorize(math.exp)
def softmax(x):
    S = sum(exp(x))
    for i in range(len(x)):
        x[i] = (1/S)*exp(x[i])
    return x

# Initialise the activation function and it's derivative globally
f = 0
df = 0

# Global number of layers
N = 0


# The weights in the layer will be the weights going in
class Layer:
    def __init__(self, prevnodes, nodes, af):
        # prevnodes is the number of nodes in the previous layer
        # nodes is the number of nodes in this layer
        # af is the activation function at the layer
        
        # Initialise the layer's nodes and af
        self.nodes = nodes
        self.af = af

        # Initialise weights based on which activation function will be used.
        # Specifically, ReLU shouldn't be initialised to 0s
        # We'll start with initialising Sigmoid layers
        if af == 0:
            # This is (part of the) the range 
            a = 2
            self.W = np.matrix(a*np.random.rand(prevnodes, nodes) - a/2)
        else:
            # Initialising ReLU using He initialisation
            a = (math.sqrt(2/prevnodes))
            self.W = np.matrix(2*a*np.random.rand(prevnodes, nodes))
         
         #  Initialise everything else
        self.B = np.matrix(np.zeros((nodes, 1))) # Biases
        self.n = np.matrix(np.zeros((nodes, 1))) # Unactivated outputs
        self.a = np.matrix(np.zeros((nodes, 1))) # Activated output
        self.S = np.matrix(np.zeros((nodes, 1))) # Sigma


class fashionMNISTNN:

    def __init__(self, layerSizes, af):
        # layerSizes is a list with the sizes for the input, hidden and output layers
        #   Should be at least 2, for input and ouput
        # af is the activation function at all layers, except the last
        #   0 means Sigmoid, anything else means ReLU

        # Get the number of layers
        self.N = len(layerSizes)

        #TODO Check that inputs are valid
        if len(layerSizes) < 2:
            print("Number of layers should be at least 2.")
            sys.exit(1)
               
        # Initialise the layers
        for i in range(1,len(layerSizes)):
            layer.append(Layer(layerSizes[i-1], layerSizes[i], af))

        

    def train(self, lr, N_ep, tsf, cf, af):
        # lr is the learning rate
        # N_ep is the number of epochs
        # tsf is the traning size fraction (0 < tsf <= 1)
        # cf is the cost function, it calculates the error
        #   1 means cross entropy (XE), 0 means Total Square Error (TSE)
        # data is an array of the form ["/dir/to/train.csv", "/dir/to/test.csv"]

        # Get N
        N = self.N
        
        # Get the activation functions and their derivatives
        if af == 0:
            f = np.vectorize(sigmoid)
            df = np.vectorize(dSigmoid)
        else:
            f = np.vectorize(relu)
            df = np.vectorize(dRelu)

        # Check that the inputs are valid
        if lr <= 0:
            print("Learning rate cannot be less than or equal to 0.")
            return
        if N_ep <=0:
            print("N_ep, number of ep's nodes aochs has to be greater than 0.")
            return
        if tsf <=0 or tsf > 1:
            print("tsf, training size fraction has to be between 0 and 1, it can be 1.")
            return
        
        # Load the Data
        Data = np.matrix(np.loadtxt("Data.csv", delimiter=","))

        # Partition into Train, Validation, Test from 74 Datapoints:
        # 60 training, 7 Validation, 7 Test
        reducer = 10**(10)
        Data[:, 0] = Data[:, 0]*reducer
        trainSet = Data[:40]/reducer
        valSet = Data[40:60]/reducer
        testSet =Data[67:]/reducer
        
        # Add sets to self
        self.trainSet = Data[:40]/reducer
        self.valSet   = Data[40:60]/reducer
        self.testSet  = Data[67:]/reducer
        
        # Get the number of train, validation, and test data points
        oldNTrain = len(trainSet)
        nVal = len(valSet)
        nTest = len(testSet)
        
        self.nVal = nVal
        self.nTest = nTest

        # Get the number of training samples for the training size fraction
        # nTrain will be indexed so it has to be an int
        nTrain = int(oldNTrain*tsf)

        # Train the network
        # Mixup the indices of the complete dataset
        mixup = np.random.permutation(oldNTrain)
        values = mixup[:nTrain]

        # Remember epoch starts at 0, this is sort of irrelevant, but it'll be 
        #   relevant later.
        for epoch in range(N_ep):
            # Print progress
            if epoch % 10 == 9:
                print(epoch)
            else:
                print(f"{epoch}  ", end="")

            
            # Get a random permutation of the data at each epoch to prevent 
            #   overfitting on the order of the data.
            values = values[np.random.permutation(nTrain)]
            
            # Train over the data 
            for j in range(nTrain):
                # Get a random index
                i = values[j]

                # Get the input
                x = trainSet[i, 1:].T

                # Get the true output
                t =  np.matrix(trainSet[i, 0])
                
                if t > trainSet[i,1]*reducer:
                    t = np.matrix([1 ,0]).T
                else:
                    t = np.matrix([0 ,1]).T
            
                
                # Forward Feeding
                # First feeding will be outside loop because we use x
                layer[0].n = layer[0].W.T.dot(x)
                layer[0].a = f(layer[0].n)
                
                # print(f"W[1,1]: {layer[0].W[1,1]}")
        
                               
                # for l in layer:
                #     print(l.a.shape)
                
                # Loop to forward feed all the way to the end
                for k in range(1,N-1):
                    # Using A.dot(B) because python threw an error
                    layer[k].n = layer[k].W.T.dot(layer[k-1].a) + layer[k].B
                    layer[k].a = f(layer[k].n)
                    
                    
                # Change the final activation functino to softmax if using cross entropy 
                if cf == 1:
                    layer[N-2].a = softmax(layer[N-2].n)

                # Calculate error
                e = t - layer[N-2].a
                # print("=======================================")
                # print(f"Error: {e}")
                # print(f"Output: {layer[N-2].a}")
                # print(f"True Value: {t}")
                # print("=======================================")
                
                # Get the first value of S (sigma)
                if cf == 0: # TSE
                    A = np.matrix(np.diag(df(layer[N-2].n.flat))) # NOTE FIRST POENTIAL ERROR MAY ARISE
                    layer[N-2].S = -2*A.dot(e)
                else: # Cross Entropy
                    layer[N-2].S = -e

                # Calculate the S value for remaining 
                for k in range(N-3,-1, -1):
                    # NOTE Not sure why I have to use np.matrix() here, since the dims agree
                    A = np.matrix(np.diag(df(layer[k].n.flat)))
                    layer[k].S = A*layer[k+1].W*layer[k+1].S
                
                # Update the weights and biases
                # First for the first set, because input includes x
                layer[0].W = layer[0].W - lr*x.dot(layer[0].S.T)
                layer[0].B = layer[0].B - lr*layer[0].S

                for k in range(1, N-1):
                    layer[k].W = layer[k].W - lr*layer[k-1].a.dot(layer[k].S.T)
                    layer[k].B = layer[k].B - lr*layer[k].S
        
    def validate(self, cf, af):
        # Get number of validation points 
        nVal = self.nVal
        valSet = self.valSet
        N = self.N
        
        # Get the activation functions and their derivatives
        if af == 0:
            f = np.vectorize(sigmoid)
            df = np.vectorize(dSigmoid)
        else:
            f = np.vectorize(relu)
            df = np.vectorize(dRelu)
        
        for i in range(nVal):
            # Get the input
            x = valSet[i, 1:].T

            # Get the true output
            t =  np.matrix(valSet[i, 0])
            
            if t > valSet[i,1]:
                t = np.matrix([1 ,0]).T
            else:
                t = np.matrix([0 ,1]).T
        
            
            # Forward Feeding
            # First feeding will be outside loop because we use x
            layer[0].n = layer[0].W.T.dot(x)
            layer[0].a = f(layer[0].n)
            
            # Loop to forward feed all the way to the end
            for k in range(1,N-1):
                # Using A.dot(B) because python threw an error
                layer[k].n = layer[k].W.T.dot(layer[k-1].a) + layer[k].B
                layer[k].a = f(layer[k].n)
                
                
            # Change the final activation functino to softmax if using cross entropy 
            if cf == 1:
                layer[N-2].a = softmax(layer[N-2].n)
            
            # Get the output
            y = layer[N-2].a
            
            # Get one-hot output
            hot = np.argmax(y)
            y = np.matrix([0,0]).T
            y[hot] = 1
            print("=====================")
            print(y)
            print(t)
            if np.subtract(t,y) == np.matrix([0, 0]).T:
                print("hi")
            
            
    
        pass
    
    def test():
        # Initialise a counter for the number of correct guesses
        count = 0

        # Test the network
        for i in range(nTest):

            # Get the input
            x = testSet[i, 1:].T

            # Get the true output
            t = np.zeros((10,1))
            t[int(testSet[i,0])] += 1

            ## Forward Feeding
            # First do it for the first layer
            layer[0].n = layer[0].W.T*x
            layer[0].a = f(layer[0].n)

            # For the remaining layers
            for i in range(1, N-1):
                layer[i].n = layer[i].W.T*layer[i-1].a
                layer[i].a = f(layer[i].n)
            
            # Change output layer AF to softmax if using XE
            layer[N-2].a = softmax(layer[N-2].n)
            y = layer[N-2].a
            
            # Get a one hot output
            # The index of the hottest element in the output
            ind = np.argmax(y)

            # Get the number of correct outputs
            if t[ind] == 1:
                count += 1
            
        # Print the number of correct guesses
        print(f"{count} correct guesses!")



                    

# Directories to the data
lr = 10**(-10)
N_ep = 30
tsf = 1
cf = 0 # XE
af = 1 # Sigmoid
layers = [170, 128, 64, 32, 16, 8, 4, 2]
# layers = [170, 128, 64, 8, 1]
# layers = [170, 64, 1]


nn = fashionMNISTNN(layers, af)
nn.train(lr, N_ep, tsf, cf, af)
nn.validate(cf, af)

