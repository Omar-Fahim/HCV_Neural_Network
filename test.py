import np
class Neuron:
    def __init__(self, number_of_inputs,inputs):
        self.number_of_inputs = number_of_inputs
        self.weights = []
        self.inputs = inputs
        self.bias = 0
        # self.bias = 1
        for i in range(number_of_inputs):
            self.weights.append(random.uniform(-1, 1) )
            # self.weights.append(1)

    def setInputs(self,inputs):
        self.inputs = inputs
        
    def Z(self):
        z = 0
        for i in range(self.number_of_inputs):
            z += self.weights[i] * self.inputs[i]
        z += self.bias
        return z
    
    def sigmoid(self):
        return 1 / (1 + np.exp(-self.Z()))
    
    def differentiationOfSigmoid(self):
        return self.sigmoid() * (1 - self.sigmoid())
    
    def differentiationOfZ(self,target): # w1674368   x468287  y4893804   w3   x5   y8
        index = int(target[1:]) -1
        if "w" in target.lower():
            return self.inputs[index]
        elif "y" in target.lower() or "x" in target.lower():
            # Your code here
            return self.weights[index]
    def __str__(self) -> str:
        return str(self.weights) + " " + str(self.bias)