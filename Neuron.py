import math
import random
class Neuron:
    def __init__(self, number_of_inputs,inputs,weights):
        self.number_of_inputs = number_of_inputs
        self.weights = []
        self.inputs = inputs
        self.bias = 0
        self.weights = weights

    def setInputs(self,inputs):
        self.inputs = inputs
        
    def Z(self):
        z = 0
        for i in range(self.number_of_inputs):
            z += self.weights[i] * self.inputs[i]
        z += self.bias
        return z
    
    def sigmoid(self):
        return 1 / (1 + math.exp(-self.Z()))
    
    def differentiationOfSigmoid(self):
        return self.sigmoid() * (1 - self.sigmoid())
    
    def differentiationOfZ(self,target): # w1674368   x468287  y4893804   w3   x5   y8
        index = int(target[1:]) -1
        if "w" in target.lower():
            return self.inputs[index]
        elif "y" in target.lower() or "x" in target.lower():
            # Your code here
            return self.weights[index]
        

inputs = [0.23,0.82]

NeuronC = Neuron(2,inputs,[0.1,0.4])
NeuronD = Neuron(2,inputs,[0.5,0.3])

NeuronF = Neuron(2,[NeuronC.sigmoid(),NeuronD.sigmoid()],[0.2,0.6])

print(f"Neuron F sigmoid: {NeuronF.sigmoid()}")



def calculateErrorOf1Input(predictedVector, targetVector):  # 1/2 * sum(target - predicted)^2
   error = 0
   print(f"Predicted Vector is : {predictedVector}")
   print(f"Target Vector is : {targetVector}")
   for i in range(len(targetVector)):
      error += (targetVector[i]-predictedVector[i])**2   # target : [0, 0, 1, 0]   predicted: [0.1, 0.2, 0.7, 0]
   return error * 0.5

def targetToVector(target):
   """
   Converts a target integer into a one-hot encoded vector of length 4.

   Args:
      target (int): The target integer, expected to be in the range 1 to 4.

   Returns:
      list: A one-hot encoded list of length 4, where the position corresponding 
         to the target integer is set to 1, and all other positions are set to 0.

   Example:
      >>> targetToVector(3)
      [0, 0, 1, 0]
   """
   return [1 if i+1 == target else 0 for i in range(1)]

def train(data , expectedOutput , hiddenLayer, outputLayer):
      hiddenLayerOutputs = []
      outputLayerOutputs = []
      hotY = targetToVector(expectedOutput)
      for neuron in hiddenLayer:
         neuron.setInputs(data)
         hiddenLayerOutputs.append(neuron.sigmoid())
      for i in range(len(outputLayer)):
         outputLayer[i].setInputs(hiddenLayerOutputs)
         outputLayerOutputs.append(outputLayer[i].sigmoid())
      return calculateErrorOf1Input(outputLayerOutputs, hotY) , hiddenLayerOutputs, outputLayerOutputs


batches = [[([0.23,0.82],1)]]
hiddenLayer = [NeuronC,NeuronD]
outputLayer = [NeuronF]
learningRate = 0.7
number_of_batches = len(batches)
print("Number of Batches: ", number_of_batches)
for batch in batches: 
    error = 0
    for dataPoint in batch:
        print("Data Point: ", dataPoint) # should print [0.23,0.82]
        singleError , hiddenLayerOutputs, outputLayerOutputs = train(dataPoint[0], dataPoint[1], hiddenLayer, outputLayer)
        error += singleError  
    # after finishing the batch
    # diff output * y/z  * z/w
    for i in range(len(outputLayer)): # back propagation for output layer
        # print(f"len of outputLayer: {len(outputLayer)}")
        diffOfError = - (dataPoint[1] - outputLayerOutputs[i])
        diffOfSigmoid = outputLayer[i].differentiationOfSigmoid()
        for j in range(len(outputLayer[i].weights)):
           # print(f"len of outputLayer[i].weights: {len(outputLayer[i].weights)}")
            diffOfZ = outputLayer[i].differentiationOfZ(f"w{j+1}")
            deltaW = diffOfError * diffOfSigmoid * diffOfZ
            print(f"Old w{j+1}: {outputLayer[i].weights[j]}")
            outputLayer[i].weights[j] -= learningRate * deltaW
            print(f"New w{j+1}: {outputLayer[i].weights[j]}")

    for i in range(len(outputLayer)): # back propagation for hidden layer
        # print(f"len of outputLayer: {len(outputLayer)}")
        diffOfError = - (dataPoint[1] - outputLayerOutputs[i])
        diffOfSigmoid = outputLayer[i].differentiationOfSigmoid()
        for j in range(len(outputLayer[i].weights)):
           # print(f"len of outputLayer[i].weights: {len(outputLayer[i].weights)}")
            diffOfZ = outputLayer[i].differentiationOfZ(f"y{j+1}")
            #for k in range(len(hiddenLayer[j])):
            #  print(f"len of hiddenLayer: {len(hiddenLayer)}")
            diffOf2ndSigmoid = hiddenLayer[j].differentiationOfSigmoid()
            for l in range(len(hiddenLayer[j].weights)):
            # print(f"len of hiddenLayer[k].weights: {len(hiddenLayer[k].weights)}")
                diffOfInput = hiddenLayer[j].differentiationOfZ(f"x{l+1}")
                deltaW = diffOfError * diffOfSigmoid * diffOfZ  * diffOf2ndSigmoid * diffOfInput
                print(f"Old w{l+1}: {hiddenLayer[j].weights[l]}")
                hiddenLayer[j].weights[l] -= learningRate * deltaW
                print(f"New w{l+1}: {hiddenLayer[j].weights[l]}")