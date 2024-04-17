import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



class ClassicalNN(nn.Module):
    def __init__(self, n_classes, use_qiskit = False):
        super(ClassicalNN, self).__init__()
        #TODO change default linear layer sizes to match kind of what QNN does
        
        # self.fc1 = nn.Linear(16, 16) # First hidden layer
        # self.fc2 = nn.Linear(16, 16) # Second hidden layer
        self.fc3 = nn.Linear(16, n_classes) # Output layer

    def forward(self, x, use_qiskit=False): # input is bs, 1, 28, 28; cannot use qiskit here
        # print("x.shape", x.shape) #TODO if performance of QNN is too low can add equivalent pooling layer here
        x = F.avg_pool2d(x, 6).reshape(x.shape[0], -1)
        # print("x.shape", x.shape)
        x = x.reshape(x.shape[0], -1)
        # x = F.relu(self.fc1(x)) # Activation function for first hidden layer
        # x = F.relu(self.fc2(x)) # Activation function for second hidden layer
        x = self.fc3(x) # No activation, raw scores
        return F.log_softmax(x, dim=1) # Log-softmax for output