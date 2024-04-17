import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



class ClassicalNN(nn.Module):
    def __init__(self, n_classes, use_qiskit = False):
        super(ClassicalNN, self).__init__()
        #TODO change default linear layer sizes to match kind of what QNN does
        
        self.fc1 = nn.Linear(16, 50)
        self.fc2 = nn.Linear(50, 10)
        self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(10, 10)
        self.fc5 = nn.Linear(10, n_classes) # Output layer

    def forward(self, x, use_qiskit=False): # input is bs, 1, 28, 28; cannot use qiskit here
        # print("x.shape", x.shape)
        bsz = x.shape[0]
        x = F.avg_pool2d(x, kernel_size=6, stride=6)
        x = x.reshape(bsz, -1) # bsz, 16
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x) # No activation, raw scores
        return F.log_softmax(x, dim=1) # Log-softmax for output