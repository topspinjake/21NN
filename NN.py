# Jake Williams - NN to play Blackjack

import torch

device = torch.device('cpu')

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 10, 12, 20, 10

class BodyNet(torch.nn.Module):
    def __init__(self):
        super(BodyNet, self).__init__()
        self.linearOne = torch.nn.Linear(D_in, H),
        self.lineartwo = torch.nn.Linear(H, D_out),

    def forward(self, x):
        x = torch.relu(self.linearone(x))
        x = self.lineartwo(x)
        #x = self.dropout(x)
        #x = x.view(-1, poolSize)
        #x = self.fc1(x)
        return torch.sigmoid(x)

# V_h is the hidden dimension; V_out is the output dimension
V_h, V_out = 10, 1

class ValueHead(torch.nn.Module):
    def __init__(self):
        super(ValueHead, self).__init__()
        self.linearOne = torch.nn.Linear(D_out, V_h),
        self.lineartwo = torch.nn.Linear(V_h, V_out),

    def forward(self, x):
        x = torch.relu(self.linearone(x))
        x = self.lineartwo(x)
        return torch.tanh(x)

# A_h is the hidden dimension; A_out is the output dimension
A_h, A_out = 10, 1

class ActionHead(torch.nn.Module):
    def __init__(self):
        super(ActionHead, self).__init__()
        self.linearOne = torch.nn.Linear(D_out, A_h),
        self.lineartwo = torch.nn.Linear(A_h, A_out),

    def forward(self, x):
        x = torch.relu(self.linearone(x))
        x = self.lineartwo(x)
        return torch.sigmoid(x)