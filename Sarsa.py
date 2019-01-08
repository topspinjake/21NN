# Jake Williams - program to train 21NN using on-policy Sarsa TD learning

from Blackjack import *
from NN import *
import torch

trainingEpochs = 100
playEpochs = 100
totalEpochs = 100

env = Blackjack()

#One play loop looks like:
data = torch.zeros(14)
while not env.done:
    

#One training loop looks like:
