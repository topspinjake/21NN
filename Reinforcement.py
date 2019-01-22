# Jake Williams - program to train 21NN using on-policy Sarsa TD learning

from Blackjack import *
from NN import *
import torch
import random

trainingEpochs = 100
playEpochs = 100
totalEpochs = 100

env = Blackjack()

body = BodyNet()
valueHead = ValueHead()
actionHead = ActionHead()

#One play loop looks like:
data = torch.zeros(1,14)
value = 0
state = env.state()
while not env.done:
    newCard = random.randint(0, 51)
    while env.cards[newCard] == 1:
        newCard = random.randint(0, 51)
    mcState = state
    for i in range(2, 11):
        if mcState[i] == 0:
            mcState[i] = newCard
            break
    vS0 = valueHead(body(state))
    vS1 = valueHead(body(mcState))
    prob = (vS0+1) / (vS0 + vS1 + 2)
    d = torch.cat((state, torch.tensor([0.0, float(prob)])))
    data = torch.cat((data, d.expand(1, 14)), 0)
    print(data.size())
    action = round(actionHead(body(state)).tolist()[0])
    state, value = env.step(action)
data = data[1:]
for i in range(data.size()[0]):
    data[i][12] = value


#One training loop looks like:
