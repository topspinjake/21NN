# Jake Williams - Blackjack environment

import torch
import random

def rank(r):
    """Takes a card between 0 and 51 and returns the Blackjack value of the card.
    Aces are assumed to be eleven for now. Card 0 is the 2 of C, 51 is A of S."""
    v = r%13
    if v < 9:
        return v + 2
    elif v == 12:
        return 11
    else:
        return 10

class Blackjack():
    
    def __init__(self):
        self.cards = torch.zeros(52)
        self.dealer = torch.zeros(11)
        self.player = torch.zeros(11)
        self.done = False
        self.initDeal()

    def reset(self):
        self.cards = torch.zeros(52)
        self.dealer = torch.zeros(11)
        self.player = torch.zeros(11)
        self.done = False
        self.initDeal()

    def state(self):
        """State is a (12) tensor that has the player's cards and the dealer's up card."""
        return torch.cat((self.player, torch.index_select(self.dealer,0,torch.tensor([0]))), 0)

    def getCard(self):
        r = random.randint(0, 51)
        while self.cards[r] == 1:
            r = random.randint(0, 51)
        self.cards[r] = 1
        return rank(r)

    def initDeal(self):
        self.dealer[0] = self.getCard()
        self.dealer[1] = self.getCard()
        self.player[0] = self.getCard()
        self.player[1] = self.getCard()
        if sum(self.player) == 21 or sum(self.dealer) == 21:
            #If either player is deal blackjack, the game is over
            self.reset()

    def step(self, a):
        """Takes an action, a - either 1 for hit or 0 for stay, 
        and returns (state, reward) pair. Reward is always 0 when the game is still going.
        Reward is 1 when winning, 0 for tie, and -1 when losing."""
        if self.done:
            print("Game is over")
        elif a == 1: #hitting
            newCard = self.getCard()
            for i in range(2, 11):
                if self.player[i] == 0:
                    self.player[i] = newCard
                    break
            if sum(self.player) > 21:
                self.done = True
                return (self.state(), -1)
            elif sum(self.player) == 21:
                self.done = True
                dealer = sum(self.dealer)
                while dealer < 17:#ignoring soft 17 for now
                    newCard = self.getCard()
                    for i in range(2, 11):
                        if self.dealer[i] == 0:
                            self.dealer[i] = newCard
                            break
                    dealer = sum(self.dealer)
                if dealer == 21:
                    return (self.state(), 0)
                else:
                    return (self.state(), 1)
            else:
                return (self.state(), 0)
        else: #staying
            self.done = True
            player = sum(self.player)
            dealer = sum(self.dealer)
            while dealer < 17:#ignoring soft 17 for now
                newCard = self.getCard()
                for i in range(2, 11):
                    if self.dealer[i] == 0:
                        self.dealer[i] = newCard
                        break
                dealer = sum(self.dealer)
            if dealer > 21 or player > dealer:
                return (self.state(), 1)
            elif dealer == player:
                return (self.state(), 0)
            else:
                return (self.state(), -1)


