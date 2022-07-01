import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class BetterBot(nn.Module):
  def __init__(self, act_size):
    super(BetterBot, self).__init__()
    self.act_size = act_size
    self.emb_dice = nn.Embedding(15, 128)
    self.emb_star = nn.Embedding(15, 128)
    self.emb_btn = nn.Embedding(6, 128)

    self.linear1 = nn.Linear(128, 768)
    self.linear2 = nn.Linear(768, 768)
    self.linear3 = nn.Linear(768, 768)
    self.output = nn.Linear(768, act_size)

  def forward(self, board, star, btn):
    return

class Trainer():
  def __init__(self, model, lr):
    self.model = model
    self.lr = lr
    self.optim = optim.Adam(model.parameters(), lr=self.lr)
    self.loss_fn = nn.CrossEntropyLoss()

  def train_step(self, state, action, reward, next_state, end_game):
    state = torch.tensor(state, dtype=torch.long)
    action = torch.tensor(action, dtype=torch.long)
    next_state = torch.tensor(next_state, dtype=torch.long)
    reward = torch.tensor(reward, dtype=torch.float)
    action = None
    return