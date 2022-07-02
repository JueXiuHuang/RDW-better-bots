import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

from enums import *

class attn_layer(nn.Module):
  def __init__(self, dim=512):
    super(attn_layer, self).__init__()
    self.attn = nn.MultiheadAttention(dim, num_heads=2, dropout=0.2, batch_first=True)
    self.linear = nn.Linear(dim, dim)

  def forward(self, x):
    x_ = self.attn(x, x, x, need_weights=False)[0]
    x = x + x_
    x_ = self.linear(x)
    x_ = F.relu(x_)
    x = x + x_
    return x

class BetterBot(nn.Module):
  def __init__(self, act_size=len(Action)):
    super(BetterBot, self).__init__()
    self.act_size = act_size
    self.n_layers = 2
    self.dim = 8
    self.emb_dice = nn.Embedding(15, self.dim)
    self.emb_star = nn.Embedding(15, self.dim)
    self.emb_btns = nn.Embedding(2, self.dim)

    self.layers = [attn_layer(self.dim) for _ in range(self.n_layers)]
    self.linear1 = nn.Linear(self.dim, self.dim)
    self.linear2 = nn.Linear(self.dim, self.dim)
    self.linear3 = nn.Linear(self.dim, self.dim)
    self.output = nn.Linear(self.dim, act_size)

  def forward(self, dice_type, dice_star, summon_lvl):
    dt = self.emb_dice(dice_type) # (bz, 15, emb_dim)
    ds = self.emb_star(dice_star) # (bz, 15, emb_dim)
    db = self.emb_btns(summon_lvl) # (bz, 6, emb_dim)
    x = torch.cat((dt, ds, db), dim=1) # (bz, 36, emb_dim)

    for layer in self.layers:
      x = layer(x)
    x = torch.mean(x, dim=1)

    # x = self.linear1(x) # (bz, 36, hid_dim)
    # x = F.relu(x)
    # x = self.linear2(x) # (bz, 36, hid_dim)
    # x = F.relu(x)
    # x = torch.mean(x, dim=1) # (bz, hid_dim)
    # x = self.linear3(x)
    # x = F.relu(x)
    x = self.output(x) # (bz, act_size)
    return x

class Trainer():
  def __init__(self, model, lr):
    self.model = model
    self.target_model = BetterBot()
    self.target_model.load_state_dict(self.model.state_dict())
    self.target_model.eval()

    self.gamma = 0.7
    self.lr = lr
    self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
    self.loss_fn = nn.SmoothL1Loss()

  def optimize_model(self, states, actions, rewards, next_states, end_games):
    self.model.train()
    actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
    end_games = torch.tensor(end_games, dtype=torch.bool)
    rewards = torch.tensor(rewards, dtype=torch.float)
    # elements for current state
    diceT = torch.tensor(states['dice_type'], dtype=torch.long)
    diceS = torch.tensor(states['dice_star'], dtype=torch.long)
    sumLvl = torch.tensor(states['summon_lvl'], dtype=torch.long)
    # elements for next state
    ns_diceT = torch.tensor(next_states['dice_type'], dtype=torch.long)
    ns_diceS = torch.tensor(next_states['dice_star'], dtype=torch.long)
    ns_sumLvl = torch.tensor(next_states['summon_lvl'], dtype=torch.long)

    # add description
    state_action_values = self.model(diceT, diceS, sumLvl).gather(1, actions)

    # add description
    next_state_values = self.target_model(ns_diceT, ns_diceS, ns_sumLvl).max(1)[0].detach()
    # if current state is final state, expected state value should 0
    next_state_values[end_games] = 0

    expected_state_action_values = (next_state_values * self.gamma) + rewards

    loss = self.loss_fn(state_action_values, expected_state_action_values.unsqueeze(1))

    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    return