import torch
import numpy as np
import random
from collections import deque

from game import *
from enums import *
from model import *

class Agent():
  def __init__(max_memory=10000, lr=3e-4):
    self.n_game = 0
    self.batch_size = 32
    self.exploration_stage = 5000
    self.replay_memory = deque(maxlen=max_memory) # will automatically popleft when reach max_memory
    self.model = BetterBot(len(Action))
    self.trainer = Trainer(self.model, lr)

  def get_state(self):
    dice_lvl = Board.dice_lvl[1:]
    dice_type = []
    dice_star = []
    can_summon_lvl = [int(Board.summon_cost > Board.SP)] # can_summon, can_upg1, can_upg2 ...

    for lvl in dice_lvl:
      can_summon_lvl.append(int(Board.dice_lvl_cost[lvl-1] > Board.SP))
    
    for dice in Board.dice_list:
      dice_type.append(dice.dice_type)
      dice_star.append(dice.dice_star)

    dice_type = np.array(dice_type, dtype=np.int32)
    dice_star = np.array(dice_star, dtype=np.int32)
    can_summon_lvl = np.array(can_summon_lvl, dtype=np.int32)
    
    return {'dice_type':dice_type, 'dice_star':dice_star, 'summon_lvl':can_summon_lvl}

  def remember(self, state, action, reward):
    self.replay_memory.append((state, action, reward))

  def train_long_memory(self):
    sample = random.sample(self.replay_memory, self.batch_size)
    states, actions, rewards, next_states, end_games = zip(*sample)
    self.trainer.train_step(states, actions, rewards, next_states, end_games)
    return

  def train_short_memory(self):
    self.trainer.train_step([state], [action], [reward], [next_state], [end_game])
    return

  def get_action(self):
    # pass state into model, and return an action id
    if self.n_game < self.exploration_stage:
      action = np.random.randint(0, len(Action))
    else:
      state = self.get_state()
      prediction = self.model(state)
      action = torch.argmax(prediction).item()

    return action

def train():
  game = Game()
  while True:
    old_state = game.get_state()
    
    # action = input('choose an action: from 0 to 111:')
    # game_over, reward = game.play_step(int(action))
    # print('game over:', game_over)
    # print('reward:', reward)
    # print('='*10)