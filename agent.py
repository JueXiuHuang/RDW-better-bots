import torch
import numpy as np
import random
from collections import deque

from game import *
from enums import *
from model import *
from utils import *

class Agent():
  def __init__(self, max_memory=10000, lr=3e-4):
    self.n_game = 0
    self.target_update = 10
    self.steps_times = 0
    self.batch_size = 32
    self.exploration_stage = 5000
    self.replay_memory = deque(maxlen=max_memory) # will automatically popleft when reach max_memory
    self.model = BetterBot()
    self.trainer = Trainer(self.model, lr)

  def get_state(self):
    dice_lvl = Board.dice_lvl[1:]
    dice_type = []
    dice_star = []
    can_summon_lvl = [int(Board.SP > Board.summon_cost)] # can_summon, can_upg1, can_upg2 ...

    for lvl in dice_lvl:
      can_summon_lvl.append(int(Board.SP > Board.dice_lvl_cost[lvl-1]))
    
    for dice in Board.dice_list:
      dice_type.append(dice.dice_type)
      dice_star.append(dice.dice_star)

    dice_type = np.array(dice_type, dtype=np.int32)
    dice_star = np.array(dice_star, dtype=np.int32)
    can_summon_lvl = np.array(can_summon_lvl, dtype=np.int32)
    
    return {'dice_type':dice_type, 'dice_star':dice_star, 'summon_lvl':can_summon_lvl}

  def remember(self, state, action, reward, next_state, game_over):
    self.replay_memory.append((state, action, reward, next_state, game_over))

  def update_model(self):
    if len(self.replay_memory) < self.batch_size:
      return
    sample = random.sample(self.replay_memory, self.batch_size)
    states, actions, rewards, next_states, end_games = zip(*sample)
    states = self.repack_states(states)
    next_states = self.repack_states(next_states)
    self.trainer.optimize_model(states, actions, rewards, next_states, end_games)
    return

  def repack_states(self, states):
    sample = states[0]
    repacked = {}
    for key in sample:
      repacked[key] = []
    
    for state in states:
      for k, v in state.items():
        repacked[k].append(v)

    return repacked

  def get_action(self):
    # pass state into model, and return an action id
    threshold = max(1 - self.steps_times / self.exploration_stage, 0.1)
    from_random = threshold > np.random.random_sample()
    if from_random:
      action = np.random.randint(0, len(Action))
    else:
      state = self.get_state()
      prediction = self.model(state)
      action = torch.argmax(prediction).item()

    return action

def train():
  game = Game()
  agent = Agent()
  scores = []

  # control how many round
  while True:
    game.reset()
    agent.n_game += 1

    # In game loop
    while True:
      state = agent.get_state()
      action = agent.get_action()
      game_over, reward = game.play_step(action)
      next_state = agent.get_state()

      agent.remember(state, action, reward, next_state, game_over)

      agent.update_model()

      if game_over:
        scores.append(game.cal_score())
        plot(scores)
        break
    
    if agent.n_game % agent.target_update == 0:
      agent.trainer.target_model.load_state_dict(agent.model.state_dict())