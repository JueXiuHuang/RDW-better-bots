import torch
import numpy as np

from game import *
from enums import *

def train():
  game = Game()
  while True:
    action = input('choose an action: from 0 to 111:')
    game_over, reward = game.play_step(int(action))
    print('game over:', game_over)
    print('reward:', reward)
    print('='*10)