import numpy as np
import os

from dice import *
from enums import *

class Board():
  dice_list = []
  imgs = []
  gridsize = 80
  dicesize = 70
  board_h = 3
  board_w = 5
  SP = 50
  summon_cost = 5
  wave = 0

  dice_lvl = [0, 1, 1, 1, 1, 1]
  dice_lvl_cost = [150, 300, 500, 800, 1200, float('inf')]
  SP_list = [160, 200, 280, 400, 560, 760]
  SP_level_wave = [3, 7, 12, 18, 23]
  Arcade_level_wave = [4, 8, 14, 22, 30]

  @staticmethod
  def reset_game():
    Board.dice_list = []
    Board.init_board()
    Board.SP = 50
    Board.summon_cost = 5
    Board.wave = 0
    Board.dice_lvl = [0, 1, 1, 1, 1, 1]
  
  @staticmethod
  def get_board_info():
    return

  @staticmethod
  def generate_random_dice():
    dice_type = np.random.randint(1, 6)
    image = Board.imgs[dice_type]

    if dice_type == DiceType.Growth.value:
      new_dice = GrowthDice(image, dice_type, 0, 0, Board.dicesize, Board.dicesize, (dice_type != 0))
    elif dice_type == DiceType.Joker.value:
      new_dice = JokerDice(image, dice_type, 0, 0, Board.dicesize, Board.dicesize, (dice_type != 0))
    elif dice_type == DiceType.Golem.value:
      new_dice = NormalDice(image, dice_type, 0, 0, Board.dicesize, Board.dicesize, (dice_type != 0))
    elif dice_type == DiceType.Typhoon.value:
      new_dice = NormalDice(image, dice_type, 0, 0, Board.dicesize, Board.dicesize, (dice_type != 0))
    elif dice_type == DiceType.Wind.value:
      new_dice = NormalDice(image, dice_type, 0, 0, Board.dicesize, Board.dicesize, (dice_type != 0))
    
    return new_dice

  @staticmethod
  def generate_empty_dice():
    dice_type = DiceType.Blank.value
    image = Board.imgs[dice_type]
    new_dice = NormalDice(image, dice_type, 0, 0, Board.dicesize, Board.dicesize, (dice_type != 0))
    # new_dice = NormalDice(image, dice_type, 0, 0, Board.dicesize, Board.dicesize, True)
    return new_dice

  @staticmethod
  def summon_dice():
    # no slot for summon new dice
    if np.sum([dice.dice_type == DiceType.Blank.value for dice in Board.dice_list]) == 0:
      # print('No more space')
      return False
    
    if Board.SP < Board.summon_cost:
      # print('Not enough SP')
      return False
    
    Board.SP -= Board.summon_cost
    Board.summon_cost += 5
    new_dice = Board.generate_random_dice()

    candidate = np.where([dice.dice_type == DiceType.Blank.value for dice in Board.dice_list])[0]
    loc = np.random.choice(candidate)

    new_dice = Board.dice_list[loc].copy_info(new_dice, need_cp_star=True)
    new_dice.dice_star += 1
    Board.dice_list[loc] = new_dice

    return True

  @staticmethod
  def lvlup_dice(dice_type):
    if Board.dice_lvl[dice_type] > 5:
      # print('Already max lvl...')
      return False
    lvl = Board.dice_lvl[dice_type] - 1
    if Board.dice_lvl_cost[lvl] > Board.SP:
      # print('Not enough SP...')
      return False
    # print('Lvl up dice...')
    Board.SP -= Board.dice_lvl_cost[lvl]
    Board.dice_lvl[dice_type] += 1
    return True
    
  def __init__(self):
    self.initialization()

  def initialization(self):
    Board.load_img_from_disk()
    Board.init_board()

  @staticmethod
  def load_img_from_disk():
    img_names = ['Blank.png', 'Growth.png', 'Joker.png',
                  'Golem.png', 'Typhoon.png', 'Wind.png']
    
    for img in img_names:
      path = os.path.join('./img', img)
      _img = pyg.image.load(path)
      _img = pyg.transform.scale(_img, (Board.dicesize, Board.dicesize))
      Board.imgs.append(_img)

  @staticmethod
  def merge_check(pos):
    # search the dice which can be merged
    dice_a_idx = None
    dice_b_idx = None
    
    for idx in range(len(Board.dice_list)):
      dice = Board.dice_list[idx]
      if dice.is_drag:
        dice_a_idx = idx
        dice.is_drag = False
      elif dice.rect.collidepoint(pos):
        dice_b_idx = idx
      dice.reset_dice_loc()
    
    if dice_a_idx != None and dice_b_idx != None:
      # merge dice
      Board.merge_dice(dice_a_idx, dice_b_idx)
  
  @staticmethod
  def check_skill():
    skill_reward = 0
    for idx in range(len(Board.dice_list)):
      can_activate = Board.dice_list[idx].check_skill_tick()
      if can_activate:
        new_dice, reward = Board.dice_list[idx].skill(Board.generate_random_dice)
        if new_dice:
          Board.dice_list[idx] = new_dice
        skill_reward += reward
    return skill_reward

  @staticmethod
  def merge_dice(loc_a, loc_b):
    new_dice_a = Board.generate_empty_dice()
    new_dice_b = Board.generate_random_dice()
    success, new_dice_a = Board.dice_list[loc_a].after_merge(new_dice_a, Board.dice_list[loc_b])
    success_, new_dice_b = Board.dice_list[loc_a].merge(new_dice_b, Board.dice_list[loc_b])
    Board.dice_list[loc_a] = new_dice_a
    Board.dice_list[loc_b] = new_dice_b
    return success and success_

  @staticmethod
  def check_dice_select(pos):
    for dice in Board.dice_list:
      dice.is_dice_select(pos)
  
  @staticmethod
  def update(pos=None):
    for dice in Board.dice_list:
      if dice.is_drag and pos:
        dice.update_loc(pos)
      dice_type = dice.dice_type
      dice.draggable = dice_type != DiceType.Blank.value
      dice.set_content(Board.imgs[dice_type])

  @staticmethod
  def draw():
    for dice in Board.dice_list:
      dice.draw()

  @staticmethod
  def init_board():
    for y in range(Board.board_h):
      for x in range(Board.board_w):
        posx = x*Board.gridsize + 50
        posy = y*Board.gridsize + 50
        dice = Board.generate_empty_dice()
        dice.original_x = posx
        dice.original_y = posy
        dice.reset_dice_loc()

        Board.dice_list.append(dice)