import pygame as pyg
import numpy as np
import os

from board import *
from button import *
from displayer import *
from enums import *

pyg.init()
font = pyg.font.Font(None, 30)

class Game():
  def __init__(self):
    self.surface = pyg.display.set_mode((600, 400))
    pyg.display.set_caption('Random Dice Wars Emulator')
    icon = pyg.image.load('img/AppIcon.png')
    pyg.display.set_icon(icon)

    Dice.set_font_and_surface(font, self.surface)
    Button.set_font_and_surface(font, self.surface)
    Displayer.set_font_and_surface(font, self.surface)
    self.gridsize = Board.gridsize
    self.dicesize = Board.dicesize
    self.board = Board()
    self.init_all()
    self.tick = 0
    self.tick_sec_ratio = 10
    self.tick_wave_ratio = 13
    self.update_ui()

  def reset(self):
    self.tick = 0
    Board.reset_game()
  
  def game_ended(self):
    return Board.wave > 30

  def play_step(self, action):
    # action should be an int which maps to the
    #   value of Action class im myEnum.py

    self.tick += 1
    if self.tick % self.tick_wave_ratio == 0:
      Board.wave += 1
      idx = np.sum([Board.wave >= wave for wave in Board.SP_level_wave])
      Board.SP += Board.SP_list[idx]
    
    # Step 1, collect user input
    for event in pyg.event.get():
      if event.type == pyg.QUIT:
        pyg.quit()
        quit()

    reward = 0
    game_over = False

    # Step 2, execute action from Agent
    print(Action(action).name)
    if action == Action.Pass.value:
      # Agent choose to do nothing
      reward = 0
    elif action == Action.Summon.value:
      # Agent choose to summon a new dice
      success = Board.summon_dice()
      if success:
        reward += Reward.summon_success.value
      else:
        reward += Reward.summon_fail.value
    elif action in range(Action.Upg_1.value, Action.Upg_5.value+1):
      # Agent choose to upgrade dice
      dice_type = Action(action).name
      dice_type = int(dice_type.split('_')[1])
      success = Board.lvlup_dice(dice_type)
      if success:
        reward += Reward.upg_success.value
      else:
        reward += Reward.upg_fail.value
    elif action in range(Action.Merge_0_1.value, Action.Merge_13_14.value+1):
      # Agent choose to merge dice
      action_str = Action(action).name
      _, loc_a, loc_b = action_str.split('_')
      loc_a = int(loc_a)
      loc_b = int(loc_b)
      success = Board.merge_dice(loc_a, loc_b)
      if success:
        reward += Reward.merge_success.value
      else:
        reward += Reward.merge_fail.value

    # Step 3, check if game over
    
    if Board.wave > 30:
      game_over = True
      reward += self.cal_reward()
      return game_over, reward

    # Step 4, update UI
    self.update_ui()

    reward += self.cal_reward()

    # Step 5, return reward for this step
    return game_over, reward

  def cal_reward(self):
    return 0
  
  def update_ui(self):
    self.surface.fill(pyg.Color("black"))
    self.update()
    self.draw()
    pyg.display.flip()

  def update(self):
    Board.update()
    for dp in self.displayer_list:
      dp.update()
    for btn in self.BtnList:
      btn.update()

  def draw(self):
    Board.draw()
    for dp in self.displayer_list:
      dp.draw()
    for btn in self.BtnList:
      btn.draw()

  def init_all(self):
    self.init_btn()
    self.init_displayer()

  def init_btn(self):
    self.BtnList = []
    
    # Create Summon button
    path = 'img/Summon.png'
    image = pyg.image.load(path)
    image = pyg.transform.scale(image, (self.dicesize-5, self.dicesize-5))
    x = self.gridsize*5.2 + 50
    y = self.gridsize*0.5 + 50
    h = self.gridsize*2.3
    btn = SummonBtn(image, x, y, self.dicesize+5, h)
    self.BtnList.append(btn)
    
    # Create Reset button
    path = 'img/Reset.png'
    image = pyg.image.load(path)
    image = pyg.transform.scale(image, (30, 30))
    x = self.gridsize*5.2 + 50
    y = self.gridsize*3.2 + 50
    h = self.gridsize*0.6
    btn = ResetBtn(image, x, y, self.dicesize+5, h)
    self.BtnList.append(btn)

    for i in range(1, 6):
      x = self.gridsize*(i-1) + 50
      y = self.gridsize*3.4 + 50
      h = self.gridsize*0.6
      btn = LvlUpBtn(i, x, y, self.dicesize+5, h)
      self.BtnList.append(btn)
  
  def init_displayer(self):
    self.displayer_list  = []

    # Create Wave display button
    x = self.gridsize*1.4 + 50
    y = self.gridsize*0
    h = self.gridsize*0.4
    wd = WaveDisplayer(x, y, self.dicesize*2.4, h)
    self.displayer_list.append(wd)

    # Create SP display button
    x = self.gridsize*5.2 + 50
    y = self.gridsize*0
    h = self.gridsize*0.4
    sd = SPDisplayer(x, y, self.dicesize*1, h)
    self.displayer_list.append(sd)