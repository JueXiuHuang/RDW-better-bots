import pygame as pyg

from enums import *

class Dice():
  # class attribute
  surface = None
  font = None
  
  @classmethod
  def set_font_and_surface(cls, font, surface):
    cls.font = font
    cls.surface = surface
  
  def __init__(self, image, dice_type, posx, posy, width, height, draggable):
    self.original_x = posx
    self.original_y = posy
    self.width = width
    self.height = height
    self.dice_type = dice_type
    self.dice_star = 0
    self.is_drag = False
    self.draggable = draggable
    self.rect = pyg.Rect(posx, posy, width, height)

    # 10 tick equals to 1 second
    self.skill_tick = 50
    self.skill_count_down = self.skill_tick
    
    self.set_content(image)

  def copy(self):
    class_type = type(self)
    new_dice = class_type(self.content, self.dice_type,
                          self.original_x, self.original_y,
                          self.width, self.height, self.draggable)
    
    return new_dice
  
  def draw_content(self):
    image_rect = self.content.get_rect(center=self.rect.center)
    Dice.surface.blit(self.content, image_rect)
    text_surf = Dice.font.render(str(self.dice_star), True, '#ff0000')
    text_rect = text_surf.get_rect(center=self.rect.center)
    Dice.surface.blit(text_surf, text_rect)
  
  def draw(self):
    self.draw_content()
    pyg.draw.rect(Dice.surface, pyg.Color("white"), self.rect, 1)

  def is_dice_select(self, mouse_pos):
    if self.rect.collidepoint(mouse_pos) and self.draggable:
      self.is_drag = True
    else:
      self.is_drag = False

  def reset_dice_loc(self):
    self.rect.x = self.original_x
    self.rect.y = self.original_y
    self.is_drag = False
  
  def update_loc(self, mouse_pos):
    rel_x, rel_y = mouse_pos
    self.rect.x += rel_x
    self.rect.y += rel_y

  def set_content(self, new_content):
    self.content = new_content

  def check_skill_tick(self):
    self.skill_count_down -= 1
    if self.skill_count_down == 0:
      self.skill_count_down = self.skill_tick
      return True
    else:
      return False
  
  def skill(self, func):
    reward = 0
    return None, reward

  def copy_info(self, dice, need_cp_star=True):
    dice.original_x = self.original_x
    dice.original_y = self.original_y
    dice.rect.x = self.rect.x
    dice.rect.y = self.rect.y
    if need_cp_star:
      dice.dice_star = self.dice_star
    
    return dice
  
  def merge(self, new_dice, dice_b):
    # Dice A + Dice B -> Empty Dice + New Dice
    # This function will return New Dice
    if self.dice_type != dice_b.dice_type or \
      self.dice_star != dice_b.dice_star or \
      self.dice_type == DiceType.Blank.value or \
      self.dice_star > 6:
      return False, dice_b
    
    new_dice = dice_b.copy_info(new_dice, need_cp_star=True)
    new_dice.reset_dice_loc()
    new_dice.dice_star += 1

    return True, new_dice

  def after_merge(self, new_dice, dice_b):
    # Dice A + Dice B -> Empty Dice + New Dice
    # This function will return Empty Dice
    if self.dice_type != dice_b.dice_type or \
      self.dice_star != dice_b.dice_star or \
      self.dice_type == DiceType.Blank.value or \
      self.dice_star > 6:
      self.reset_dice_loc()
      return False, self

    new_dice = self.copy_info(new_dice, need_cp_star=False)
    new_dice.reset_dice_loc()

    return True, new_dice


class NormalDice(Dice):
  def __init__(self, image, dice_type, posx, posy, width, height, draggable):
    super().__init__(image, dice_type, posx, posy, width, height, draggable)


class JokerDice(Dice):
  def __init__(self, image, dice_type, posx, posy, width, height, draggable):
    super().__init__(image, dice_type, posx, posy, width, height, draggable)
  
  def merge(self, new_dice, dice_b):
    # Dice A + Dice B -> Dice B + Dice B
    # This function will return original Dice B

    if self.dice_type == dice_b.dice_type and \
      self.dice_star == dice_b.dice_star and self.dice_star < 7:
      new_dice = dice_b.copy_info(new_dice, need_cp_star=True)
      new_dice.reset_dice_loc()
      new_dice.dice_star += 1
      return True, new_dice
    elif self.dice_star == dice_b.dice_star:
      return True, dice_b
    else:    
      return False, dice_b

  def after_merge(self, new_dice, dice_b):
    # Dice A + Dice B -> Dice B + Dice B
    # This function will return copied Dice B

    if self.dice_type == dice_b.dice_type and \
      self.dice_star == dice_b.dice_star and self.dice_star < 7:
      new_dice = self.copy_info(new_dice, need_cp_star=False)
      new_dice.reset_dice_loc()
      return True, new_dice
    elif self.dice_star == dice_b.dice_star:
      cp_dice_b = dice_b.copy()
      cp_dice_b = self.copy_info(cp_dice_b)
      cp_dice_b.dice_star = dice_b.dice_star
      cp_dice_b.reset_dice_loc()
      return True, cp_dice_b
    else:
      self.reset_dice_loc()
      return False, self

class GrowthDice(Dice):
  def __init__(self, image, dice_type, posx, posy, width, height, draggable):
    super().__init__(image, dice_type, posx, posy, width, height, draggable)
    self.skill_tick = 20
    self.skill_count_down = self.skill_tick

  def skill(self, func):
    if self.dice_star > 6:
      reward = 0
      return self, reward
    reward = Reward.growth_success.value
    new_dice = func()
    new_dice = self.copy_info(new_dice, need_cp_star=True)
    new_dice.dice_star += 1

    return new_dice, reward

class SacrificeDice(Dice):
  def __init__(self, image, dice_type, posx, posy, width, height, draggable):
    super().__init__(image, dice_type, posx, posy, width, height, draggable)