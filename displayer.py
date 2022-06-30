import pygame as pyg

from board import *

class Displayer():
  font = None
  surface = None

  @classmethod
  def set_font_and_surface(cls, font, surface):
    cls.font = font
    cls.surface = surface

  @classmethod
  def set_surface(cls, surface):
    cls.surface = surface

  def __init__(self, x, y, w, h):
    self.x = x
    self.y = y
    self.w = w
    self.h = h
    self.rect = pyg.Rect(x, y, w, h)
  
  def draw(self):
    return

  def update(self):
    return

class WaveDisplayer(Displayer):
  def __init__(self, x, y, w, h):
    super().__init__(x, y, w, h)
    self.wave = str(Board.wave)

  def draw(self):
    text_surf = Displayer.font.render(' '.join(['Wave', self.wave]), True, '#9966ff')
    text_rect = text_surf.get_rect(center=self.rect.center)
    Displayer.surface.blit(text_surf, text_rect)
    return
  
  def update(self):
    self.wave = str(Board.wave)
    return

class SPDisplayer(Displayer):
  def __init__(self, x, y, w, h):
    super().__init__(x, y, w, h)
    self.SP = str(Board.SP)

  def draw(self):
    text_surf = Displayer.font.render(self.SP, True, '#66ff66')
    text_rect = text_surf.get_rect(center=self.rect.center)
    Displayer.surface.blit(text_surf, text_rect)
    return

  def update(self):
    self.SP = str(Board.SP)
    return