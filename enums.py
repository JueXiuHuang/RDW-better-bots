from enum import Enum, auto

class DiceType(Enum):
  Blank = 0
  Growth = 1
  Joker = 2
  Golem = 3
  Typhoon = 4
  Wind = 5
  Sacrifice = 6

class Reward(Enum):
  merge_success = 25
  summon_success = 25
  growth_success = 25
  upg_success = 1
  merge_fail = -5
  summon_fail = -5
  upg_fail = -5

class Action(Enum):
  Pass = 0
  Summon = auto()
  Upg_1 = auto()
  Upg_2 = auto()
  Upg_3 = auto()
  Upg_4 = auto()
  Upg_5 = auto()
  Merge_0_1 = auto()
  Merge_0_2 = auto()
  Merge_0_3 = auto()
  Merge_0_4 = auto()
  Merge_0_5 = auto()
  Merge_0_6 = auto()
  Merge_0_7 = auto()
  Merge_0_8 = auto()
  Merge_0_9 = auto()
  Merge_0_10 = auto()
  Merge_0_11 = auto()
  Merge_0_12 = auto()
  Merge_0_13 = auto()
  Merge_0_14 = auto()
  Merge_1_2 = auto()
  Merge_1_3 = auto()
  Merge_1_4 = auto()
  Merge_1_5 = auto()
  Merge_1_6 = auto()
  Merge_1_7 = auto()
  Merge_1_8 = auto()
  Merge_1_9 = auto()
  Merge_1_10 = auto()
  Merge_1_11 = auto()
  Merge_1_12 = auto()
  Merge_1_13 = auto()
  Merge_1_14 = auto()
  Merge_2_3 = auto()
  Merge_2_4 = auto()
  Merge_2_5 = auto()
  Merge_2_6 = auto()
  Merge_2_7 = auto()
  Merge_2_8 = auto()
  Merge_2_9 = auto()
  Merge_2_10 = auto()
  Merge_2_11 = auto()
  Merge_2_12 = auto()
  Merge_2_13 = auto()
  Merge_2_14 = auto()
  Merge_3_4 = auto()
  Merge_3_5 = auto()
  Merge_3_6 = auto()
  Merge_3_7 = auto()
  Merge_3_8 = auto()
  Merge_3_9 = auto()
  Merge_3_10 = auto()
  Merge_3_11 = auto()
  Merge_3_12 = auto()
  Merge_3_13 = auto()
  Merge_3_14 = auto()
  Merge_4_5 = auto()
  Merge_4_6 = auto()
  Merge_4_7 = auto()
  Merge_4_8 = auto()
  Merge_4_9 = auto()
  Merge_4_10 = auto()
  Merge_4_11 = auto()
  Merge_4_12 = auto()
  Merge_4_13 = auto()
  Merge_4_14 = auto()
  Merge_5_6 = auto()
  Merge_5_7 = auto()
  Merge_5_8 = auto()
  Merge_5_9 = auto()
  Merge_5_10 = auto()
  Merge_5_11 = auto()
  Merge_5_12 = auto()
  Merge_5_13 = auto()
  Merge_5_14 = auto()
  Merge_6_7 = auto()
  Merge_6_8 = auto()
  Merge_6_9 = auto()
  Merge_6_10 = auto()
  Merge_6_11 = auto()
  Merge_6_12 = auto()
  Merge_6_13 = auto()
  Merge_6_14 = auto()
  Merge_7_8 = auto()
  Merge_7_9 = auto()
  Merge_7_10 = auto()
  Merge_7_11 = auto()
  Merge_7_12 = auto()
  Merge_7_13 = auto()
  Merge_7_14 = auto()
  Merge_8_9 = auto()
  Merge_8_10 = auto()
  Merge_8_11 = auto()
  Merge_8_12 = auto()
  Merge_8_13 = auto()
  Merge_8_14 = auto()
  Merge_9_10 = auto()
  Merge_9_11 = auto()
  Merge_9_12 = auto()
  Merge_9_13 = auto()
  Merge_9_14 = auto()
  Merge_10_11 = auto()
  Merge_10_12 = auto()
  Merge_10_13 = auto()
  Merge_10_14 = auto()
  Merge_11_12 = auto()
  Merge_11_13 = auto()
  Merge_11_14 = auto()
  Merge_12_13 = auto()
  Merge_12_14 = auto()
  Merge_13_14 = auto()