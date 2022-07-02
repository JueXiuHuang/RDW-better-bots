import torch
import numpy as np
import random
def set_seed(seed=777):
    np.random.seed(seed)                       
    torch.manual_seed(seed)                    
    torch.cuda.manual_seed(seed)               
    torch.cuda.manual_seed_all(seed)           
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    
set_seed()

from agent import *

if __name__ == '__main__':
  train()