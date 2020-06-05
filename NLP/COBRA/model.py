import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import math

class Dense_Net(nn.Module):
    def __init__(self, input_dim, out_dim):
        super(Dense_Net, self).__init__()