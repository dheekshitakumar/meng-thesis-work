import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from torch.nn import init, Parameter
from torch.autograd import Variable


Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, noisy, nodes, input_size=25, output_size=24):
        # super(DQN, self).__init__()
        # self.relu = nn.ReLU()
        # self.sig = nn.Sigmoid()
        # if noisy == True:
        #     self.fc1 = NoisyLinear(input_size, nodes)
        #     self.fc2 = NoisyLinear(nodes, nodes)
        #     self.fc3 = NoisyLinear(nodes, output_size)
        # else:
        #     self.fc1 = nn.Linear(input_size, nodes)
        #     self.fc2 = nn.Linear(nodes, nodes)
        #     self.fc3 = nn.Linear(nodes, output_size)
        super(DQN, self).__init__()
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()
        if noisy == True:
            self.fc1 = NoisyLinear(input_size, nodes)
            self.fc_value = NoisyLinear(nodes, 16)
            self.fc_adv = NoisyLinear(nodes, 16)
            self.value = NoisyLinear(16, 1)
            self.adv = NoisyLinear(16, output_size)
        else:
            self.fc1 = nn.Linear(input_size, nodes)
            self.fc_value = nn.Linear(nodes, 16)
            self.fc_adv = nn.Linear(nodes, 16)
            self.value = nn.Linear(16, 1)
            self.adv = nn.Linear(16, output_size)

    # def forward(self, state):
    #     out = self.relu(self.fc1(state))
    #     out = self.relu(self.fc2(out))
    #     out = self.sig(self.fc3(out))
    #     return out

    def sample_noise(self):
        self.fc1.sample_noise()
        # self.fc2.sample_noise()
        # self.fc3.sample_noise()        

    def forward(self, state):
        y = self.relu(self.fc1(state))
        value = self.relu(self.fc_value(y))
        adv = self.relu(self.fc_adv(y))

        value = self.tanh(self.value(value))
        adv = self.sig(self.adv(adv))

        # print('adv: ', adv)
        # print('value: ', value)

        advAverage = torch.mean(adv, dim=1, keepdim=True)
        Q = value + adv - advAverage

        #print('Q: ', Q)
        return Q

# Noisy linear layer with independent Gaussian noise
class NoisyLinear(nn.Linear):
  def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
    super(NoisyLinear, self).__init__(in_features, out_features, bias=True)
    # µ^w and µ^b reuse self.weight and self.bias
    self.sigma_init = sigma_init
    self.sigma_weight = Parameter(torch.Tensor(out_features, in_features))  # σ^w
    self.sigma_bias = Parameter(torch.Tensor(out_features))  # σ^b
    self.register_buffer('epsilon_weight', torch.zeros(out_features, in_features))
    self.register_buffer('epsilon_bias', torch.zeros(out_features))
    self.reset_parameters()

  def reset_parameters(self):
    if hasattr(self, 'sigma_weight'):  # Only init after all params added (otherwise super().__init__() fails)
      init.uniform(self.weight, -math.sqrt(3 / self.in_features), math.sqrt(3 / self.in_features))
      init.uniform(self.bias, -math.sqrt(3 / self.in_features), math.sqrt(3 / self.in_features))
      init.constant(self.sigma_weight, self.sigma_init)
      init.constant(self.sigma_bias, self.sigma_init)

  def forward(self, input):
    return F.linear(input, self.weight + self.sigma_weight * Variable(self.epsilon_weight), self.bias + self.sigma_bias * Variable(self.epsilon_bias))

  def sample_noise(self):
    self.epsilon_weight = torch.randn(self.out_features, self.in_features)
    self.epsilon_bias = torch.randn(self.out_features)

  def remove_noise(self):
    self.epsilon_weight = torch.zeros(self.out_features, self.in_features)
    self.epsilon_bias = torch.zeros(self.out_features)
