import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from os import path
#import plotly.graph_objects as go

from battery_env_deg_wind import BatteryEnvDegWind
from classes import DQN
from classes import ReplayMemory

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.autograd import Variable



data_file = 'Prices2013_day.csv'
wind_file = 'WindData2013.csv'
model_file = "dqn_current_wind_degpenalty400000_gamma0.999_week.pt"
filename = "test_{}_{}.csv".format(model_file[:-3], data_file[:-4])

df = pd.read_csv('./data/'+data_file)
wind_df = pd.read_csv('./data/'+wind_file)
#Battery = BatteryEnv(df)
Battery = BatteryEnvDegWind(df,wind_df,400000)
Battery.reset()

DQN_current = torch.load(model_file)

line, = plt.plot([],[])

plt.xlim(0, len(df))
plt.ylim(-100, 100000)

data_tracker = []

def update_line(line, x, y):
    line.set_xdata(np.append(line.get_xdata(), x))
    line.set_ydata(np.append(line.get_ydata(), y))
    plt.draw()

iteration = 0
done = False 

while done == False:
    done, s_t = Battery._next_obs()
    s_t = Variable(s_t) #1 x state_size
    a_t = DQN_current.select_action(s_t).item()
    current_price = s_t[0][1].float().numpy()
    current_wind = s_t[0][25].float().numpy()
    s_next, r_t, done = Battery.take_step(a_t, current_price, current_wind)
    (profit, socmax, reward_total, soc, pow, price, wind) = Battery.render()
    data_tracker.append([profit, socmax, reward_total, soc, pow, price, wind])
    update_line(line,[iteration],[profit])
    plt.pause(0.05)
    iteration += 1

plt.show()

df = pd.DataFrame(data=data_tracker,columns=["profit", "SOCmax", "reward_total", "soc", "power", "price","wind"])
df.to_csv(filename, index=False)
print("done")