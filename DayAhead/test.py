import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from os import path
import time
#import plotly.graph_objects as go

from battery_env_dam import BatteryEnv
from classes import DQN
from classes import ReplayMemory

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.autograd import Variable

data_file = 'RTMPrices2019_summerweek.csv'
dam_file = 'DAMPower2019_0k_summer.csv'
model_file = "dam_current_fix_deg_6.pt"
filename = "test_{}_{}.csv".format(model_file[:-3], data_file[:-4])

df = pd.read_csv('./data/'+data_file)
dam_df = pd.read_csv('./data/'+dam_file)
#Battery = BatteryEnv(df)
Battery = BatteryEnv(df=df, dam_df=dam_df)
Battery.reset()
Battery.SOC = 0

DQN_current = torch.load(model_file)

line, = plt.plot([],[])

plt.xlim(0, len(df))
plt.ylim(-100, 12000)

data_tracker = []

def update_line(line, x, y):
    line.set_xdata(np.append(line.get_xdata(), x))
    line.set_ydata(np.append(line.get_ydata(), y))
    plt.draw()

iteration = 0
done = False 

# start = time.clock()
while done == False:
    done, s_t = Battery._next_obs()
    s_t = Variable(s_t) #1 x state_size
    a_t = DQN_current.select_action(s_t).item()
    current_price = s_t[0][1].float().numpy()
    s_next, r_t, done = Battery.take_step(a_t, current_price)
    (profit, soc, socmax, power, rtmprice,dampower,damprice) = Battery.render()
    data_tracker.append([profit, soc, socmax, power, rtmprice, dampower, damprice])
    update_line(line,[iteration],[profit])
    plt.pause(0.05)
    iteration += 1
    # end = time.clock()

# delta_time = end - start

# print("Start: ", start)
# print("End: ", end)
# print("Time delta: ", delta_time)
plt.show()

df = pd.DataFrame(data=data_tracker,columns=["profit", "soc", "socmax","power", "rtmprice","dampower", "damprice"])
df.to_csv(filename, index=False)
print("done")