import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from os import path
#import plotly.graph_objects as go

from battery_env import BatteryEnv
from battery_env_deg import BatteryConstEnvDeg, BatteryDynEnvDeg
from classes import DQN, DQNConv
from classes import ReplayMemory

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.autograd import Variable


num_eps = 5000
C = 100
df_file = '../data/Prices2013_summerweek.csv'
df = pd.read_csv(df_file)
batch_size = 32
gamma = 0.95
#Battery = BatteryEnv(df)
degcost = 40000
Battery = BatteryConstEnvDeg(df=df, dcost=degcost,flag=10)
greedy_max = 0.8
greedy_min = 0.00025
greedy_deg=5
every_epoch = 1000

filename = "summerweek2"
#filename = 'winter_deg{}_gamma{}_consteff_e{}_C{}_gmax{}_gmin{}'.format(degcost,gamma,num_eps,C,greedy_max,greedy_min)
#filename = 'degpenalty{}_workoff'.format(degcost)
model_file = "dqn_current_{}.pt".format(filename)
target_file = "dqn_target_{}.pt".format(filename)
filename = filename+'.csv'

files = pd.read_csv('parameters.csv')['filename'].tolist()

while filename in files:
    answer = input("This file already exists, do you want to overwrite it? [Y/N]:  ")
    while answer not in ["Y", "N"]:
        answer = input("This file already exists, do you want to overwrite it? [Y/N]. Please use 'Y' and 'N' to indicate preference.:  ")
    if answer == "N":
        filename = input("Please input a new filename: ")
        if filename[-4:] != ".csv":
            filename = filename+".csv"

myCsvRow =  "\n{},{},{},{},{},{},{},{},{},{},{}".format(filename,df_file,degcost,gamma,Battery.nu,Battery.flag,greedy_max,greedy_min,greedy_deg,num_eps,C)
with open('parameters.csv','a') as fd:
    fd.write(myCsvRow)

memory = 1000 #how many examples do we want to remember in the past
noisy = True #do we want Noisy Linear Layers
nodes = 16 #how many inner nodes in the DQN

fig, (ax1, ax2) = plt.subplots(2,1)

line1, = ax1.plot([], [], lw=2)
line2, = ax2.plot([], [], lw=2, color='r')

ax1.set_xlim(0,num_eps)
ax1.set_ylim(-1000,2000)
ax2.set_xlim(0,num_eps)
ax2.set_ylim(-5000,2000)

data_tracker = []

def update_line(line, x, y):
    line.set_xdata(np.append(line.get_xdata(), x))
    line.set_ydata(np.append(line.get_ydata(), y))
    plt.draw()

def train():
    #initialize the network and target parameters with their noisy params
    DQN_target = DQN(noisy,nodes)
    DQN_current = DQN(noisy,nodes)
    #DQN_target = torch.load("dqn_target_degpenalty0_gamma0.999_consteff_e5000_C100_gmax0.8_gmin0.01.pt")
    #DQN_current = torch.load("dqn_current_degpenalty0_gamma0.999_consteff_e5000_C100_gmax0.8_gmin0.01.pt")
    optimizer = optim.Adam(DQN_current.parameters(), lr=0.00025)
    # greed_mod = 31
    #initialize the replay memory
    Memory = ReplayMemory(memory)
    t = 0
    greedy_eps = greedy_max

    for e in range(num_eps):
        print("EPISODE: ", e)
        #reset the battery and obtain the state for the battery. Also -- we just started so we're not done yet.
        Battery.reset()
        done = False

        greedy_eps -= greedy_deg*(greedy_max-greedy_min)/num_eps
        greedy_eps = max(greedy_min, greedy_eps)
        print("GREEDY EPS: ", greedy_eps)
        
        while done == False: #for t = 1,....,T
            if noisy:
                DQN_target.sample_noise()
                DQN_current.sample_noise()

            done, s_t = Battery._next_obs()
            s_t = Variable(s_t) #1 x state_size

            if random.random() <= (1-greedy_eps):
                a_t = DQN_current.select_action(s_t).item() #select an action based on what DQN thinks is best. This is a single integer in the range(all_action_size)
            else:
                a_t = random.randint(0,4)

            current_price = s_t[0][1].float().numpy()

            s_next, r_t, done = Battery.take_step(a_t, current_price) #execute action and recieve reward
            Memory.push(s_t,a_t,s_next,r_t) #store (s_t, a_t, s_next, r_t) in ReplayMemory

            if noisy: #sample noise again in the Neural Networks
                DQN_target.sample_noise()
                DQN_current.sample_noise()

            if len(Memory) < batch_size:
                batch = Memory.sample(len(Memory))
            else:
                batch = Memory.sample(batch_size)

            s_batch = torch.cat([getattr(x, 'state') for x in batch], dim=0) #batch_size x state_size
            s_next_batch = torch.cat([getattr(x, 'next_state') for x in batch], dim=0) #batch_size x state_size
            a_batch = torch.Tensor([[getattr(x, 'action')] for x in batch]).long() #batch_size x 1
            r_batch = torch.Tensor([[getattr(x, 'reward')] for x in batch]) #batch_size x 1

            a_prime_batch = DQN_current.select_action(s_next_batch).long().unsqueeze(1) #batch_size x 1

            # print('size', a_batch.size(), 'a_batch: ', a_batch)
            # print('size', s_batch.size(), 's_batch: ', s_batch)
            # print('size', r_batch.size(), 'r_batch: ', r_batch)
            # print('size', a_prime_batch.size(),'a_prime_batch: ', a_prime_batch)
            # print('size', s_next_batch.size, 's_next_batch: ', s_next_batch)

            targets  = DQN_target.forward(s_next_batch) #estimate the target values for all actions. This returns a Q value matrix of size: batch_size x all_action_size
            y_t = r_batch + gamma*torch.gather(targets, 1, a_prime_batch) #batch_size x 1 (i.e. a single Qvalue for each input in the batch)


            Q_current =  DQN_current.forward(s_batch)

            if len(Q_current.size()) > 2:
                Q_current = torch.squeeze(Q_current, 1)

            q_t = torch.gather(Q_current, -1, a_batch)

            loss = F.mse_loss(y_t,q_t)#Do a gradient descent with loss of y_t - (DQN(state,action)^2) and update DQN


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #every C steps also set DQN_target weights to be DQN weights
            if t % C == 0:
                DQN_target.load_state_dict(DQN_current.state_dict())
            t += 1
            Battery.render()
            (profit, socmax, reward_total, soc, pow, price) = Battery.render()
            data_tracker.append([profit, socmax, reward_total, soc, pow, price])
        update_line(line1,[e],[profit])
        update_line(line2,[e],[reward_total])
        plt.pause(0.05)
        if e % every_epoch == 0:
            # torch.save(DQN_current, "epoch{}".format(e)+model_file)
            # torch.save(DQN_target, "epoch{}".format(e)+target_file)
            df = pd.DataFrame(data=data_tracker,columns=["profit", "SOCmax", "reward_total", "soc", "power", "price"])
            df.to_csv(filename, index=False)
    plt.show()
    torch.save(DQN_current, model_file)
    torch.save(DQN_target, target_file)
    print("done")
    df = pd.DataFrame(data=data_tracker,columns=["profit", "SOCmax", "reward_total", "soc", "power", "price"])
    df.to_csv(filename, index=False)
train()
