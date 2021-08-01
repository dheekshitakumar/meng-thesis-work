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

###########################################
#CONSTANTS
###########################################

num_eps = 5000
C = 1000
df = pd.read_csv('./data/Prices2013_week.csv')
winddf = pd.read_csv('./data/WindData2013_week.csv')
batch_size = 32
gamma = 0.999
degcost=400000
Battery = BatteryEnvDegWind(pricedf=df, winddf=winddf, dcost=degcost)
greedy_max = 0.9
greedy_min = 0.01

###########################################
#FILENAME
###########################################
filename = 'wind_degpenalty{}_gamma{}_overpen100_week'.format(degcost,gamma)
model_file = "dqn_current_{}.pt".format(filename)
target_file = "dqn_target_{}.pt".format(filename)
filename = filename+'.csv'

while path.exists(filename):
    answer = input("This file already exists, do you want to overwrite it? [Y/N]:  ")
    while answer not in ["Y", "N"]:
        answer = input("This file already exists, do you want to overwrite it? [Y/N]. Please use 'Y' and 'N' to indicate preference.:  ")
    if answer == "N":
        filename = input("Please input a new filename: ")
        if filename[-4:] != ".csv":
            filename = filename+".csv"

while path.exists(model_file):
    answer = input("The model file already exists, do you want to overwrite it? [Y/N]:  ")
    while answer not in ["Y", "N"]:
        answer = input("This file already exists, do you want to overwrite it? [Y/N]. Please use 'Y' and 'N' to indicate preference.:  ")
    if answer == "N":
        model_file = input("Please input a new filename: ")
        if model_file[-3:] != ".pt":
            model_file = model_file+".pt"

while path.exists(target_file):
    answer = input("The target file already exists, do you want to overwrite it? [Y/N]:  ")
    while answer not in ["Y", "N"]:
        answer = input("This file already exists, do you want to overwrite it? [Y/N]. Please use 'Y' and 'N' to indicate preference.:  ")
    if answer == "N":
        target_file = input("Please input a new filename: ")
        if target_file[-3:] != ".pt":
            target_file = target_file+".pt"

###########################################
#TRAINING
###########################################

memory = 1000 #how many examples do we want to remember in the past
noisy = True #do we want Noisy Linear Layers
nodes = 16 #how many inner nodes in the DQN

line, = plt.plot([],[])

plt.xlim(0,num_eps)
plt.ylim(-500,10000)

data_tracker = []

def update_line(line, x, y):
    line.set_xdata(np.append(line.get_xdata(), x))
    line.set_ydata(np.append(line.get_ydata(), y))
    plt.draw()

def train():
    #initialize the network and target parameters with their noisy params
    DQN_target = DQN(noisy,nodes,input_size=49)
    DQN_current = DQN(noisy,nodes,input_size=49)
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

        greedy_eps -= 3*(greedy_max-greedy_min)/num_eps
        greedy_eps = max(greedy_min, greedy_eps)

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
            current_wind = s_t[0][25].float().numpy()

            s_next, r_t, done = Battery.take_step(a_t, current_price, current_wind) #execute action and recieve reward
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
            q_t = torch.gather(Q_current, 1, a_batch)

            loss = F.mse_loss(y_t,q_t)#Do a gradient descent with loss of y_t - (DQN(state,action)^2) and update DQN


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #every C steps also set DQN_target weights to be DQN weights
            if t % C == 0:
                DQN_target.load_state_dict(DQN_current.state_dict())

            t += 1
            #Battery.render()
            (profit, socmax, reward_total, soc, pow, price, wind_power) = Battery.render()
            data_tracker.append([profit, socmax, reward_total, soc, pow, price, wind_power])
        update_line(line,[e],[profit])
        plt.pause(0.05)
    plt.show()
    torch.save(DQN_current, model_file)
    torch.save(DQN_target, target_file)
    print("done")
    df = pd.DataFrame(data=data_tracker,columns=["profit", "SOCmax", "reward_total", "soc", "power", "price","wind"])
    df.to_csv(filename, index=False)
train()
