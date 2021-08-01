import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from os import path
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


num_eps = 1000
C = 1000
dam_df = pd.read_csv('./data/DAMPricing2019_week.csv')
rtm_df = pd.read_csv('./data/RTMPrices2019_week.csv')
batch_size = 32
gamma = 0.95
degcost = 0
Battery = BatteryEnv(dam_df=dam_df, rtm_df=rtm_df)
greedy_max = 0.9
greedy_min = 0.01

filename = 'gamma{}'.format(gamma)
model_rtm_file = "rtm_current_{}.pt".format(filename)
target_rtm_file = "rtm_target_{}.pt".format(filename)
model_dam_file = "dam_current_{}.pt".format(filename)
target_dam_file = "dam_target_{}.pt".format(filename)
filename = filename+'.csv'

while path.exists(filename):
    answer = input("This file already exists, do you want to overwrite it? [Y/N]:  ")
    while answer not in ["Y", "N"]:
        answer = input("This file already exists, do you want to overwrite it? [Y/N]. Please use 'Y' and 'N' to indicate preference.:  ")
    if answer == "N":
        filename = input("Please input a new filename: ")
        if filename[-4:] != ".csv":
            filename = filename+".csv"
    else:
        break

while path.exists(model_rtm_file):
    answer = input("The model file already exists, do you want to overwrite it? [Y/N]:  ")
    while answer not in ["Y", "N"]:
        answer = input("This file already exists, do you want to overwrite it? [Y/N]. Please use 'Y' and 'N' to indicate preference.:  ")
    if answer == "N":
        model_rtm_file = input("Please input a new filename: ")
        if model_rtm_file[-3:] != ".pt":
            model_rtm_file = model_rtm_file+".pt"
    else:
        break

while path.exists(target_rtm_file):
    answer = input("The target file already exists, do you want to overwrite it? [Y/N]:  ")
    while answer not in ["Y", "N"]:
        answer = input("This file already exists, do you want to overwrite it? [Y/N]. Please use 'Y' and 'N' to indicate preference.:  ")
    if answer == "N":
        target_rtm_file = input("Please input a new filename: ")
        if target_rtm_file[-3:] != ".pt":
            target_rtm_file = target_rtm_file+".pt"
    else:
        break

memory = 1000 #how many examples do we want to remember in the past
noisy = True #do we want Noisy Linear Layers
nodes = 16 #how many inner nodes in the DQN

line, = plt.plot([],[])

plt.xlim(0,num_eps)
plt.ylim(-1000,3000)

data_tracker = []

def update_line(line, x, y):
    line.set_xdata(np.append(line.get_xdata(), x))
    line.set_ydata(np.append(line.get_ydata(), y))
    plt.draw()

def train():
    #initialize the network and target parameters with their noisy params
    DQN_target_DAM = DQN(noisy,nodes)
    DQN_current_DAM = DQN(noisy,nodes)
    DQN_target_RTM = DQN(noisy,nodes,input_size=13)
    DQN_current_RTM = DQN(noisy,nodes,input_size=13)

    optimizer_rtm = optim.Adam(DQN_current_RTM.parameters(), lr=0.00025)
    optimizer_dam = optim.Adam(DQN_current_DAM.parameters(), lr=0.00025)
    # greed_mod = 31
    #initialize the replay memory
    MemoryRTM = ReplayMemory(memory)
    MemoryDAM = ReplayMemory(memory)

    rtm_t = 0
    dam_t = 0

    greedy_eps = greedy_max

    for e in range(num_eps):
        print("EPISODE: ", e)
        #reset the battery and obtain the state for the battery. Also -- we just started so we're not done yet.
        Battery.reset()
        done_dam = False
        done_rtm = False

        greedy_eps -= 3*(greedy_max-greedy_min)/num_eps
        greedy_eps = max(greedy_min, greedy_eps)

        while (done_rtm or done_dam) == False: #for t = 1,....,T
            if noisy:
                DQN_target_RTM.sample_noise()
                DQN_current_RTM.sample_noise()
                DQN_target_DAM.sample_noise()
                DQN_current_DAM.sample_noise()

            done_rtm, s_t_rtm = Battery._next_obs_rtm()
            s_t_rtm = Variable(s_t_rtm) #1 x state_size
            if random.random() <= (1-greedy_eps):
                a_t_rtm = DQN_current_RTM.select_action(s_t_rtm).item() #select an action based on what DQN thinks is best. This is a single integer in the range(all_action_size)
            else:
                a_t_rtm = random.randint(0,4)
            rtm_price = s_t_rtm[0][1].float().numpy()

            if dam_t%12 == 0:  
                done_dam, s_t_dam= Battery._next_obs_dam()
                s_t_dam = Variable(s_t_dam)
                if random.random() <= (1-greedy_eps):
                    a_t_dam = DQN_current_DAM.select_action(s_t_dam).item() #select an action based on what DQN thinks is best. This is a single integer in the range(all_action_size)
                else:
                    a_t_dam = random.randint(0,4)
                dam_price = s_t_dam[0][1].float().numpy()
                s_next_rtm, r_t_rtm, done_rtm,s_next_dam, r_t_dam, done_dam  = Battery.take_step(rtm_action=a_t_rtm, current_rtm_price=rtm_price,dam_action=a_t_dam, dam_valid=True, current_dam_price=dam_price) #execute action and recieve reward
                MemoryRTM.push(s_t_rtm,a_t_rtm,s_next_rtm,r_t_rtm) #store (s_t_rtm, a_t_rtm, s_next, r_t) in ReplayMemory
                MemoryDAM.push(s_t_dam,a_t_dam,s_next_dam,r_t_dam) #store (s_t_rtm, a_t_rtm, s_next, r_t) in ReplayMemory
                if noisy: #sample noise again in the Neural Networks
                    DQN_target_RTM.sample_noise()
                    DQN_current_RTM.sample_noise()
                    DQN_target_DAM.sample_noise()
                    DQN_current_DAM.sample_noise()

                if len(MemoryRTM) < batch_size:
                    batch_rtm = MemoryRTM.sample(len(MemoryRTM))
                else:
                    batch_rtm = MemoryRTM.sample(batch_size)

                if len(MemoryDAM) < batch_size:
                    batch_dam = MemoryDAM.sample(len(MemoryDAM))
                else:
                    batch_dam = MemoryDAM.sample(batch_size)

                s_batch_rtm = torch.cat([getattr(x, 'state') for x in batch_rtm], dim=0) #batch_size x state_size
                s_next_batch_rtm = torch.cat([getattr(x, 'next_state') for x in batch_rtm], dim=0) #batch_size x state_size
                a_batch_rtm = torch.Tensor([[getattr(x, 'action')] for x in batch_rtm]).long() #batch_size x 1
                r_batch_rtm = torch.Tensor([[getattr(x, 'reward')] for x in batch_rtm]) #batch_size x 1

                a_prime_batch_rtm = DQN_current_RTM.select_action(s_next_batch_rtm).long().unsqueeze(1) #batch_size x 1
                targets_rtm = DQN_target_RTM.forward(s_next_batch_rtm) #estimate the target values for all actions. This returns a Q value matrix of size: batch_size x all_action_size
                y_t_rtm = r_batch_rtm + gamma*torch.gather(targets_rtm, 1, a_prime_batch_rtm) #batch_size x 1 (i.e. a single Qvalue for each input in the batch)

                Q_current_rtm =  DQN_current_RTM.forward(s_batch_rtm)
                q_t_rtm = torch.gather(Q_current_rtm, 1, a_batch_rtm)
                loss_rtm = F.mse_loss(y_t_rtm,q_t_rtm)#Do a gradient descent with loss of y_t - (DQN(state,action)^2) and update DQN
                optimizer_rtm.zero_grad()
                loss_rtm.backward()
                optimizer_rtm.step()

                ###DAM####
                s_batch_dam = torch.cat([getattr(x, 'state') for x in batch_dam], dim=0) #batch_size x state_size
                s_next_batch_dam = torch.cat([getattr(x, 'next_state') for x in batch_dam], dim=0) #batch_size x state_size
                a_batch_dam = torch.Tensor([[getattr(x, 'action')] for x in batch_dam]).long() #batch_size x 1
                r_batch_dam = torch.Tensor([[getattr(x, 'reward')] for x in batch_dam]) #batch_size x 1

                a_prime_batch_dam = DQN_current_DAM.select_action(s_next_batch_dam).long().unsqueeze(1) #batch_size x 1
                targets_dam = DQN_target_DAM.forward(s_next_batch_dam) #estimate the target values for all actions. This returns a Q value matrix of size: batch_size x all_action_size
                y_t_dam = r_batch_dam + gamma*torch.gather(targets_dam, 1, a_prime_batch_dam) #batch_size x 1 (i.e. a single Qvalue for each input in the batch)

                Q_current_dam =  DQN_current_DAM.forward(s_batch_dam)
                q_t_dam = torch.gather(Q_current_dam, 1, a_batch_dam)
                loss_dam = F.mse_loss(y_t_dam,q_t_dam)#Do a gradient descent with loss of y_t - (DQN(state,action)^2) and update DQN
                optimizer_dam.zero_grad()
                loss_dam.backward()
                optimizer_dam.step()
                
                #every C steps also set DQN_target weights to be DQN weights
                if rtm_t % C == 0:
                    DQN_target_RTM.load_state_dict(DQN_current_RTM.state_dict())
                if dam_t % C == 0:
                    DQN_target_DAM.load_state_dict(DQN_current_DAM.state_dict())

                rtm_t += 1
                dam_t += 1
            else:
                s_next_rtm, r_t_rtm, done_rtm,_,_,_= Battery.take_step(a_t_rtm, rtm_price) #execute action and recieve reward
                MemoryRTM.push(s_t_rtm,a_t_rtm,s_next_rtm,r_t_rtm) #store (s_t_rtm, a_t_rtm, s_next, r_t) in ReplayMemory

                if noisy: #sample noise again in the Neural Networks
                    DQN_target_RTM.sample_noise()
                    DQN_current_RTM.sample_noise()

                if len(MemoryRTM) < batch_size:
                    batch_rtm = MemoryRTM.sample(len(MemoryRTM))
                else:
                    batch_rtm = MemoryRTM.sample(batch_size)

                s_batch_rtm = torch.cat([getattr(x, 'state') for x in batch_rtm], dim=0) #batch_size x state_size
                s_next_batch_rtm = torch.cat([getattr(x, 'next_state') for x in batch_rtm], dim=0) #batch_size x state_size
                a_batch_rtm = torch.Tensor([[getattr(x, 'action')] for x in batch_rtm]).long() #batch_size x 1
                r_batch_rtm = torch.Tensor([[getattr(x, 'reward')] for x in batch_rtm]) #batch_size x 1

                a_prime_batch_rtm = DQN_current_RTM.select_action(s_next_batch_rtm).long().unsqueeze(1) #batch_size x 1
                targets_rtm = DQN_target_RTM.forward(s_next_batch_rtm) #estimate the target values for all actions. This returns a Q value matrix of size: batch_size x all_action_size
                y_t_rtm = r_batch_rtm + gamma*torch.gather(targets_rtm, 1, a_prime_batch_rtm) #batch_size x 1 (i.e. a single Qvalue for each input in the batch)

                Q_current_rtm =  DQN_current_RTM.forward(s_batch_rtm)
                q_t_rtm = torch.gather(Q_current_rtm, 1, a_batch_rtm)
                loss_rtm = F.mse_loss(y_t_rtm,q_t_rtm)#Do a gradient descent with loss of y_t - (DQN(state,action)^2) and update DQN
                optimizer_rtm.zero_grad()
                loss_rtm.backward()
                optimizer_rtm.step()

                #every C steps also set DQN_target weights to be DQN weights
                if rtm_t % C == 0:
                    DQN_target_RTM.load_state_dict(DQN_current_RTM.state_dict())

                rtm_t += 1
                #Battery.render()
            (profit, socmax, reward_total, soc, actions_rtm, actions_dam) = Battery.render()
            data_tracker.append([profit, socmax, reward_total, soc, actions_rtm, actions_dam])
            update_line(line,[e],[profit])
            plt.pause(0.05)
    plt.show()
    torch.save(DQN_current, model_file)
    torch.save(DQN_target, target_file)
    print("done")
    df = pd.DataFrame(data=data_tracker,columns=["profit", "SOCmax", "reward_total", "soc", "power", "price","wind"])
    df.to_csv(filename, index=False)
train()
