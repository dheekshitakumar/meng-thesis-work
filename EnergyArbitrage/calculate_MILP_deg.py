import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from os import path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.autograd import Variable
#import plotly.graph_objects as go

class MILPBatteryEnvDeg(gym.Env):
    def __init__(self, df, dcost, Eess=1, SOCmax=1, SOCmin=0, nu=0.9, PowStBy=0):
        self.df = df

        self.Eess = Eess #MWh
        self.eol = 0.3
        self.p_cyc = 0.5
        self.SOC = SOCmin #a percentage of capacity
        self.SOCmin = SOCmin
        self.SOCmax = SOCmax
        self.steps = len(self.df)
        self.T = 1 #1 hour
        self.actions = np.zeros(self.steps) #stores historical actions that I take
        self.nu= nu #efficiency
        self.PowStBy = PowStBy
        self.alpha = 0 #apparently this needs to be updated every episodes
        self.eff_matrix = {0: np.array([0.085580145,0.14076747,10000,10000,10000,10000,10000,10000]),
        0.1: np.array([0.072265769,0.093899638,0.111961879,0.1308937,0.135539658,10000,10000,10000]),
        0.2: np.array([0.070363069,0.086865001, 0.099606347, 0.115347959, 0.115347959, 0.151795977, 10000, 10000]),
        0.3: np.array([0.069750174,0.084556325,0.095340474,0.110500948,0.110500948,0.140631875,0.143902655,10000]),
        0.4: np.array([0.069565483,0.083779348,0.093530846,0.109166803,0.109166803,0.135942257,0.141546866,0.15311391]),
        0.5: np.array([0.069604292,0.083774388,0.092827279,0.109689971,0.109689971,0.134128164,0.142470176,0.150039168]),
        0.6: np.array([0.069784472,0.084230127,0.092642634,0.111430597,0.111430597,0.133664437,0.145546597,0.148557198]),
        0.7: np.array([0.070061048,0.084968586,0.092604345,0.114056482,0.114056482,0.133597382,0.150200734,0.150200734]),
        0.8: np.array([0.070403289,0.085860892,0.092408538,0.117354506,0.117354506,0.133149138,0.15606859,10000]),
        0.9: np.array([0.071151422,0.087603779,0.091552401,0.123976622,0.124939013,0.128615332,0.169658505,10000]),
        1.0: np.array([0.071151422,0.087603779,0.091552401,0.123976622,0.124939013,0.128615332,0.169658505,10000])}
        self.size_piece = 0.1672
        self.degcost = dcost
        self.reward_total = 0
        self.step = 0
        self.scaling_factor = 1
        self.reward = 0
        # Actions of the format Buy x% of capacity, Sell x% of capacity, Hold, etc.
        self.action_space = spaces.Discrete( 5 )

        lows_array = np.zeros([1,25])
        highs_array = np.ones([1,25])
        lows_array[0] = self.SOCmin
        highs_array = highs_array*(max(self.df['Hourly_USD_per_MWh'])+1)

        highs_array[:,0] = self.SOCmax
        # Prices contains the energy values

        self.observation_space = spaces.Box(low=lows_array, high=highs_array, dtype=np.float16)

    def _next_obs(self):
        prices = self.df.Hourly_USD_per_MWh[self.step:self.step+24].to_numpy()
        done = self.step >= len(self.df)-25
        obs = torch.from_numpy(np.expand_dims(np.append([self.SOC],[prices]), 0)).float()
        # print("STEP: ", self.step, "\n", "obs: ", obs)
        return done, obs

    def eff(self, power=None):
        #make this efficiency
        #round power to determine row to index into
        #floor of the difference in power
        # p = abs(self.SOC - power/2)
        # key_power = math.ceil(10*p)/10
        # power = abs(power)
        # eff_row = self.eff_matrix[key_power]
        # num_power_pieces = int(power//self.size_piece)
        # loss = self.size_piece*np.sum(eff_row[:num_power_pieces]) + (power -self.size_piece*num_power_pieces)*eff_row[num_power_pieces]
        return 0.9

    def deg(self):
        DOD = abs(self.Pow)*100/self.Eess
        if self.Pow == 0:
            return self.Eess*self.eol*(1-self.p_cyc)/(365*24*10) 
        else:
            N_cyc = 0.0035*DOD**3 + 0.2215*DOD**2-132.29*DOD+10555
            return self.eol*self.p_cyc*abs(self.Pow)/(2*N_cyc)
    
    def limit_power(self,Pow_low,Pow_high,Pow):
        flag = 0
        eps = 0.1
        if not(Pow_low-eps <= Pow <= Pow_high+eps):
            flag = 1
        if not(Pow_low <= Pow <= Pow_high):
            if Pow > Pow_high:
                Pow = Pow_high
            elif Pow < Pow_low:
                Pow = Pow_low
        return Pow, flag

    def _take_action(self, action, ct):
        '''
        ct = current price of electricity
        action = integer between 0 and 4 inclusive

        Power is in units MW
        '''
        if action == 0:
            SOC_dif = 1/self.Eess*self.T*self.PowStBy
            self.SOC -= SOC_dif
            self.Pow = 0
            deg = self.deg()
            Rt = -deg*self.degcost*10/self.eol
        else:
            #self.Pow = Pow_grid + self.eff(Pow_grid)
            #self.loss = self.eff(Pow_grid)
            #self.SOC -= (1/self.Eess*self.T*self.Pow) #charging when power is negative
            if action > 0:
                self.Pow = action/self.eff()
            else:
                self.Pow = action*self.eff()
            self.SOC -= (1/self.Eess*self.T*self.Pow)
            if self.SOC > self.SOCmax:
                self.SOC = self.SOCmax 
            if self.SOC < self.SOCmin:
                self.SOC = self.SOCmin
            P_max = self.Eess
            # print(ct, Pow, self.alpha,P_max+0.0001)
            deg = self.deg()
            Rt = self.scaling_factor*(ct*(action/P_max)) - deg*self.degcost*10/0.3
            self.profit += ct*(action)
            self.Pow = action
        self.step += 1
        return Rt

    def take_step(self, action, current_price):
        self.actions[self.step] = action
        reward = self._take_action(action, current_price)
        self.SOCmax -= self.deg()
        self.reward_total += reward
        self.reward = reward
        done, observation = self._next_obs()
        return observation, reward, done


    def reset(self):
        self.SOC = self.SOCmin #a percentage of capacity
        self.actions = np.zeros(self.steps)
        self.step = 0
        self.profit = 0
        self.reward_total = 0
        self.Pow = 0
        self.SOCmax = 1

    def render(self, mode='human'):
        print("SOC:", self.SOC, "power:", self.Pow, "SOC_max: ", self.SOCmax, "action:", self.actions[self.step-1], "profit: ", self.profit, "price", self.df.Hourly_USD_per_MWh[self.step],"reward", self.reward)
        return(self.profit, self.SOCmax, self.reward_total, self.SOC, self.Pow, self.df.Hourly_USD_per_MWh[self.step])


data = pd.read_csv("data/Prices2013.csv")
actions = pd.read_csv('0k_2013.csv')['Power']

Battery = MILPBatteryEnvDeg(df=data, dcost=10000)
Battery.reset()

data_tracker = []
iteration = 0
done = False 

while done == False:
    done, s_t = Battery._next_obs()
    s_t = Variable(s_t) #1 x state_size
    a_t = actions[iteration]
    current_price = s_t[0][1].float().numpy()
    s_next, r_t, done = Battery.take_step(a_t, current_price)
    (profit, socmax, reward_total, soc, pow, price) = Battery.render()
    data_tracker.append([socmax, soc, pow, price])
    plt.pause(0.05)
    iteration += 1

try:
    df = pd.DataFrame(data=data_tracker,columns=["profit", "SOCmax", "reward_total", "soc", "power", "price"])
except:
    df = pd.DataFrame(data=data_tracker)

df.to_csv("power_milp_0kdegradation.csv", index=False)
print("Saved to CSV")