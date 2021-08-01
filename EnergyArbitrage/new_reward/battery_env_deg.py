import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import pandas as pd
import torch
import math

class BatteryConstEnvDeg(gym.Env):
    def __init__(self, df, dcost, Eess=1, SOCmax=1, SOCmin=0, nu=0.9, PowStBy=0, flag=10):
        self.df = df
        self.flag = flag
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
        self.degcost = dcost
        self.reward_total = 0
        self.step = 0
        self.scaling_factor =  1
        self.reward = 0
        self.avg_price = 0
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
        return self.nu

    def deg(self):
        DOD = abs(self.Pow)*100/self.Eess
        if self.Pow > 0:
            return 0
        elif self.Pow == 0:
            return self.Eess*self.eol*(1-self.p_cyc)/(365*24*10)
        else:
            N_cyc = 0.0035*DOD**3 + 0.02214*DOD**2-132.29*DOD+10555
            return self.eol*self.p_cyc*abs(self.Pow)/(N_cyc)
    
    def limit_power(self,Pow_low,Pow_high,Pow):
        flag = 0
        if not(Pow_low <= Pow <= Pow_high):
            if Pow > Pow_high:
                Pow = Pow_high
                flag = 1
            elif Pow < Pow_low:
                flag = 1
                Pow = Pow_low
        return Pow, flag

    def _take_action(self, action, ct):
        '''
        ct = current price of electricity
        action = integer between 0 and 4 inclusive

        Power is in units MW
        '''
        action_type = ACTION_LOOKUP[action]
        Pow_high = min((self.SOC - self.SOCmin)*self.Eess/(self.eff()*self.T), self.Eess)
        Pow_low = max((self.SOC - self.SOCmax)*self.Eess/(self.eff()*self.T), -self.Eess)
        if Pow_high < Pow_low: #if the bounds aren't possible -- which shouldnt ever come up
            Pow_grid = 0
            flag=0
        else:
            if action_type == "dischargefull":
                Pow_grid = self.Eess
                Pow_grid, flag = self.limit_power(Pow_low,Pow_high,Pow_grid)
            elif action_type == "dischargehalf":
                Pow_grid = self.Eess/2
                Pow_grid, flag = self.limit_power(Pow_low,Pow_high,Pow_grid)
            elif action_type == "chargefull":
                Pow_grid = -self.Eess
                Pow_grid, flag = self.limit_power(Pow_low,Pow_high,Pow_grid)
            elif action_type == "chargehalf":
                Pow_grid = -self.Eess/2
                Pow_grid, flag = self.limit_power(Pow_low,Pow_high,Pow_grid)
        if action_type == "rest":
            SOC_dif = 1/self.Eess*self.T*self.PowStBy
            self.SOC -= SOC_dif
            Rt = 0
            self.Pow = 0
        else:
            #self.Pow = Pow_grid + self.eff(Pow_grid)
            #self.loss = self.eff(Pow_grid)
            #self.SOC -= (1/self.Eess*self.T*self.Pow) #charging when power is negative
            self.SOC -= 1/self.Eess*self.T*self.eff()*Pow_grid
            if self.SOC > self.SOCmax:
                self.SOC = self.SOCmax 
            if self.SOC < self.SOCmin:
                self.SOC = self.SOCmin
            P_max = self.Eess
            # print(ct, Pow, self.alpha,P_max+0.0001)
            deg = self.deg()
            #future_avg = sum(self.df.Hourly_USD_per_MWh[self.step+1:self.step+24])/23
            #past_avg = sum(self.df.Hourly_USD_per_MWh[self.step-24:self.step])/23
            Rt = self.scaling_factor*(ct*(Pow_grid/P_max) + ((flag==1)* -self.flag) - deg*self.degcost) #+ ((ct-past_avg)*(Pow_grid/P_max)) + ((ct-future_avg)*(Pow_grid/P_max))
            self.profit += ct*(Pow_grid)
            self.Pow = Pow_grid*self.eff()
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
        self.avg_price = 0

    def render(self, mode='human'):
        print("SOC:", "{:.4f}".format(self.SOC), "power:", "{:.4f}".format(self.Pow), "SOC_max: ", "{:.6f}".format(self.SOCmax), "action:", self.actions[self.step-1], "profit: ", "{:.4f}".format(self.profit), "price: ", self.df.Hourly_USD_per_MWh[self.step],"reward: ", "{:.4f}".format(self.reward), "deg: ", "{:.4f}".format(self.deg()*self.degcost), "reward_total: ", "{:.4f}".format(self.reward_total))
        return(self.profit, self.SOCmax, self.reward_total, self.SOC, self.Pow, self.df.Hourly_USD_per_MWh[self.step])


ACTION_LOOKUP = {
    0: "dischargefull", #1 MW -- max(lowerbound, -500)
    1: "dischargehalf", #500 kW
    2: "rest", #0 kW
    3: "chargehalf", #-500kW
    4: "chargefull" #-1 MW
}
