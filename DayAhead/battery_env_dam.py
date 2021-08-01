import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import pandas as pd
import torch

class BatteryEnv(gym.Env):
    def __init__(self, df, dam_df, degcost=0, flag=1, Eess=1, SOCmax=1, SOCmin=0, nu=0.9):
        self.df = df
        self.dam_df = dam_df
        self.Eess = Eess #MWh
        self.SOC = SOCmin #a percentage of capacity
        self.SOCmin = SOCmin
        self.SOCmax = SOCmax
        self.steps = len(self.df)
        self.actions = np.zeros(self.steps) #stores historical actions that I take
        self.nu= nu #efficiency
        self.hour2rtm = 12
        self.ACTION_LOOKUP = {
            0: self.Eess/self.hour2rtm*2, #500 kW -- max(lowerbound, -500)
            1: self.Eess/self.hour2rtm, #250 kW
            2: 0, #0 kW
            3: -self.Eess/self.hour2rtm, #-250kW
            4: -self.Eess/self.hour2rtm*2 #-500 kW
            }
        self.reward_total = 0
        self.flag = flag
        self.degcost = degcost
        self.step = 0
        self.eol = 0.3
        self.p_cyc = 0.5
        self.scaling_factor = 1
        # Actions of the format Buy x% of capacity, Sell x% of capacity, Hold, etc.
        self.action_space = spaces.Discrete( 5 )

        lows_array = np.zeros([1,self.hour2rtm+1])
        highs_array = np.ones([1,self.hour2rtm+1])
        lows_array[0] = self.SOCmin
        highs_array = highs_array*(max(self.df['system_energy_price_rt'])+1)

        highs_array[:,0] = self.SOCmax
        # Prices contains the energy values

        self.observation_space = spaces.Box(low=lows_array, high=highs_array, dtype=np.float16)

    def get_dam(self):
        t = self.step//self.hour2rtm
        return self.dam_df.power[t]
    
    def get_dam_price(self):
        t = self.step//self.hour2rtm
        return self.dam_df.price[t]

    def _next_obs(self):
        prices = self.df.system_energy_price_rt[self.step:self.step+self.hour2rtm].to_numpy()
        done = self.step >= len(self.df)-(self.hour2rtm+1)
        # if self.SOCmax < 0.99:
        #     done = True
        obs = torch.from_numpy(np.expand_dims(np.append([self.SOC],[prices]), 0)).float()
        # print("STEP: ", self.step, "\n", "obs: ", obs)
        return done, obs

    def eff(self):
        return self.nu

    def deg(self):
        DOD = abs(self.Pow)*100/self.Eess
        if abs(self.Pow) < 0.01:
            return self.Eess*self.eol*(1-self.p_cyc)/(365*24*10*12) 
        else:
            N_cyc = 0.0035*DOD**3 + 0.2215*DOD**2-132.29*DOD+10555
            return self.eol*self.p_cyc*abs(self.Pow)/(2*N_cyc)

    def _take_action(self, action, ct):
        '''
        ct = current price of electricity
        action = integer between 0 and 4 inclusive

        Power is in units MW
        '''
        self.SOC -= self.get_dam()/self.hour2rtm

        Pow_high = min((self.SOC - self.SOCmin)*self.Eess/(self.eff()), self.ACTION_LOOKUP[0])
        Pow_low = max((self.SOC - self.SOCmax)*self.Eess/(self.eff()), self.ACTION_LOOKUP[4])
        flag = 0

        Pow = self.ACTION_LOOKUP[action]

        #if Power is negative, it's charging. If Power is positive, it's discharging
        if Pow_high < Pow_low: #if the bounds aren't possible -- which shouldnt ever come up
            Pow = 0
            print("INVALID BOUNDS")
            print("Pow High: ", Pow_high)
            print("Pow Low: ", Pow_low)
        else:
            if not(Pow_low <= Pow <= Pow_high):
                flag = 1
                if Pow > Pow_high:
                    Pow = Pow_high
                elif Pow < Pow_low:
                    Pow = Pow_low
            self.SOC -= self.eff()*Pow #charging when power is negative
            P_max = self.ACTION_LOOKUP[0]
            # print(ct, Pow, self.alpha,P_max+0.0001)
            deg = self.deg()
            Rt = self.scaling_factor*((ct*Pow)/P_max) + ((flag==1)* -self.flag) - deg*self.degcost*10/self.eol
            self.profit += ct*Pow + self.get_dam()/self.hour2rtm*self.get_dam_price()
            self.Pow = Pow
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
        self.SOC = np.random.choice([0,0.16,0.32,0.48, 0.64, 0.80, 0.96]) #self.SOCmin #a percentage of capacity
        self.SOCmax = self.Eess
        self.actions = np.zeros(self.steps)
        self.step = 0
        self.profit = 0
        self.reward_total = 0
        self.reward = 0
        self.Pow = 0

    def render(self, mode='human'):
        print("SOC: ", self.SOC, "SOC Max: ", self.SOCmax, "action: ", self.actions[self.step-1], "profit: ", self.profit, "price: ", self.df.system_energy_price_rt[self.step], "reward: ", self.reward, "reward total: ", self.reward_total, "deg: ", self.deg()*self.degcost*10/self.eol)
        return(self.profit, self.SOC, self.SOCmax, self.Pow, self.df.system_energy_price_rt[self.step],self.get_dam(),self.get_dam_price())