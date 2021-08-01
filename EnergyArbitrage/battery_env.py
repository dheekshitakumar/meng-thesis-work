import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import pandas as pd
import torch

class BatteryEnv(gym.Env):
    def __init__(self, df, Eess=1, SOCmax=1, SOCmin=0, nu=0.9, PowStBy=0):
        self.df = df

        self.Eess = Eess #MWh
        self.SOC = SOCmin #a percentage of capacity
        self.SOCmin = SOCmin
        self.SOCmax = SOCmax
        self.steps = len(self.df)
        self.T = 1 #1 hour
        self.actions = np.zeros(self.steps) #stores historical actions that I take
        self.nu= nu #efficiency
        self.PowStBy = PowStBy
        self.alpha = 0 #apparently this needs to be updated every episodes


        self.reward_total = 0
        self.step = 0
        self.scaling_factor = 1
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

    def eff(self):
        return self.nu

    def deg(self):
        return self.alpha

    def _take_action(self, action, ct):
        '''
        ct = current price of electricity
        action = integer between 0 and 4 inclusive

        Power is in units MW
        '''
        action_type = ACTION_LOOKUP[action]
        #if Power is negative, it's charging. If Power is positive, it's discharging
        Pow_high = min((self.SOC - self.SOCmin)*self.Eess/(self.eff()*self.T), self.Eess/2)
        Pow_low = max((self.SOC - self.SOCmax)*self.Eess/(self.eff()*self.T), -self.Eess/2)
        flag = 0
        if Pow_high < Pow_low: #if the bounds aren't possible -- which shouldnt ever come up
            Pow = 0
        else:
            if action_type == "dischargefull":
                Pow = self.Eess/2
                if not(Pow_low <= Pow <= Pow_high):
                    if Pow > Pow_high:
                        Pow = Pow_high
                        flag = 1
                    elif Pow < Pow_low:
                        Pow = Pow_low
            elif action_type == "dischargehalf":
                Pow = self.Eess/4
                if not(Pow_low <= Pow <= Pow_high):
                    if Pow > Pow_high:
                        Pow = Pow_high
                        flag = 1
                    elif Pow < Pow_low:
                        Pow = Pow_low
            elif action_type == "chargefull":
                Pow = -self.Eess/2
                if not(Pow_low <= Pow <= Pow_high):
                    if Pow > Pow_high:
                        Pow = Pow_high
                    elif Pow < Pow_low:
                        Pow = Pow_low
            elif action_type == "chargehalf":
                Pow = -self.Eess/4
                if not(Pow_low <= Pow <= Pow_high):
                    if Pow > Pow_high:
                        Pow = Pow_high
                    elif Pow < Pow_low:
                        Pow = Pow_low
        if action_type == "rest":
            SOC_dif = 1/self.Eess*self.T*self.PowStBy
            self.SOC -= SOC_dif
            Rt = 0
            self.Pow = 0
        else:
            self.SOC -= 1/self.Eess*self.T*self.eff()*Pow #charging when power is negative
            P_max = self.Eess/2
            # print(ct, Pow, self.alpha,P_max+0.0001)
            Rt = self.scaling_factor*((ct*Pow - self.deg()*abs(Pow))/P_max) + ((flag==1)* -5)
            self.profit += ct*Pow - self.deg()*abs(Pow)
            self.Pow = Pow
        self.step += 1
        return Rt

    def take_step(self, action, current_price):
        self.actions[self.step] = action
        reward = self._take_action(action, current_price)
        self.reward_total += reward
        done, observation = self._next_obs()
        return observation, reward, done


    def reset(self):
        self.SOC = self.SOCmin #a percentage of capacity
        self.actions = np.zeros(self.steps)
        self.step = 0
        self.profit = 0
        self.reward_total = 0
        self.Pow = 0

    def render(self, mode='human'):
        print("SOC:", self.SOC, "action:", self.actions[self.step-1], "profit: ", self.profit, "price", self.df.Hourly_USD_per_MWh[self.step])
        return(self.profit, self.SOCmax, self.reward_total, self.SOC, self.Pow, self.df.Hourly_USD_per_MWh[self.step])


ACTION_LOOKUP = {
    0: "dischargefull", #500 kW -- max(lowerbound, -500)
    1: "dischargehalf", #250 kW
    2: "rest", #0 kW
    3: "chargehalf", #-250kW
    4: "chargefull" #-500 kW
}
