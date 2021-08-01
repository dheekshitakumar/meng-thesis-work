import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import pandas as pd
import torch

class BatteryEnv(gym.Env):
    def __init__(self, rtm_df, dam_df, Eess=1, SOCmin=0, SOCmax=1):
        self.dam_df = dam_df
        self.rtm_df = rtm_df
        self.Eess = Eess #MWh
        self.SOC = SOCmin #a percentage of capacity
        self.SOCmin = SOCmin
        self.SOCmax = SOCmax
        self.nu = 0.9
        self.alpha = 0
        self.scaling_factor = 1
        self.PowStBy = 0

        self.steps_dam = len(self.dam_df)
        self.steps_rtm = len(self.rtm_df)

        self.T = 1 #1 hour

        self.actions_rtm = np.zeros(self.steps_rtm) #stores real time actions that I take in real-time market
        self.actions_dam = np.zeros(self.steps_dam) #stores historical actions that I take in day ahead market

        self.reward_rtm = 0
        self.reward_dam = 0

        self.rtmstep = 0
        self.damstep = 0


        # Actions of the format Buy x% of capacity, Sell x% of capacity, Hold, etc.
        self.action_space_dam = spaces.Discrete( 5 )
        self.action_space_rtm = spaces.Discrete( 5 )

        lows_array_rtm = np.zeros([1,13])
        highs_array_rtm = np.ones([1,13])
        lows_array_dam = np.zeros([1,25])
        highs_array_dam = np.ones([1,25])
        lows_array_rtm[0] = 0
        lows_array_dam[0] = 0
        max_price = max(max(self.rtm_df['total_lmp_rt']),max(self.dam_df['total_lmp_da']))
        highs_array_rtm = highs_array_rtm*(max_price+1)
        highs_array_dam = highs_array_dam*(max_price+1)
        highs_array_dam[:,0] = 1
        highs_array_rtm[:,0] = 1
        # Prices contains the energy values
        self.observation_space_rtm = spaces.Box(low=lows_array_rtm, high=highs_array_rtm, dtype=np.float16)
        self.observation_space_dam = spaces.Box(low=lows_array_dam, high=highs_array_dam, dtype=np.float16)

    def _next_obs_rtm(self):
        rtm_prices = self.rtm_df.total_lmp_rt[self.rtmstep:self.rtmstep+12].to_numpy()
        if len(rtm_prices) < 12:
            rtm_prices = np.concatenate([rtm_prices, [0]])
        done = self.rtmstep > len(self.rtm_df)-13
        obs = torch.from_numpy(np.expand_dims(np.append([self.SOC],[rtm_prices]), 0)).float()
        return done, obs
    
    def _next_obs_dam(self):
        dam_prices = self.dam_df.total_lmp_da[self.damstep:self.damstep+24].to_numpy()
        if len(dam_prices) < 24:
            dam_prices = np.concatenate([dam_prices, [0]])
        done = self.rtmstep > len(self.rtm_df)-25
        obs = torch.from_numpy(np.expand_dims(np.append([self.SOC],[dam_prices]), 0)).float()
        return done, obs

    def eff(self):
        return self.nu

    def deg(self):
        return self.alpha

    def _take_action_rtm(self, action, ct):
        '''
        ct = current price of electricity
        action = integer between 0 and 4 inclusive

        Power is in units MW
        '''
        action_type = ACTION_LOOKUP_RTM[action]
        #if Power is negative, it's charging. If Power is positive, it's discharging
        Pow_high = min((self.SOC - self.SOCmin)*self.Eess/(self.eff()*self.T), 0.04)
        Pow_low = max((self.SOC - self.SOCmax)*self.Eess/(self.eff()*self.T), -0.04)
        flag = 0
        if Pow_high < Pow_low: #if the bounds aren't possible -- which shouldnt ever come up
            Pow = 0
        else:
            if action_type == "dischargefull":
                Pow = 0.04
                if not(Pow_low <= Pow <= Pow_high):
                    if Pow > Pow_high:
                        Pow = Pow_high
                        flag = 1
                    elif Pow < Pow_low:
                        Pow = Pow_low
            elif action_type == "dischargehalf":
                Pow = 0.02
                if not(Pow_low <= Pow <= Pow_high):
                    if Pow > Pow_high:
                        Pow = Pow_high
                        flag = 1
                    elif Pow < Pow_low:
                        Pow = Pow_low
            elif action_type == "chargefull":
                Pow = -0.04
                if not(Pow_low <= Pow <= Pow_high):
                    if Pow > Pow_high:
                        Pow = Pow_high
                    elif Pow < Pow_low:
                        Pow = Pow_low
            elif action_type == "chargehalf":
                Pow = -0.02
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
            Rt = self.scaling_factor*((ct*Pow - self.deg()*abs(Pow))/P_max) + ((flag== 1)* -0.5)
            self.profit += ct*Pow - self.deg()*abs(Pow)
            self.Pow = Pow
        self.rtmstep += 1
        return Rt
    
    def _take_action_dam(self, action, ct):
        '''
        ct = current price of electricity
        action = integer between 0 and 4 inclusive

        Power is in units MW
        '''
        action_type = ACTION_LOOKUP_DAM[action]
        #if Power is negative, it's charging. If Power is positive, it's discharging
        Pow_high = min((self.SOC - self.SOCmin)*self.Eess/(self.eff()*self.T), 0.04)
        Pow_low = max((self.SOC - self.SOCmax)*self.Eess/(self.eff()*self.T), -0.04)
        flag = 0
        if Pow_high < Pow_low: #if the bounds aren't possible -- which shouldnt ever come up
            Pow = 0
        else:
            if action_type == "dischargefull":
                Pow = 0.5
                if not(Pow_low <= Pow <= Pow_high):
                    if Pow > Pow_high:
                        Pow = Pow_high
                        flag = 1
                    elif Pow < Pow_low:
                        Pow = Pow_low
            elif action_type == "dischargehalf":
                Pow = 0.25
                if not(Pow_low <= Pow <= Pow_high):
                    if Pow > Pow_high:
                        Pow = Pow_high
                        flag = 1
                    elif Pow < Pow_low:
                        Pow = Pow_low
            elif action_type == "chargefull":
                Pow = -0.5
                if not(Pow_low <= Pow <= Pow_high):
                    if Pow > Pow_high:
                        Pow = Pow_high
                    elif Pow < Pow_low:
                        Pow = Pow_low
            elif action_type == "chargehalf":
                Pow = -0.25
                if not(Pow_low <= Pow <= Pow_high):
                    if Pow > Pow_high:
                        Pow = Pow_high
                    elif Pow < Pow_low:
                        Pow = Pow_low
        if action_type == "rest":
            SOC_dif = 1/self.Eess*self.T*self.PowStBy
            self.SOC -= SOC_dif
            Rt = 0
        else:
            self.SOC -= Pow #charging when power is negative
            P_max = self.Eess/2
            # print(ct, Pow, self.alpha,P_max+0.0001)
            Rt = self.scaling_factor*((ct*Pow )/P_max) + ((flag==1)* -5)
            self.profit += ct*Pow
        self.damstep += 1
        return Rt

    def take_step(self, rtm_action, current_rtm_price, dam_action=None, dam_valid=False, current_dam_price=None):
        self.actions_rtm[self.rtmstep] = rtm_action
        reward_rtm = self._take_action_rtm(rtm_action, current_rtm_price)
        self.reward_total += reward_rtm
        self.SOCmax -= self.deg()
        done_rtm, observation_rtm = self._next_obs_rtm()
        observation_dam = None
        reward_dam = None
        done_dam = None
        if dam_valid:
            self.actions_dam[self.damstep] = dam_action
            reward_dam = self._take_action_dam(dam_action, current_dam_price)
            self.reward_total += reward_dam
            done_dam, observation_dam= self._next_obs_dam()
        return observation_rtm, reward_rtm, done_rtm,observation_dam, reward_dam, done_dam


    def reset(self):
        self.SOC = self.SOCmin #a percentage of capacity
        self.profit = 0
        self.reward_total = 0
        self.Pow = 0

        self.actions_rtm = np.zeros(self.steps_rtm) #stores real time actions that I take in real-time market
        self.actions_dam = np.zeros(self.steps_dam) #stores historical actions that I take in day ahead market
        self.rtmstep = 0
        self.damstep = 0

    def render(self, mode='human'):
        print("SOC:", self.SOC, "profit: ", self.profit)
        return(self.profit, self.SOCmax, self.reward_total, self.SOC, self.actions_rtm[self.steps_rtm-1], self.actions_dam[self.steps_dam-1])


ACTION_LOOKUP_DAM = {
    0: "dischargefull", #500 kW
    1: "dischargehalf", #250 kW
    2: "rest", #0 kW
    3: "chargehalf", #-250kW
    4: "chargefull" #-500 kW
}


ACTION_LOOKUP_RTM = {
    0: "dischargefull", #40 kW
    1: "dischargehalf", #20 kW
    2: "rest", #0 kW
    3: "chargehalf", #-40kW
    4: "chargefull" #-20 kW
}