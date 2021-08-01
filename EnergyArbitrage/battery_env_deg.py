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
        self.scaling_factor =  1
        self.reward = 0
        self.profit = 0
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
        # if self.profit < -200:
        #     done = True
        obs = torch.from_numpy(np.expand_dims(np.append([self.SOC],[prices]), 0)).float()
        # print("STEP: ", self.step, "\n", "obs: ", obs)
        return done, obs

    def eff(self, power=None):
        return self.nu

    def deg(self):
        DOD = abs(self.Pow)*100/self.Eess
        if self.Pow == 0:
            return self.Eess*self.eol*(1-self.p_cyc)/(365*24*10) 
        else:
            N_cyc = 0.0035*DOD**3 + 0.2215*DOD**2-132.29*DOD+10555
            return self.eol*self.p_cyc*abs(self.Pow)/(2*N_cyc)
    
    def limit_power(self,Pow_low,Pow_high,Pow):
        flag = 0
        eps = 0.25
        if not(Pow_low-eps <= Pow <= Pow_high+eps):
            flag = 1
        if not(Pow_low <= Pow <= Pow_high):
            if Pow > Pow_high:
                Pow = Pow_high
            elif Pow < Pow_low:
                Pow = Pow_low
        return Pow, flag

    # def _take_action(self, action, ct):
    #     '''
    #     ct = current price of electricity
    #     action = integer between 0 and 4 inclusive

    #     Power is in units MW
    #     '''
    #     action_type = ACTION_LOOKUP[action]
    #     # #if Power is negative, it's charging. If Power is positive, it's discharging
    #     Pow_high = min((self.SOC - self.SOCmin)*self.Eess/(self.eff()*self.T), self.Eess)
    #     Pow_low = max((self.SOC - self.SOCmax)*self.Eess/(self.eff()*self.T), -self.Eess)
    #     # Pow_high = min((self.SOC - self.SOCmin)*self.Eess/(self.eff()*self.T), self.Eess)
    #     # Pow_low = max((self.SOC - self.SOCmax)*self.Eess/(self.eff()*self.T), -self.Eess)
    #     if Pow_high < Pow_low: #if the bounds aren't possible -- which shouldnt ever come up
    #         Pow_grid = 0
    #         flag=0
    #     else:
    #         if action_type == "dischargefull":
    #             Pow_grid = self.Eess
    #             Pow_grid, flag = self.limit_power(Pow_low,Pow_high,Pow_grid)
    #         elif action_type == "dischargehalf":
    #             Pow_grid = self.Eess/2
    #             Pow_grid, flag = self.limit_power(Pow_low,Pow_high,Pow_grid)
    #         elif action_type == "chargefull":
    #             Pow_grid = -self.Eess
    #             Pow_grid, flag = self.limit_power(Pow_low,Pow_high,Pow_grid)
    #         elif action_type == "chargehalf":
    #             Pow_grid = -self.Eess/2
    #             Pow_grid, flag = self.limit_power(Pow_low,Pow_high,Pow_grid)
    #     if action_type == "rest":
    #         SOC_dif = 1/self.Eess*self.T*self.PowStBy*self.eff()
    #         self.SOC -= SOC_dif
    #         self.Pow = 0
    #         deg = self.deg()
    #         Rt = -deg*self.degcost*10/self.eol
    #     else:
    #         #self.Pow = Pow_grid + self.eff(Pow_grid)
    #         #self.loss = self.eff(Pow_grid)
    #         self.Pow = Pow_grid
    #         if self.Pow > 0:
    #             self.SOC -= (1/self.Eess*self.T*self.Pow/self.eff()) #charging when power is negative
    #         else:
    #             self.SOC -= (1/self.Eess*self.T*self.Pow*self.eff())
    #         #self.SOC -= 1/self.Eess*self.T*self.eff()*Pow_grid
    #         if self.SOC > self.SOCmax:
    #             self.SOC = self.SOCmax 
    #         if self.SOC < self.SOCmin:
    #             self.SOC = self.SOCmin
    #         P_max = self.Eess
    #         # print(ct, Pow, self.alpha,P_max+0.0001)
    #         deg = self.deg()
    #         #future_avg = sum(self.df.Hourly_USD_per_MWh[self.step+1:self.step+24])/23
    #         #past_avg = sum(self.df.Hourly_USD_per_MWh[self.step-24:self.step])/23
    #         Rt = self.scaling_factor*(ct*(Pow_grid/P_max) + ((flag==1)* -self.flag) - deg*self.degcost*10/self.eol)
    #         #((ct-past_avg)*(Pow_grid/P_max)) + ((ct-future_avg)*(Pow_grid/P_max))
    #         self.profit += ct*(Pow_grid)
    #         self.Pow = Pow_grid
    #     self.step += 1
    #     return Rt

    # def _take_action(self, action, ct):
    #     '''
    #     ct = current price of electricity
    #     action = integer between 0 and 4 inclusive

    #     Power is in units MW
    #     '''
    #     action_type = ACTION_LOOKUP[action]
    #     if False: #if the bounds aren't possible -- which shouldnt ever come up
    #         Pow_grid = 0
    #         flag=0
    #     else:
    #         if action_type == "dischargefull":
    #             Pow_grid = self.SOC*self.eff()
    #             self.SOC = 0
    #         elif action_type == "dischargehalf":
    #             new_SOC = max(self.SOC-0.5, 0)
    #             Pow_grid = (self.SOC-new_SOC)*self.eff()
    #             self.SOC = new_SOC
    #         elif action_type == "chargefull":
    #             Pow_grid =  (self.SOC-self.SOCmax)/self.eff()
    #             self.SOC = self.SOCmax
    #         elif action_type == "chargehalf":
    #             new_SOC = min(self.SOC+0.5, self.SOCmax)
    #             Pow_grid = (max(self.SOC - new_SOC, -0.5))/self.eff()
    #             self.SOC = new_SOC
    #     if action_type == "rest":
    #         SOC_dif = 1/self.Eess*self.T*self.PowStBy
    #         self.SOC -= SOC_dif
    #         self.Pow = 0
    #         deg = self.deg()
    #         Rt = -deg*self.degcost*10/self.eol
    #     else:
    #         self.Pow = Pow_grid
    #         if self.SOC > self.SOCmax:
    #             self.SOC = self.SOCmax 
    #         if self.SOC < self.SOCmin:
    #             self.SOC = self.SOCmin
    #         P_max = self.Eess
    #         deg = self.deg()
    #         #future_avg = sum(self.df.Hourly_USD_per_MWh[self.step+1:self.step+24])/23
    #         #past_avg = sum(self.df.Hourly_USD_per_MWh[self.step-24:self.step])/23
    #         Rt = self.scaling_factor*(ct*(Pow_grid/P_max) - deg*self.degcost*10/self.eol) #+ ((ct-past_avg)*(Pow_grid/P_max)) + ((ct-future_avg)*(Pow_grid/P_max))
    #         self.profit += ct*(Pow_grid)
    #     self.step += 1
    #     return Rt

    # def _take_action(self, action, ct):
    #     '''
    #     ct = current price of electricity
    #     action = integer between 0 and 4 inclusive

    #     Power is in units MW
    #     '''
    #     action_type = ACTION_LOOKUP[action]
    #     flag = 0
    #     # Pow_high = min(self.SOC - 0, 1)
    #     # Pow_low = max(self.SOC - self.SOCmax, -1)
    #     if False: #if the bounds aren't possible -- which shouldnt ever come up
    #         Pow_grid = 0
    #         flag=0
    #     else:
    #         if action_type == "dischargefull":
    #             Pow_grid = self.SOCmax*self.eff()
    #             self.Pow = Pow_grid
    #             if self.SOC == 0:
    #                 flag = 1
    #                 Pow_grid = 0
    #                 self.Pow = 0
    #             self.SOC = 0
    #         elif action_type == "chargefull":
    #             Pow_grid = -self.SOCmax/self.eff()
    #             self.Pow = Pow_grid
    #             if self.SOC > 0.75:
    #                 flag = 1
    #                 Pow_grid = 0
    #                 self.Pow = 0
    #             self.SOC = self.SOCmax
    #     if action_type == "rest":
    #         self.Pow = 0
    #         deg = self.deg()
    #         Rt = -deg*self.degcost*10/self.eol
    #     else:
    #         P_max = self.Eess
    #         deg = self.deg()
    #         Rt = self.scaling_factor*(ct*(Pow_grid/P_max) + ((flag==1)* -self.flag) - deg*self.degcost*10/self.eol)
    #         self.profit += ct*(Pow_grid)     
    #     self.step += 1
    #     return Rt

    def _take_action(self, action, ct):
            '''
            ct = current price of electricity
            action = integer between 0 and 4 inclusive

            Power is in units MW
            '''
            action_type = ACTION_LOOKUP[action]
            Pow_high = min((self.SOC - self.SOCmin)*self.Eess/self.T, self.Eess)
            Pow_low = max((self.SOC - self.SOCmax)*self.Eess/self.T, -self.Eess)
            # Pow_high = min(self.SOC - 0, 1)
            # Pow_low = max(self.SOC - self.SOCmax, -1)
            if Pow_high < Pow_low: #if the bounds aren't possible -- which shouldnt ever come up
                Pow_grid = 0
                flag=0
            else:
                if action_type == "dischargefull":
                    Pow_grid = self.Eess
                    Pow_grid, flag = self.limit_power(Pow_low,Pow_high,Pow_grid)
                    Pow_grid = Pow_grid*self.eff()
                    #Pow = Pow_grid*self.eff()
                elif action_type == "dischargehalf":
                    Pow_grid = self.Eess/2
                    Pow_grid, flag = self.limit_power(Pow_low,Pow_high,Pow_grid)
                    Pow_grid = Pow_grid*self.eff()
                    #Pow = Pow_grid*self.eff()
                elif action_type == "chargefull":
                    Pow_grid = -self.Eess
                    Pow_grid, flag = self.limit_power(Pow_low,Pow_high,Pow_grid)
                    Pow_grid = Pow_grid/self.eff()
                    #Pow = Pow_grid/self.eff()
                elif action_type == "chargehalf":
                    Pow_grid = -self.Eess/2
                    Pow_grid, flag = self.limit_power(Pow_low,Pow_high,Pow_grid)
                    Pow_grid = Pow_grid/self.eff()
                    #Pow = Pow_grid/self.eff()
            if action_type == "rest":
                SOC_dif = 1/self.Eess*self.T*self.PowStBy
                self.SOC -= SOC_dif
                self.Pow = 0
                deg = self.deg()
                Rt = -deg*self.degcost*10/self.eol
            else:
                #self.Pow = Pow_grid + self.eff(Pow_grid)
                #self.loss = self.eff(Pow_grid)
                if Pow_grid < 0:
                    self.Pow = Pow_grid*self.eff()
                else:
                    self.Pow = Pow_grid/self.eff()
                self.SOC -= (1/self.Eess*self.T*self.Pow) #charging when power is negative
                # self.SOC -= Pow_grid
                if self.SOC > self.SOCmax:
                    self.SOC = self.SOCmax 
                if self.SOC < self.SOCmin:
                    self.SOC = self.SOCmin
                P_max = self.Eess
                # print(ct, Pow, self.alpha,P_max+0.0001)
                deg = self.deg()
                #future_avg = sum(self.df.Hourly_USD_per_MWh[self.step+1:self.step+24])/23
                #past_avg = sum(self.df.Hourly_USD_per_MWh[self.step-24:self.step])/23
                #self.Pow = Pow_grid
                Rt = self.scaling_factor*(ct*(Pow_grid/P_max) + ((flag==1)* -self.flag) - deg*self.degcost*10/self.eol) #+ ((ct-past_avg)*(Pow_grid/P_max)) + ((ct-future_avg)*(Pow_grid/P_max))
                self.profit += ct*(Pow_grid)
                self.Pow = Pow_grid     
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
        self.SOC = np.random.choice([0,0.5,1])#self.SOCmin #a percentage of capacity
        self.actions = np.zeros(self.steps)
        self.step = 0
        self.profit = 0
        self.reward_total = 0
        self.Pow = 0
        self.SOCmax = 1

    def render(self, mode='human'):
        print("SOC:", "{:.4f}".format(self.SOC), "power:", "{:.4f}".format(self.Pow), "SOC_max: ", "{:.6f}".format(self.SOCmax), "action:", self.actions[self.step-1], "profit: ", "{:.4f}".format(self.profit), "price: ", self.df.Hourly_USD_per_MWh[self.step],"reward: ", "{:.4f}".format(self.reward), "deg: ", "{:.4f}".format(self.deg()*self.degcost*10/self.eol), "reward_total: ", "{:.4f}".format(self.reward_total))
        return(self.profit, self.SOCmax, self.reward_total, self.SOC, self.Pow, self.df.Hourly_USD_per_MWh[self.step])


class BatteryDynEnvDeg(gym.Env):
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
        # make this efficiency
        # round power to determine row to index into
        # floor of the difference in power
        p = abs(self.SOC - power/2)
        key_power = math.ceil(10*p)/10
        power = abs(power)
        eff_row = self.eff_matrix[key_power]
        num_power_pieces = int(power//self.size_piece)
        loss = self.size_piece*np.sum(eff_row[:num_power_pieces]) + (power -self.size_piece*num_power_pieces)*eff_row[num_power_pieces]
        return loss 

    def deg(self):
        DOD = abs(self.Pow)*100/self.Eess
        if self.Pow == 0:
            return self.Eess*self.eol*(1-self.p_cyc)/(365*24*10) 
        else:
            N_cyc = 0.0035*DOD**3 + 0.2215*DOD**2-132.29*DOD+10555
            return self.eol*self.p_cyc*abs(self.Pow)/(2*N_cyc)
    
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
            Pow_high = min((self.SOC - self.SOCmin)*self.Eess/self.T, self.Eess)
            Pow_low = max((self.SOC - self.SOCmax)*self.Eess/self.T, -self.Eess)
            # Pow_high = min(self.SOC - 0, 1)
            # Pow_low = max(self.SOC - self.SOCmax, -1)
            if Pow_high < Pow_low: #if the bounds aren't possible -- which shouldnt ever come up
                Pow_grid = 0
                flag=0
            else:
                if action_type == "dischargefull":
                    Pow_grid = self.Eess
                    Pow_grid, flag = self.limit_power(Pow_low,Pow_high,Pow_grid)
                    self.loss = self.eff(Pow_grid)
                    Pow_grid -= self.loss
                    #Pow = Pow_grid*self.eff()
                elif action_type == "dischargehalf":
                    Pow_grid = self.Eess/2
                    Pow_grid, flag = self.limit_power(Pow_low,Pow_high,Pow_grid)
                    self.loss = self.eff(Pow_grid)
                    Pow_grid -= self.loss
                    #Pow = Pow_grid*self.eff()
                elif action_type == "chargefull":
                    Pow_grid = -self.Eess
                    Pow_grid, flag = self.limit_power(Pow_low,Pow_high,Pow_grid)
                    self.loss = self.eff(Pow_grid)
                    Pow_grid -= self.loss
                    #Pow = Pow_grid/self.eff()
                elif action_type == "chargehalf":
                    Pow_grid = -self.Eess/2
                    Pow_grid, flag = self.limit_power(Pow_low,Pow_high,Pow_grid)
                    self.loss = self.eff(Pow_grid)
                    Pow_grid -= self.loss
                    #Pow = Pow_grid/self.eff()
            if action_type == "rest":
                SOC_dif = 1/self.Eess*self.T*self.PowStBy
                self.SOC -= SOC_dif
                self.Pow = 0
                deg = self.deg()
                Rt = -deg*self.degcost*10/self.eol
            else:
                self.Pow = Pow_grid + self.loss
                #self.loss = self.eff(Pow_grid)
                self.SOC -= (1/self.Eess*self.T*self.Pow) #charging when power is negative
                # self.SOC -= Pow_grid
                if self.SOC > self.SOCmax:
                    self.SOC = self.SOCmax 
                if self.SOC < self.SOCmin:
                    self.SOC = self.SOCmin
                P_max = self.Eess
                # print(ct, Pow, self.alpha,P_max+0.0001)
                deg = self.deg()
                #future_avg = sum(self.df.Hourly_USD_per_MWh[self.step+1:self.step+24])/23
                #past_avg = sum(self.df.Hourly_USD_per_MWh[self.step-24:self.step])/23
                #self.Pow = Pow_grid
                Rt = self.scaling_factor*(ct*(Pow_grid/P_max) + ((flag==1)* -self.flag) - deg*self.degcost*10/self.eol) #+ ((ct-past_avg)*(Pow_grid/P_max)) + ((ct-future_avg)*(Pow_grid/P_max))
                self.profit += ct*(Pow_grid)
                self.Pow = Pow_grid     
            self.step += 1
            return Rt

    # def _take_action(self, action, ct):
    #     '''
    #     ct = current price of electricity
    #     action = integer between 0 and 4 inclusive

    #     Power is in units MW
    #     '''
    #     action_type = ACTION_LOOKUP[action]
    #     no_loss_high = (self.SOC - self.SOCmin)*self.Eess/(self.T)
    #     no_loss_low = (self.SOC - self.SOCmax)*self.Eess/(self.T)
    #     # #if Power is negative, it's charging. If Power is positive, it's discharging
    #     Pow_high = min(no_loss_high+self.eff(no_loss_high), self.Eess)
    #     Pow_low = max(no_loss_low+self.eff(no_loss_low), -self.Eess)
    #     # Pow_high = min((self.SOC - self.SOCmin)*self.Eess/(self.eff()*self.T), self.Eess)
    #     # Pow_low = max((self.SOC - self.SOCmax)*self.Eess/(self.eff()*self.T), -self.Eess)
    #     if Pow_high < Pow_low: #if the bounds aren't possible -- which shouldnt ever come up
    #         Pow_grid = 0
    #         flag=0
    #     else:
    #         if action_type == "dischargefull":
    #             Pow_grid = self.Eess
    #             Pow_grid, flag = self.limit_power(Pow_low,Pow_high,Pow_grid)
    #         elif action_type == "dischargehalf":
    #             Pow_grid = self.Eess/2
    #             Pow_grid, flag = self.limit_power(Pow_low,Pow_high,Pow_grid)
    #         elif action_type == "chargefull":
    #             Pow_grid = -self.Eess
    #             Pow_grid, flag = self.limit_power(Pow_low,Pow_high,Pow_grid)
    #         elif action_type == "chargehalf":
    #             Pow_grid = -self.Eess/2
    #             Pow_grid, flag = self.limit_power(Pow_low,Pow_high,Pow_grid)
    #     if action_type == "rest":
    #         SOC_dif = 1/self.Eess*self.T*self.PowStBy
    #         self.SOC -= SOC_dif
    #         self.Pow = 0
    #         deg = self.deg()
    #         Rt = -deg*self.degcost
    #     else:
    #         self.Pow = Pow_grid + self.eff(Pow_grid)
    #         #self.loss = self.eff(Pow_grid)
    #         self.SOC -= (1/self.Eess*self.T*self.Pow) #charging when power is negative
    #         #self.SOC -= 1/self.Eess*self.T*self.eff()*Pow_grid
    #         if self.SOC > self.SOCmax:
    #             self.SOC = self.SOCmax 
    #         if self.SOC < self.SOCmin:
    #             self.SOC = self.SOCmin
    #         P_max = self.Eess
    #         # print(ct, Pow, self.alpha,P_max+0.0001)
    #         deg = self.deg()
    #         future_avg = sum(self.df.Hourly_USD_per_MWh[self.step+1:self.step+24])/23
    #         past_avg = sum(self.df.Hourly_USD_per_MWh[self.step-24:self.step])/23
    #         Rt = self.scaling_factor*(ct*(Pow_grid/P_max) + ((flag==1)* -self.flag) - deg*self.degcost)
    #         #((ct-past_avg)*(Pow_grid/P_max)) + ((ct-future_avg)*(Pow_grid/P_max))
    #         self.profit += ct*(Pow_grid)
    #         self.Pow = Pow_grid
    #     self.step += 1
    #     return Rt

    def take_step(self, action, current_price):
        self.actions[self.step] = action
        reward = self._take_action(action, current_price)
        self.SOCmax -= self.deg()
        self.reward_total += reward
        self.reward = reward
        done, observation = self._next_obs()
        return observation, reward, done


    def reset(self):
        self.SOC = np.random.choice([0,0.5,1]) #self.SOCmin #a percentage of capacity
        self.actions = np.zeros(self.steps)
        self.step = 0
        self.profit = 0
        self.reward_total = 0
        self.Pow = 0
        self.SOCmax = 1

    def render(self, mode='human'):
        print("SOC:", "{:.4f}".format(self.SOC), "power:", "{:.4f}".format(self.Pow), "SOC_max: ", "{:.6f}".format(self.SOCmax), "action:", self.actions[self.step-1], "profit: ", "{:.4f}".format(self.profit), "price: ", self.df.Hourly_USD_per_MWh[self.step],"reward: ", "{:.4f}".format(self.reward), "deg: ", "{:.4f}".format(self.deg()*self.degcost*10/self.eol), "reward_total: ", "{:.4f}".format(self.reward_total), "loss", "{:.3f}".format(self.loss))
        return(self.profit, self.SOCmax, self.reward_total, self.SOC, self.Pow, self.df.Hourly_USD_per_MWh[self.step])


ACTION_LOOKUP = {
    0: "dischargefull", #1 MW -- max(lowerbound, -500)
    1: "dischargehalf", #500 kW
    2: "rest", #0 kW
    3: "chargehalf", #-500kW
    4: "chargefull" #-1 MW
}

# ACTION_LOOKUP = {
#     0: "dischargefull", #1 MW -- max(lowerbound, -500)
#     1: "chargefull", #500 kW
#     2: "rest", #0 kW
# }
