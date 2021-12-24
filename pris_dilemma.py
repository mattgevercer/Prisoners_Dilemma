#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 15:38:35 2021

@author: mattgevercer
"""

import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
#payoffs in order: [(coop,coop),(coop, rat), (rat, coop),(rat,rat)]

payoffs = [(-1,-1),(-3,0),(0,-3),(-2,-2)]

class prisoners_dilemma(gym.Env):
   # metadata = {'render.modes': ['human']}
    def __init__(self):
        self.action_space = spaces.Discrete(2) #1 for cooperate and 0 for defect
        self.observation_space = spaces.Discrete(4)
        self.steps = 1
        
    def step(self, action1, action2):
        out = [payoffs, True, {}]
        if action1 == action2 == 1:
            out.insert(1, payoffs[0])
        elif action1 == 1 and action2 == 0:
            out.insert(1, payoffs[1])
        elif action1 == 0 and action2 == 1:
            out.insert(1, payoffs[2])
        elif action1 == action2== 0:
            out.insert(1, payoffs[3])
        self.steps += 1
        return out[1]
        
class EpsilonGreedyAgent():
    def __init__(self, epsilon, env): #initialize with environment variable
        self.FirstStep = True    
        self.epsilon = epsilon
        self.envionment = env
        self.q_dict = {}
        self.act_dict = {}
        self.prev_action = None
        self.prev_reward = None #this will be updated with env.step(action)
    
    def action(self):   
        if self.FirstStep == True: #random action for the first step
            act = self.envionment.action_space.sample()
            self.FirstStep = False
            self.prev_action = act
            return act
        else: #choose action based on E-Greedy algo
            if np.random.uniform() > self.epsilon:
                self.prev_action = max(self.q_dict,key=self.q_dict.get)
                return max(self.q_dict,key=self.q_dict.get)
            else:
                self.prev_action = self.envionment.action_space.sample()
                return self.envionment.action_space.sample()
   
    def update(self, reward, steps):
        if self.prev_action not in self.q_dict.keys(): #add new states to q_dict
            self.q_dict.update({self.prev_action : reward}) 
            self.act_dict.update({self.prev_action: 1})#count action
        else:
            self.act_dict[self.prev_action]+=1 #count actions
            #implement q-update rule
            self.q_dict[self.prev_action] = self.q_dict[self.prev_action] + (
                reward - self.q_dict[self.prev_action]
                )/self.act_dict[self.prev_action]


env = prisoners_dilemma() #instantiate environment
env.action_space.seed(0)
np.random.seed(1)
player1 = EpsilonGreedyAgent(0.25, env)
player2 = EpsilonGreedyAgent(0.25, env)
hist1 =[]
hist2 =[]
for t in range(1000):
    action1 = player1.action()
    action2 = player2.action()
    player1.update(env.step(action1,action2)[0], env.steps)
    player2.update(env.step(action1,action2)[1], env.steps-1)
    player1.epsilon /= 1.002
    player2.epsilon /= 1.002
    hist1.append((action1,env.step(action1,action2),player1.epsilon))
    hist2.append((action2,env.step(action1,action2)))

#Build Plot
action1 = np.array([i[0] for i in hist1])
action2 = np.array([i[0] for i in hist2])
prop_coop1 = np.sum(action1.reshape(-1, 100), axis=1)/100
prop_coop2 = np.sum(action2.reshape(-1, 100), axis=1)/100

N=10
ind = np.arange(N)
fig = plt.figure()
ax = fig.add_subplot(111)
width = 0.30 
rects1 = ax.bar(ind, prop_coop1, width)
rects2 = ax.bar(ind+width, prop_coop2, width)
ax.set_xticks(ind+0.1)
ax.set_xticklabels(('0-1','1-2','2-3','3-4','4-5','5-6','6-7','7-8','8-9','9-10'))
ax.set_ylabel('Proportion of Cooperative Actions')
ax.set_xlabel('Episodes (in 100s)')
ax.legend( (rects1[0], rects2[0]), ('Player 1', 'Player 2') )
plt.style.use('fivethirtyeight')
plt.show()