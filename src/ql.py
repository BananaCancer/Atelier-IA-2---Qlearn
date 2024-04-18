import numpy as np
import tensorflow as tf
import random

class QLearning:
    def __init__(self, state_size, action_size, exploration_max=1.0, 
                 exploration_min=0.01, exploration_decay=0.0005, 
                 learning_rate=0.7, gamma=0.95):
        self.state_size = state_size
        self.action_size = action_size
        
        # Bellman coefficients
        self.learning_rate = learning_rate
        self.gamma = gamma

        # Exploration vars
        self.exploration_rate = exploration_max
        self.exploration_max = exploration_max
        self.exploration_min = exploration_min
        self.exploration_decay = exploration_decay

        self.q_table = self._build_table()
        # May need to change function to be updateExplorationRateGym for the frozenlake game
        self.exploration_function = self.updateExplorationRateCatcher
        
            

    def _build_table(self):
        table = np.zeros((self.state_size, self.action_size))
        return table
    def updateExplorationRate(self, episode):
        self.exploration_function(episode)

    def updateExplorationRateGym(self, episode):
        self.exploration_rate = self.exploration_min + (self.exploration_max - self.exploration_min)*np.exp(-self.exploration_decay*episode)

    def updateExplorationRateCatcher(self, episode):
        self.exploration_rate = self.exploration_max*self.exploration_decay**episode
        self.exploration_rate = max(self.exploration_min, self.exploration_rate)

    def act(self, state, epsilon=None):
        if not isinstance(state, int):
            if len(state) == 2:
                state = state[0]
            else:
                print(state)
        if epsilon == None:
            epsilon = self.exploration_rate
        if np.random.uniform(0, 1) > epsilon:
            return np.argmax(self.q_table[state])
        else:
            return random.randrange(self.action_size)
    
    def update(self, state, action, new_state, reward):
        if not isinstance(state, int):
            if len(state) == 2:
                state = state[0]
            else:
                print(state)
        
        self.q_table[state][action] = self.q_table[state][action] + self.learning_rate * (reward + self.gamma * np.max(self.q_table[new_state]) - self.q_table[state][action])
    
    def load(self, name):
        self.q_table = np.load(name)

    def save(self, name):
        np.save(name, self.q_table)