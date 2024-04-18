from src import game
import gym
from gym.spaces.box import Box

class env:
    def __init__(self, name, *args, **kwargs) -> None:
        self.env, self.state_space, self.action_space = self.chooseEnv(name, *args, **kwargs)
        # May be possible to change exploration_min to 0.05 and decay to 0.005 for the game frozenLake
        self.exploration_max = 1.0
        self.exploration_min = 0.001
        self.exploration_decay = 0.95

        if name == "Catcher":
            self.envType = "homemade"
        else:
            self.envType = "gym"
            
        
    def chooseEnv(self, name, *args, **kwargs):
        if name == "Catcher":
            env = game.Environment(*args, **kwargs)
            state_space = env.STATE_SPACE_SIZE
            action_space = env.ACTION_SPACE_SIZE
            return env, state_space, action_space
        else:
            env = gym.make(name, *args, **kwargs)
            if isinstance(env.observation_space, Box):
                state_space = env.observation_space.shape[0]
                print(env.observation_space.shape[0])
            else:
                state_space = env.observation_space.n
            action_space = env.action_space.n
            return env, state_space, action_space
        
    def step(self, action):
        if self.envType == "homemade":
            next_state, reward, done, score = self.env.step(action)
            return next_state, reward, done, score
        else:
            next_state, reward, done, truncated, info = self.env.step(action)
            return next_state, reward, done, 0
    
    def reset(self):
        return self.env.reset()
    
    def render(self, screen):
        if self.envType == "homemade":
            self.env.render(screen)
        else:
            self.env.render()