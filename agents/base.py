from abc import ABC, abstractmethod
import time

import numpy as np

class Agent(ABC):
  def __init__(self):
    pass

  @abstractmethod
  def train_init(self):
    pass


  @abstractmethod
  def episode_init(self, initial_observation, train=True):
    pass

  @abstractmethod
  def act(self, rng):
    pass

  def eval_act(self, rng):
    return self.act()

  def observe(self, action, observation, remember=False):
    raise("Not implemented")

  def get_stats(self):
    raise("Not implemented")

  def learn_one_ep(self, episode):
    raise("Not implemented")

  def learn_batch_transitions(self, batch_size):
    raise("Not implemented")

class RandomAgent(Agent):
  def __init__(self, env, memory=None, learning_rate=0.1):
    self.env = env
    self.memory = memory
    self.state_space = env.observation_space
    self.action_space = env.action_space
    self.discount=0.9
    

  def train_init(self, rng_key):
    pass

  def episode_init(self, initial_observation, train=True):
    pass

  def act(self, rng):
    rand_action = rng.integers(0, self.action_space.n)
    return rand_action, self.discount

  def eval_act(self, rng):
    return 0, 0
  
  def observe(self, action, observation, remember=False):
    pass

  def learn_one_ep(self, episode):
    pass

  def get_stats(self):
    pass

  def learn_batch_transitions(self, rng, batch_size):
    pass