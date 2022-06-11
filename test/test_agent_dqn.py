import unittest
import sys
import os
from pathlib import Path

import torch
import gym
import numpy as np
from numpy.random import default_rng

sys.path.insert(0, str(Path(os.path.abspath(__file__)).parent.parent))
from agents.dqn import MLP, DQN_CNN, DQN_agent
from agents.base import Agent
from common.memory import TimeStep, ReplayMemory

def param_is_equal(model1, model2):
  for p1, p2 in zip(model1.parameters(), model2.parameters()):
      if p1.data.ne(p2.data).sum() > 0:
          return False
  return True

class TestReproducibility(unittest.TestCase):

  @classmethod
  def setUpClass(cls) -> None:
    cls.env = gym.make('CartPole-v1')
    cls.rng = default_rng(42)
    return super().setUpClass()

  def setUp(self) -> None:
    config = {
      'eps_decay_rate':1-3e-3, 
      'learning_rate': .01,
      'delay_update':100
    }
    torch.manual_seed(0)
    model = MLP(inputs = 4, outputs= TestReproducibility.env.action_space.n)
    epsilon = 1
    self.agent1 = DQN_agent(TestReproducibility.env, 
                      model, 
                      epsilon, 
                      memory = ReplayMemory(100),
                      eps_decay_rate=config['eps_decay_rate'], 
                      learning_rate=config['learning_rate'],
                      delay_update=config['delay_update'])
    self.agent2 = DQN_agent(TestReproducibility.env, 
                      model, 
                      epsilon, 
                      memory = ReplayMemory(100),
                      eps_decay_rate=config['eps_decay_rate'], 
                      learning_rate=config['learning_rate'],
                      delay_update=config['delay_update'])
    return super().setUp()

  def test_initialization(self):
    self.agent1.train_init(default_rng(42))
    self.agent2.train_init(default_rng(42))
    is_eq = param_is_equal(self.agent1.replay_model, self.agent2.replay_model)
    self.assertTrue(is_eq)

    is_eq = param_is_equal(self.agent1.target_model, self.agent2.target_model)
    self.assertTrue(is_eq)

    self.assertTrue(all([
      self.agent1.epsilon == self.agent2.epsilon,
      self.agent1.episode_count == self.agent2.episode_count,
    ]))

  def test_action_and_memory(self):
    self.agent1.train_init(default_rng(42))
    self.agent2.train_init(default_rng(42))
    is_eq = param_is_equal(self.agent1.replay_model, self.agent2.replay_model)
    self.assertTrue(is_eq)

    observation = TestReproducibility.env.reset(seed=int(TestReproducibility.rng.integers(0,1e5)))
    timestep_t = TimeStep(obsv = observation)
    self.agent1.episode_init(observation)
    self.agent2.episode_init(observation)
    self.agent1.observe(None, timestep_t, remember = True)
    self.agent2.observe(None, timestep_t, remember = True)
    
    done = False
    while not done:
      rng_seed = TestReproducibility.rng.integers(1e5)
      action1, discount = self.agent1.act(default_rng(rng_seed))
      action2, discount = self.agent2.act(default_rng(rng_seed))
      self.assertEqual(action1, action2)
      observation, reward, done, info = self.env.step(action1)
      timestep_t = TimeStep(
          int(done),
          observation,
          reward,
          discount
      )
      self.agent1.observe(action1, timestep_t, remember = True)
      self.agent2.observe(action2, timestep_t, remember = True)
      self.assertTrue(np.array_equal(self.agent1.short_memory._storage, 
                                      self.agent2.short_memory._storage))


if __name__ == '__main__':
  unittest.main()