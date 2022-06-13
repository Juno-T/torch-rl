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

def random_transition(rng):
  action = random.normal()
  step_type = rng.choice(np.array([0,1]), p=np.array([0.8,0.2]))
  obsv = rng.normal(size=(2,3,))
  reward = rng.normal()
  discount = rng.normal()
  timestep = TimeStep(step_type, obsv, reward, discount)
  return action, timestep

def transition_i(i, termination=False):
  action = 1
  step_type = 1 if termination else 0
  obsv = np.array([[i,i,i],[i,i,i]])
  reward = 1
  discount = 0
  timestep = TimeStep(step_type, obsv, reward, discount)
  return action, timestep

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

class TestFunctionality(unittest.TestCase):
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
    model = MLP(inputs = 4, outputs= TestFunctionality.env.action_space.n)
    epsilon = 1
    self.look_back = 2
    self.cp_agent = DQN_agent(TestFunctionality.env, 
                      model, 
                      epsilon, 
                      memory = ReplayMemory(100),
                      look_back = self.look_back,
                      eps_decay_rate=config['eps_decay_rate'], 
                      learning_rate=config['learning_rate'],
                      delay_update=config['delay_update'])
    return super().setUp()

  def test_det0(self):
    action, timestep = transition_i(-1)
    self.cp_agent.train_init(TestFunctionality.rng)
    self.cp_agent.episode_init(timestep.obsv)
    self.cp_agent.observe(None, timestep, remember=True)

    num_trial = 5
    for i in range(num_trial):
      self.cp_agent.observe(*transition_i(i), remember=True)
      self.assertTrue(np.sum(self.cp_agent.internal_s_t)==6*(i+i-1))
      self.assertTrue(self.cp_agent.short_memory.size==min(i+2, self.look_back))

    self.cp_agent.observe(*transition_i(num_trial, termination=True))
    self.assertTrue(np.sum(self.cp_agent.internal_s_t)==6*(num_trial*2-1))
    
    self.assertTrue(self.cp_agent.memory.size==num_trial)
    self.assertTrue(np.sum(self.cp_agent.memory.at(0).s_tm1)==-6)
    self.assertTrue(np.sum(self.cp_agent.memory.at(1).s_t)==6)
    # print(self.cp_agent.memory.sample(TestFunctionality.rng, 1))
    self.assertTrue(self.cp_agent.memory.sample(TestFunctionality.rng, 10).s_tm1.shape==(10, self.look_back*2, 3))

    # test multiple episode

class TestModels(unittest.TestCase):
  def test_DQN_CNN_forward(self):
    batch = 2
    h, w, in_channels, outputs = 84, 84, 4, 10
    model = DQN_CNN(h, w, in_channels, outputs)
    X = torch.rand((batch, in_channels, h, w))
    y = model(X)
    assert(y.shape == (batch, outputs))

if __name__ == '__main__':
  unittest.main()