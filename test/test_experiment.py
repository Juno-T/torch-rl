import unittest
import sys
import os
from pathlib import Path
import gym
import numpy as np
from numpy.random import default_rng
import stable_baselines3 as sb3

sys.path.insert(0, str(Path(os.path.abspath(__file__)).parent.parent))
from common import experiment
from agents.base import RandomAgent

class TestReproducibility(unittest.TestCase):

  @classmethod
  def setUpClass(cls) -> None:
    cls.rng = default_rng(42)
    return super().setUpClass()

  def setUp(self) -> None:
    return super().setUp()
  
  def test_gym_CartPole(self):
    self.env1 = gym.make('CartPole-v1')
    self.env2 = gym.make('CartPole-v1')
    for i in range(10):
      reset_seed= TestReproducibility.rng.integers(0,1e5)
      observation1 = self.env1.reset(seed=int(reset_seed))
      observation2 = self.env2.reset(seed=int(reset_seed))
      self.assertTrue(np.array_equal(observation1, observation2))

      done = False
      while not done:
        action = self.env1.action_space.sample()
        observation1, reward1, done, info = self.env1.step(action)
        observation2, reward2, _, _ = self.env2.step(action)
        self.assertTrue(np.array_equal(observation1, observation2))
        self.assertEqual(reward1, reward2)

  def test_wrapped_atari(self):
    gym_env1 = gym.make('ALE/BeamRider-v5')
    gym_env2 = gym.make('ALE/BeamRider-v5')
    self.env1 = sb3.common.atari_wrappers.AtariWrapper(gym_env1, 
                          noop_max=30, 
                          frame_skip=4, 
                          screen_size=84, 
                          terminal_on_life_loss=True, 
                          clip_reward=True)
    self.env2 = sb3.common.atari_wrappers.AtariWrapper(gym_env2, 
                          noop_max=30, 
                          frame_skip=4, 
                          screen_size=84, 
                          terminal_on_life_loss=True, 
                          clip_reward=True)
    for i in range(5):
      reset_seed= TestReproducibility.rng.integers(0,1e5)
      observation1 = self.env1.reset(seed=int(reset_seed))
      observation2 = self.env2.reset(seed=int(reset_seed))
      self.assertTrue(np.array_equal(observation1, observation2))

      done = False
      while not done:
        action = self.env1.action_space.sample()
        observation1, reward1, done, info = self.env1.step(action)
        observation2, reward2, _, _ = self.env2.step(action)
        self.assertTrue(np.array_equal(observation1, observation2))
        self.assertEqual(reward1, reward2)

  def test_Trainer(self):
    gym_env1 = gym.make('ALE/BeamRider-v5')
    gym_env2 = gym.make('ALE/BeamRider-v5')
    self.env1 = sb3.common.atari_wrappers.AtariWrapper(gym_env1, 
                          noop_max=30, 
                          frame_skip=4, 
                          screen_size=84, 
                          terminal_on_life_loss=True, 
                          clip_reward=True)
    self.env2 = sb3.common.atari_wrappers.AtariWrapper(gym_env2, 
                          noop_max=30, 
                          frame_skip=4, 
                          screen_size=84, 
                          terminal_on_life_loss=True, 
                          clip_reward=True)
    self.ab_agent1 = RandomAgent(self.env1)
    self.ab_agent2 = RandomAgent(self.env2)
    
    ep_sum1=None
    ep_sum2=None
    def onEpisodeSummary1(step, ep_sum):
      ep_sum1 = ep_sum

    def onEpisodeSummary2(step, ep_sum):
      ep_sum2 = ep_sum
    
    trainer1 = experiment.Trainer(self.env1, onEpisodeSummary=onEpisodeSummary1)
    trainer2 = experiment.Trainer(self.env2, onEpisodeSummary=onEpisodeSummary2)

    train_steps = 20
    trainer1.train(default_rng(42), self.ab_agent1, train_steps, batch_size=3, is_continue=False, learn_from_transitions=True)
    trainer2.train(default_rng(42), self.ab_agent2, train_steps, batch_size=3, is_continue=False, learn_from_transitions=True)

    self.assertTrue(trainer1.trained_step == trainer2.trained_step)
    # action1, timesteps1 = trainer1.acc.sample_one_ep(rng_key=key)
    # action2, timesteps2 = trainer2.acc.sample_one_ep(rng_key=key)
    # self.assertTrue(action1.shape==action2.shape)
    # self.assertTrue(timesteps1.obsv.shape==timesteps2.obsv.shape)
    # self.assertTrue(np.array_equal(action1, action2))
    # self.assertTrue(np.array_equal(timesteps1.obsv, timesteps2.obsv))

    if(ep_sum1 is not None) and (ep_sum2 is not None):
      self.assertTrue(ep_sum1['train']['reward'] == ep_sum2['train']['reward'])
      self.assertTrue(ep_sum1['val']['reward'] == ep_sum2['val']['reward'])


class TestTrainClass():

  @classmethod
  def setUpClass(cls) -> None:
    cls.rng = default_rng(42)
    return super().setUpClass()

  def setUp(self) -> None:
    self.cartpole = gym.make('CartPole-v1')
    gym_env = gym.make('ALE/BeamRider-v5')
    self.atari_beam = sb3.common.atari_wrappers.AtariWrapper(gym_env, 
                          noop_max=30, 
                          frame_skip=4, 
                          screen_size=84, 
                          terminal_on_life_loss=True, 
                          clip_reward=True)
    self.cp_agent = RandomAgent(self.cartpole)
    self.ab_agent = RandomAgent(self.atari_beam)
    return super().setUp()
  
  def test_train_RandomAgent(self):
    trainer = experiment.Trainer(self.cartpole, onEpisodeSummary=onEpisodeSummary)

    train_steps = 100
    trainer.train(TestTrainClass.rng, self.cp_agent, train_steps, batch_size=100, is_continue=False, learn_from_transitions=True)
    assert(1==1)
    



if __name__ == '__main__':
  unittest.main()