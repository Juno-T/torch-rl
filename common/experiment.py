import collections
from typing import NamedTuple, Any
from copy import deepcopy

import numpy as np
from tqdm import tqdm

from common.memory import TimeStep
from agents.base import Agent
import time


class Trainer:
  """
    Interaction between agent and environment
  """

  def __init__(self, 
                env, 
                onEpisodeSummary = (lambda step, data: None)):
    self.env = env
    self.trained_ep = 0
    self.trained_step = 0
    self.onEpisodeSummary = onEpisodeSummary
    
  def _reset(self):
    pass #TODO

  def train(self, 
    rng, 
    agent: Agent,
    train_steps: int, 
    batch_size: int=10, 
    evaluate_every: int=1000, 
    eval_episodes: int=1, 
    is_continue: bool=False, 
    learn_from_transitions: bool=False,
    verbose: bool=True):

    if not is_continue:
      self._reset()
      self.trained_step = 0
      agent.train_init(rng)
    train_summary={'max_val_reward': -1}

    # for episode_number in tqdm(range(self.trained_ep, self.trained_ep+train_episodes), bar_format='{l_bar}{bar:15}{r_bar}{bar:-15b}'):
    if verbose:
      pbar = tqdm(total=train_steps, bar_format='{l_bar}{bar:15}{r_bar}{bar:-15b}')
    else:
      pbar = None
    episode_number=self.trained_ep
    next_eval_step=self.trained_ep+evaluate_every
    while True:
      episode_number+=1
      episode_summary = {'train': {}, 'val':{}, 'agent': {}}

      observation = self.env.reset(seed=int(rng.integers(1e5)))
      agent.episode_init(observation)
      timestep_t = TimeStep(obsv = observation)
      agent.observe(None, timestep_t, remember = True)

      done=False
      acc_reward = 0
      length = 0
      while not done:
        action, discount = agent.act(rng)
        observation, reward, done, info = self.env.step(action)
        timestep_t = TimeStep(
            int(done),
            observation,
            reward,
            discount
        )
        agent.observe(action, timestep_t, remember = True)
        acc_reward += reward
        length+=1
        if learn_from_transitions:
          agent.learn_batch_transitions(rng, batch_size)
        self.trained_step+=1
        if self.trained_step>=train_steps:
          break
      if pbar:
        pbar.update(self.trained_step - pbar.n) 
      # if episode_number%evaluate_every==0:
      if self.trained_step >= next_eval_step:
        next_eval_step=self.trained_step+evaluate_every
        val_summary = self.eval(rng, agent, eval_episodes)
        episode_summary['val'] = val_summary
        train_summary['max_val_reward']=max(episode_summary['val']['reward'],
                                            train_summary['max_val_reward'])

      episode_summary['train']['reward']=acc_reward
      episode_summary['train']['ep_length']=length
      episode_summary['agent']=agent.get_stats()
      self.onEpisodeSummary(episode_number, episode_summary)
      if self.trained_step>=train_steps:
        break
    if pbar:
      pbar.close()
    return train_summary
      

  def eval(self, rng, agent, eval_episodes):
    for ep in range(eval_episodes):
      observation = self.env.reset(seed=int(rng.integers(1e5)))
      agent.episode_init(observation, train=False)
      timestep_t = TimeStep(obsv=observation)
      agent.observe(None, timestep_t, remember = False)

      summary={}
      done=False
      acc_reward = 0
      length = 0
      while not done:
        action, discount = agent.eval_act(rng)
        observation, reward, done, info = self.env.step(action)
        timestep_t = TimeStep(
            int(done),
            observation,
            reward,
            discount
        )
        agent.observe(action, timestep_t, remember = False)
        acc_reward += reward
        length+=1
      summary['reward']=acc_reward
      summary['ep_length']=length

    return summary