import collections
from typing import NamedTuple, Any
from copy import deepcopy

import numpy as np
from numpy.random import default_rng
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
    self.onEpisodeSummary = onEpisodeSummary
    
  def _reset(self):
    pass #TODO

  def train(self, 
    rng, 
    agent: Agent,
    train_steps: int, 
    batch_size: int=10, 
    evaluate_every: int=1000, 
    freeze_play: int=1000, # initial steps to collect memory without learning
    eval_episodes: int=1, 
    is_continue: bool=False,
    verbose: bool=True):

    if not is_continue:
      self._reset()
      agent.train_init(rng)
    total_steps = train_steps+agent.trained_step
    train_summary={'max_val_reward': -1}

    if verbose:
      pbar = tqdm(total=total_steps, bar_format='{l_bar}{bar:15}{r_bar}{bar:-15b}')
      pbar.update(agent.trained_step - pbar.n) 
    else:
      pbar = None
    next_eval_step=agent.trained_step+evaluate_every
    while agent.trained_step<total_steps:
      agent.episode_count+=1
      episode_summary = {'train': {}, 'val':{}, 'agent': {}, 'episode': agent.episode_count}

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
        if freeze_play<=0:
          agent.learn_batch_transitions(rng, batch_size)
        else:
          freeze_play-=1
        if agent.trained_step>=total_steps:
          break

      if agent.trained_step >= next_eval_step:
        if pbar:
          pbar.update(agent.trained_step - pbar.n) 
        next_eval_step=agent.trained_step+evaluate_every
        val_summary = self.eval(default_rng(int(rng.integers(0,1e6))), agent, eval_episodes)
        episode_summary['val'] = val_summary
        train_summary['max_val_reward']=max(episode_summary['val']['reward'],
                                            train_summary['max_val_reward'])

      if freeze_play<0:
        episode_summary['train']['reward']=acc_reward
        episode_summary['train']['ep_length']=length
        episode_summary['agent']=agent.get_stats()
        self.onEpisodeSummary(agent.trained_step, episode_summary)
    if pbar:
      pbar.close()
    return train_summary
      

  def eval(self, rng, agent, eval_episodes):
    summary={}
    summary['reward']=0
    summary['ep_length']=0
    for ep in range(eval_episodes):
      observation = self.env.reset(seed=int(rng.integers(1e5)))
      agent.episode_init(observation, train=False)
      timestep_t = TimeStep(obsv=observation)
      agent.observe(None, timestep_t, remember = False)

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
      summary['reward']+=acc_reward
      summary['ep_length']+=length

    summary['reward']/=eval_episodes
    summary['ep_length']/=eval_episodes
    return summary