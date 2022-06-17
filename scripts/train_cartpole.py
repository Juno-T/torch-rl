import gym
from collections.abc import Iterable
import stable_baselines3 as sb3
import sys
import os
from pathlib import Path
import torch
import numpy as np
from numpy.random import default_rng
import wandb

sys.path.insert(0, str(Path(os.path.abspath(__file__)).parent.parent))
from agents.dqn import MLP, DQN_CNN, DQN_agent
from common.memory import ReplayMemory
from common import wrapper, experiment, utils
import argparse

def onEpisodeSummary(step, data):
  wandb.log(data=data, step=step)
  # pass

def prep_env(env_name):
  env = gym.make('CartPole-v1')
  env = wrapper.Normalize(env, mean=np.array([0,0,0,0]), sd=np.array([2.4, 10, 0.42, 10]))
  return env

# def get_one_hparams(rng):
  # hparams = {
  #   'eps_decay': 1-5e-3, 
  #   'learning_rate': utils.round_any(10**(rng.random()*3-4), n=2), # [1e-4, 1e-1]
  #   'delay_update': rng.integers(10,100), # [10,100]
  #   'look_back': 4,
  #   'memory_size': 1000,
  #   'epsilon': 1,
  #   'batch_size': 100,
  #   'grad_clip': utils.round_any(10**(rng.random()*3-2), n=2) # [1e-2,10]
  # }
  # return hparams

def main(config, trial_number):
  rng = default_rng(42) # control everything in the experiment
  torch.manual_seed(rng.integers(1e5))

  wandb.init(
    entity="yossathorn-t",
    project="torch-rl_cartpole",
    notes=f"Manual tuning",
    tags=["dqn", "vanilla", "cartpole", "replicate"],
    config=config
  )
  env = prep_env('CartPole-v1')

  model = MLP(inputs = 4, outputs = env.action_space.n)

  agent = DQN_agent(env, 
                    model, 
                    config['epsilon'], 
                    memory = ReplayMemory(config['memory_size']),
                    look_back = config['look_back'],
                    eps_decay=config['eps_decay'], 
                    learning_rate=config['learning_rate'],
                    delay_update=config['delay_update'],
                    grad_clip=config['grad_clip'])

  trainer = experiment.Trainer(env, onEpisodeSummary=onEpisodeSummary)

  train_episodes = 5000
  trainer.train(rng, 
                agent, 
                train_episodes, 
                batch_size=config['batch_size'], 
                evaluate_every=2, 
                eval_episodes=1, 
                is_continue=False, 
                learn_from_transitions=True)

if __name__=='__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--trial-number', type=int, default=-1,
                      help='trial number')
  args = parser.parse_args()
  trial_number = int(args.trial_number)
  print("cuda device count:", torch.cuda.device_count())

  if trial_number == -1:
    config = {
      'learning_rate': 1e-4,
      'delay_update': 25,
      'look_back': 1,
      'memory_size': 1e6,
      'epsilon': 1,
      'eps_decay':, 
      'batch_size': 32,
      'grad_clip': 1e6 # no clip
    }
  else:
    config = get_one_hparams(default_rng(trial_number))
  main(config, trial_number)
  