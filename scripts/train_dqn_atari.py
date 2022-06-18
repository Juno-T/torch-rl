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

def prep_env(env_name):
  env = gym.make(env_name)
  env = sb3.common.atari_wrappers.AtariWrapper(env, 
                        noop_max=30, 
                        frame_skip=4, 
                        screen_size=84, 
                        terminal_on_life_loss=True, 
                        clip_reward=True) # (84,84,1)
  env = wrapper.Transpose(env, axes=(2,0,1)) # (1,84,84)
  return env

def get_one_hparams(rng):
  hparams = {
    'learning_rate': utils.round_any(10**(rng.random()*3-4), n=2), # [1e-4, 1e-1]
    'delay_update': rng.integers(10,100), # [10,100]
    'look_back': 4,
    'memory_size': int(1e6),
    'epsilon': 1,
    'eps_decay': 1000,
    'eps_min': 0.1,
    'discount': 0.99,
    'batch_size': 128,
    'grad_clip': utils.round_any(10**(rng.random()*3-2), n=2) # [1e-2,10]
  }
  return hparams

def main(config, trial_number, track=True):
  rng = default_rng(42) # control everything in the experiment
  torch.manual_seed(rng.integers(1e5))

  if track:
    wandb.init(
      entity="yossathorn-t",
      project="torch-rl_dqn",
      # notes=f"trial#{trial_number} Train vanilla dqn on atari ALE/Breakout-v5",
      notes=f"Manual tuning. Train vanilla dqn on atari ALE/Breakout-v5",
      tags=["dqn", "vanilla", "atari", "Breakout", "hand-tune"],
      config=config  
    )
    def onEpisodeSummary(step, data):
      wandb.log(data=data, step=step)
  else:
    onEpisodeSummary = (lambda *_, **__: 0)
    
  # env = prep_env('ALE/BeamRider-v5')
  env = prep_env('ALE/Breakout-v5')

  model = DQN_CNN(h=84, 
                  w=84, 
                  in_channels=config['look_back'], 
                  outputs=env.action_space.n)

  agent = DQN_agent(env, 
                    model, 
                    config['epsilon'], 
                    memory = ReplayMemory(config['memory_size']),
                    look_back = config['look_back'],
                    eps_decay=config['eps_decay'], 
                    eps_min=config['eps_min'],
                    discount=config['discount'],
                    learning_rate=config['learning_rate'],
                    delay_update=config['delay_update'],
                    grad_clip=config['grad_clip'])

  trainer = experiment.Trainer(env, onEpisodeSummary=onEpisodeSummary)

  train_episodes = 5000
  train_summary = trainer.train(rng, 
                                agent, 
                                train_episodes, 
                                batch_size=config['batch_size'], 
                                evaluate_every=2, 
                                eval_episodes=1, 
                                is_continue=False, 
                                learn_from_transitions=True)
  if track:
    wandb.log(train_summary)

if __name__=='__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--trial-number', type=int, default=-1,
                      help='trial number')
  args = parser.parse_args()
  trial_number = int(args.trial_number)
  print("cuda device count:", torch.cuda.device_count())

  if trial_number == -1:
    config = {
      'learning_rate': 1e-5,
      'delay_update': 25,
      'look_back': 4,
      'memory_size': int(1e6),
      'epsilon': 1,
      'eps_decay': 1000, 
      'eps_min': 0.1,
      'discount': 0.99,
      'batch_size': 128,
      'grad_clip': 1
    }
  else:
    config = get_one_hparams(default_rng(trial_number))
  main(config, trial_number, track=True)
  