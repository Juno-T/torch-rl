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
    'learning_rate': utils.round_any(10**(rng.random()*3-5), n=2), # [1e-5, 1e-2]
    'delay_update': rng.integers(500,20000),
    'look_back': 4,
    'memory_size': int(1e5),
    'epsilon': 1,
    'eps_decay': int(10**(rng.random()*1.5+4)), #[1e4, 3e5]
    'eps_min': 0.1,
    'discount': 0.99,
    'batch_size': 128,
    'grad_clip': utils.round_any(10**(rng.random()*3-1), n=2) # [1e-1,100]
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
      notes=f"Manual tuning. Train vanilla dqn on atari BreakoutNoFrameskip-v4",
      tags=["dqn", "vanilla", "atari", "Breakout", "hand-tune"],
      config=config  
    )
    def onEpisodeSummary(step, data):
      wandb.log(data=data, step=step)
  else:
    onEpisodeSummary = (lambda *_, **__: 0)
    
  # env = prep_env('ALE/BeamRider-v5')
  env = prep_env('BreakoutNoFrameskip-v4')
  # env = prep_env('PongDeterministic-v4')

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

  train_steps = int(1e6)
  train_summary = trainer.train(rng, 
                                agent, 
                                train_steps, 
                                batch_size=config['batch_size'], 
                                evaluate_every=int(1e4), 
                                eval_episodes=3, 
                                is_continue=False, 
                                learn_from_transitions=True,
                                verbose=True)
  if track:
    wandb.log(train_summary)

if __name__=='__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--trial-number', type=int, default=-1,
                      help='trial number')
  args = parser.parse_args()
  trial_number = int(args.trial_number)

  if trial_number == -1:
    config = {
      'learning_rate': 3e-5,
      'delay_update': int(1e4),
      'look_back': 4,
      'memory_size': int(1e5), # 1e6 is prolly to much
      'epsilon': 1,
      'eps_decay': int(2e5), 
      'eps_min': 0.05,
      'discount': 0.99,
      'batch_size': 32,
      'grad_clip': 10
    }
  else:
    config = get_one_hparams(default_rng(trial_number))
  main(config, trial_number, track=True)
  