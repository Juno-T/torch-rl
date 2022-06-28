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

run_id="test-resume-trial2"

def onEpisodeSummary(step, data):
  wandb.log(data=data, step=step)
  # pass

def prep_env(env_name):
  env = gym.make('CartPole-v1')
  env = wrapper.Normalize(env, mean=np.array([0,0,0,0]), sd=np.array([2.4, 10, 0.42, 10]))
  return env

def get_one_hparams(rng):
  hparams = {
    'learning_rate': utils.round_any(10**(rng.random()*3-5), n=2), # [1e-5, 1e-2]
    'delay_update': rng.integers(500,20000),
    'look_back': 1,
    'memory_size': int(1e5),
    'epsilon': 1,
    'eps_decay': int(10**(rng.random()*1.5+4)), #[1e4, 3e5]
    'eps_min': 0.1,
    'discount': 0.99,
    'batch_size': 128,
    'grad_clip': utils.round_any(10**(rng.random()*3-1), n=2) # [1e-1,100]
  }
  return hparams

def main(config=None):
  # If called by wandb.agent, as below,
  # this config will be set by Sweep Controller
  rng = default_rng(42) # control everything in the experiment
  torch.manual_seed(rng.integers(1e5))

  wandb.init(
    id=run_id,
    resume="allow",
    entity="yossathorn-t",
    project="torch-rl_cartpole",
    notes=f"Test sweep",
    tags=["dqn", "vanilla", "cartpole", "sweep"],
    config=config
  )
  config=wandb.config
  ckp_name = "agent_ckp.pt"
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
  
  if wandb.run.resumed:
    ckp_file = wandb.restore(ckp_name)
    agent.load(ckp_file.name)

  trainer = experiment.Trainer(env, onEpisodeSummary=onEpisodeSummary)

  train_steps = 10000
  train_summary = trainer.train(rng, 
                                agent, 
                                train_steps, 
                                batch_size=config['batch_size'], 
                                evaluate_every=1000, 
                                eval_episodes=3, 
                                freeze_play=1000,
                                is_continue=wandb.run.resumed)
  wandb.log(train_summary)
  agent.save(os.path.join(wandb.run.dir, ckp_name))

if __name__=='__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--trial-number', type=int, default=-1,
                      help='trial number')
  parser.add_argument('--sweep-id', type=str, default="",
                      help='sweep id. If exist, supersede the trial number')
  args = parser.parse_args()
  trial_number = int(args.trial_number)
  sweep_id = str(args.sweep_id)
  print("cuda device count:", torch.cuda.device_count())

  if trial_number == -1:
    config = {
      'learning_rate': 1e-4,
      'delay_update': 200,
      'look_back': 1,
      'memory_size': int(1e4),
      'epsilon': 1,
      'eps_decay': 5000, 
      'batch_size': 32,
      'grad_clip': 1.0
    }
  else:
    config = get_one_hparams(default_rng(trial_number))
  
  if sweep_id!="":
    wandb.agent(sweep_id, 
      function=main, 
      entity="yossathorn-t",
      project="torch-rl_cartpole",
      count=1)
  else:
    main(config)
  