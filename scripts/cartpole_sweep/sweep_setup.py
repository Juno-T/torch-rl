import wandb
import pprint


def sweep_setup():
  metric = {
    'name': 'max_val_reward',
    'goal': 'maximize'
  }

  parameters_dict = {
    'learning_rate': {
      'distribution': 'log_uniform_values',
      'min': 1e-5,
      'max': 3e-2
    },
    'delay_update': {
      'distribution': 'int_uniform',
      'min': 200,
      'max': 2000
    },
    'look_back': {
      'value': 1
    },
    'memory_size': {
      'value': int(1e4)
    },
    'epsilon': {
      'value': 1
    },
    'eps_decay': {
      'distribution': 'q_log_uniform_values',
      'q': 1,
      'min': 1000,
      'max': 10000
    },
    'batch_size': {
      'value': 32
    },
    'grad_clip': {
      'distribution': 'log_uniform_values',
      'min': 3e-2,
      'max': 1e2
    },
  }

  sweep_config = {
    'method': 'random'
  }
  sweep_config['metric'] = metric
  sweep_config['parameters'] = parameters_dict
  pprint.pprint(sweep_config)
  sweep_id = wandb.sweep(
    sweep_config, 
    entity="yossathorn-t",
    project="torch-rl_cartpole"
  )
  print("Initiated sweep with id:",sweep_id)

if __name__=="__main__":
  sweep_setup()