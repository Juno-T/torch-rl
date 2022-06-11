import torch
import numpy

def q_learning_target(r_t, q_t, discount_t):
  """
  target = r_t + max(q_t)
  """
  target = r_t + discount_t * torch.max(q_t)
  return target

def v_q_learning_target(r_t, q_t, discount_t):
  targets = r_t + discount_t*torch.max(q_t, axis=1).values
  return targets