import collections
from typing import NamedTuple, Any

import numpy as np
import random

class TimeStep(NamedTuple):
  step_type: int = 0  # is terminal step
  obsv: Any = None
  reward: float = 0.0
  discount: float = 1.0

class Transition(NamedTuple):
  s_tm1: Any
  a_tm1: Any
  s_t: Any
  r_t: float = 0.0
  discount_t: float = 0.9
  trace: Any = None # For future implementation of traces?
  priority: float =1 # For future priority sampling implementation

class NP_deque():
  """
    Works, faster indexing than for loop over deque.
    But isn't faster than deque in Accumulator since the bottle neck is `multimap`
  """
  def __init__(self, maxlen):
    self.maxlen = maxlen
    self.reset()

  def reset(self, element=None):
    self._storage = np.array([element]*self.maxlen, dtype=object if element is None else element.dtype)
    self._head = -1

  def push(self, element):
    self._head+=1
    self._storage[self._head%self.maxlen] = element

  def at(self, index):
    """
    support both single and multiple indexing
    """
    index = np.asarray(index)
    return self._storage[index]

  def get_random_batch(self, rng, batch_size): # rng: np.random.Generator
    assert(self._head+1>0)
    indices = np.asarray(rng.integers(0, self.size, size = self.size))
    return self._storage[indices]

  def get_all_ordered(self):
    return np.roll(self._storage, -(self._head+1), axis=0)

  @property
  def size(self):
    return min(self._head+1, self.maxlen)

class ReplayMemory(object):

  def __init__(self, capacity):
    self.memory = collections.deque([],maxlen=capacity)

  def push(self, transition):
    """Save a transition"""
    self.memory.append(transition)

  def sample(self, rng, batch_size):
    random.seed(rng.integers(1e5))
    transitions = random.choices(self.memory, k = batch_size) # rigged for np
    return self._unroll(np.array(transitions, dtype=object))

  def _unroll(self, transitions):
    return Transition(*list(map(lambda *ts: np.stack(ts),*transitions)))

  def at(self, index):
    """
    support both single and multiple indexing
    """
    index = np.asarray(index)
    return self.memory[index]

  def __len__(self):
    return len(self.memory)
  
  @property
  def size(self):
    return len(self.memory)