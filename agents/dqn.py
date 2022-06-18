import torch
from torch import nn, optim
import torch.nn.functional as F
from copy import deepcopy
import numpy as np

from agents.base import Agent
from common.memory import ReplayMemory, Transition, NP_deque
from common.value_prediction import v_q_learning_target


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN_CNN(nn.Module):
  def __init__(self, h, w, in_channels, outputs):
    super(DQN_CNN, self).__init__()
    self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=8, stride=4)
    # self.bn1 = nn.BatchNorm2d(16)
    self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
    # self.bn2 = nn.BatchNorm2d(32)

    def conv2d_size_out(size, kernel_size = 5, stride = 2):
        return (size - (kernel_size - 1) - 1) // stride  + 1
    convw = conv2d_size_out(conv2d_size_out(w, 8, 4), 4, 2)
    convh = conv2d_size_out(conv2d_size_out(h, 8, 4), 4, 2)
    linear_input_size = convw * convh * 32
    self.head = nn.Sequential(
      nn.Linear(linear_input_size, 256),
      nn.ReLU(),
      nn.Linear(256, outputs),
    )

  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    return self.head(x.view(x.size(0), -1))

class MLP(nn.Module):
  def __init__(self, inputs, outputs):
    super(MLP, self).__init__()
    self.head=nn.Sequential(
      nn.Linear(inputs, 16),
      nn.ReLU(),
      nn.Linear(16, outputs)
    )

  def forward(self, x):
    return self.head(x)

class DQN_agent(Agent):
  """
  Parameters:
    env 
    model
    epsilon
    memory
    look_back
    eps_decay: epoch that epsilon reach minimum
    discount
    learning_rate
    delay_update
    gradclip
  """
  def __init__(self, env, model, epsilon, memory=None, look_back=1, eps_decay=5e2, eps_min=0.1, discount=0.9, learning_rate=0.1, delay_update=100, grad_clip=None):
    self.env = env
    self.state_space = env.observation_space
    self.action_space = env.action_space
    
    self.init_epsilon = epsilon
    self.memory = memory
    self.model=model
    self.look_back = look_back
    self.eps_decay = (epsilon-eps_min)/eps_decay
    self.eps_min = eps_min
    self.discount=discount
    self.learning_rate=learning_rate
    self.delay_update = delay_update
    self.grad_clip = grad_clip

    self.criterion = nn.SmoothL1Loss()
    self.episode_count = 0
    self.step_count=0
    self.internal_s_t = None
    self.short_memory = NP_deque(maxlen = look_back)
    self.recent_loss=0
    
  def train_init(self, rng=None):
    self.replay_model = deepcopy(self.model).to(device)
    self.target_model = deepcopy(self.replay_model).to(device)
    self.optimizer = optim.Adam(self.replay_model.parameters(), lr=self.learning_rate)
    self.episode_count = 0
    self.epsilon = self.init_epsilon+self.eps_decay #offset first episode

  def episode_init(self, initial_observation, train=True):
    if train:
      self.episode_count+=1
    self.internal_s_t = None
    self.short_memory.reset(element = np.zeros_like(initial_observation))
    
  def act(self, rng):
    r = rng.uniform()
    if r<self.epsilon:
      return int(rng.integers(self.action_space.n)), self.discount
    
    self.replay_model.eval()
    with torch.no_grad():
      q_t = self.replay_model(torch.tensor(self.internal_s_t).float().unsqueeze(0).to(device)).detach()
    return int(q_t.max(1).indices[0]), self.discount

  def eval_act(self, rng):
    self.replay_model.eval()
    with torch.no_grad():
      q_t = self.replay_model(torch.tensor(self.internal_s_t).float().unsqueeze(0).to(device)).detach()
    return int(q_t.max(1).indices[0]), self.discount

  
  def observe(self, action, timestep_t, remember=False):
    internal_s_tm1 = self.internal_s_t

    self.internal_s_t = self._process_observation(timestep_t.obsv)
    if remember and internal_s_tm1 is not None: # Not the first observation
      self.memory.push(Transition(
        s_tm1=internal_s_tm1,
        a_tm1=action,
        r_t=timestep_t.reward,
        s_t=self.internal_s_t,
        discount_t=timestep_t.discount
      ))

  def learn_one_ep(self, episode):
    pass

  def get_stats(self):
    return {
      'epsilon': self.epsilon,
      'loss': float(self.recent_loss)
    }

  def learn_batch_transitions(self, rng, batch_size):
    self.step_count+=1
    self.epsilon -= self.eps_decay
    self.epsilon = max(self.epsilon, self.eps_min)
    if self.step_count%self.delay_update==0:
      self.target_model.load_state_dict(self.replay_model.state_dict())
    if self.memory.size<batch_size:
      return 0
    self.replay_model.train()
    transitions = self.memory.sample(rng, batch_size)
    s_tm1 = torch.from_numpy(transitions.s_tm1).float().to(device)
    a_tm1 = torch.from_numpy(transitions.a_tm1).type(torch.int64).to(device)
    s_t = torch.from_numpy(transitions.s_t).float().to(device)
    r_t = torch.from_numpy(transitions.r_t).float().to(device)
    discount_t = torch.from_numpy(transitions.discount_t).float().to(device)

    prediction = self.replay_model(s_tm1).gather(dim=1, index=a_tm1.unsqueeze(0)).squeeze(0)

    with torch.no_grad():
      q_t = self.target_model(s_t).detach()
    targets = v_q_learning_target(r_t, q_t, discount_t)
    loss = self.criterion(prediction, targets)

    self.optimizer.zero_grad()
    loss.backward(retain_graph=False)
    if self.grad_clip is not None:
      nn.utils.clip_grad_norm_(self.replay_model.parameters(), max_norm=self.grad_clip)
    self.optimizer.step()

    self.recent_loss=loss.detach().item()
    return 0

  def _process_observation(self, observation):
    self.short_memory.push(observation)
    p = self.short_memory.get_all_ordered()
    # return np.vstack(p) # squash first two dim
    return np.vstack(p).reshape((-1,*p.shape[2:])) # faster

