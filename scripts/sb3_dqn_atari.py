import gym
from stable_baselines3 import DQN

env = gym.make("Breakout-v4")

model = DQN("CnnPolicy", env, verbose=1)
model.learn(total_timesteps=100000, log_interval=10)

obs = env.reset()
sum_reward = 0
done=False
while not done:
  action, _states = model.predict(obs, deterministic=True)
  obs, reward, done, info = env.step(action)
  sum_reward+=reward
print(sum_reward)