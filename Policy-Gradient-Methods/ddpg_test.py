import gym

from common.utils import mini_batch_train_xiaotong, mini_batch_train
from ddpg.ddpg import DDPGAgent

import chem_env_conv3d as env
# env = gym.make("Pendulum-v0")

max_episodes = 100000
max_steps = 500
batch_size = 128

gamma = 0.99
tau = 1e-2
buffer_maxlen = 10000
critic_lr = 1e-4
actor_lr = 1e-4

agent = DDPGAgent(env, gamma, tau, buffer_maxlen, critic_lr, actor_lr)
episode_rewards = mini_batch_train(env, agent, max_episodes, max_steps, batch_size)
# episode_rewards = mini_batch_train_xiaotong(env, agent, max_episodes, max_steps, batch_size)
