import gym

from common.utils import mini_batch_train  # import training function
from td3.td3 import TD3Agent  # import agent from algorithm of interest

# Create Gym environment
env = gym.make("Pendulum-v0")

# check agent class for initialization parameters and initialize agent
gamma = 0.99
tau = 1e-2
noise_std = 0.2
bound = 0.5
delay_step = 2
buffer_maxlen = 100000
critic_lr = 1e-3
actor_lr = 1e-3

agent = TD3Agent(env, gamma, tau, buffer_maxlen, delay_step, noise_std, bound, critic_lr, actor_lr)

# define training parameters
max_episodes = 100
max_steps = 500
batch_size = 32

# train agent with mini_batch_train function
episode_rewards = mini_batch_train(env, agent, max_episodes, max_steps, batch_size)
