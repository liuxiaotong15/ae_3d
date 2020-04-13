import numpy as np
import math
import gym
import torch
import multiprocessing
import random
import copy
import os

commit_id = str(os.popen('git --no-pager log -1 --oneline').read()).split(' ', 1)[0]

cpus = 16
g_env = None
g_agent = None
g_max_episodes = None
g_max_steps = None
g_batch_size = None

def mini_batch_train(env, agent, max_episodes, max_steps, batch_size):
    episode_rewards = []

    for episode in range(max_episodes):
        state = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            env.render()
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.replay_buffer.push(state, action, reward, next_state, done)
            episode_reward += reward

            if len(agent.replay_buffer) > batch_size:
                agent.update(batch_size)   

            if done or step == max_steps-1:
                episode_rewards.append(episode_reward)
                print(commit_id , " Episode " + str(episode) + ": " + str(episode_reward))
                break

            state = next_state

    return episode_rewards

def mul_thd_func(seed, return_dict):
    global g_env, g_agent, g_max_episodes, g_batch_size, g_max_steps, cpus
    env = g_env
    agent = g_agent
    max_episodes = g_max_episodes
    max_steps = g_max_steps
    batch_size = g_batch_size
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    state = env.reset()
    episode_reward = 0
    ret = []
    for step in range(max_steps):
        env.render()
        action = agent.get_action(state)
        next_state, reward, done, _ = env.step(action)
        ret.append((copy.deepcopy(state), action, reward, copy.deepcopy(next_state), done))
        state = next_state
        episode_reward += reward
        if done or step == max_steps-1:
            if seed % cpus == 0:
                # episode_rewards.append(episode_reward)
                print("Episode " + str(seed/cpus) + ": " + str(episode_reward))
                print('cur len(replay) is: ', len(agent.replay_buffer))
            break
    # return ret
    return_dict[seed] = ret

def mini_batch_train_xiaotong(env, agent, max_episodes, max_steps, batch_size):
    global g_env, g_agent, g_max_episodes, g_batch_size, g_max_steps, cpus
    # episode_rewards = []
    g_env = env
    g_agent = agent
    g_max_episodes = max_episodes
    g_max_steps = max_steps
    g_batch_size = batch_size

    manager = multiprocessing.Manager()
    for episode in range(max_episodes):
        # pool = multiprocessing.Pool(cpus)
        # ret_list = pool.map(mul_thd_func, range(episode*cpus, episode*cpus+cpus))
        p_lst = []
        return_dict = manager.dict()
        for i in range(episode*cpus, episode*cpus+cpus):
            p1 = multiprocessing.Process(target=mul_thd_func, args=(i, return_dict))
            p1.start()
            p_lst.append(p1)
        [p.join() for p in p_lst]
        ret_list = return_dict.values()
        for ret in ret_list:
            for state, action, reward, next_state, done in ret:
                agent.replay_buffer.push(state, action, reward, next_state, done)
                # print('cur len(replay) is: ', len(agent.replay_buffer))
                if len(agent.replay_buffer) > batch_size:
                    agent.update(batch_size)
                g_agent = agent

    return episode_rewards

def mini_batch_train_frames(env, agent, max_frames, batch_size):
    episode_rewards = []
    state = env.reset()
    episode_reward = 0

    for frame in range(max_frames):
        action = agent.get_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.replay_buffer.push(state, action, reward, next_state, done)
        episode_reward += reward

        if len(agent.replay_buffer) > batch_size:
            agent.update(batch_size)   

        if done:
            episode_rewards.append(episode_reward)
            print("Frame " + str(frame) + ": " + str(episode_reward))
            state = env.reset()
            episode_reward = 0
        
        state = next_state
            
    return episode_rewards

# process episode rewards for multiple trials
def process_episode_rewards(many_episode_rewards):
    minimum = [np.min(episode_reward) for episode_reward in episode_rewards]
    maximum = [np.max(episode_reward) for episode_reward in episode_rewards]
    mean = [np.mean(episode_reward) for episode_reward in episode_rewards]

    return minimum, maximum, mean
