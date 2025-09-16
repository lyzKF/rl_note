#! -*- encoding:utf-8 -*-
"""
"""
import torch
import random
import numpy as np
import gymnasium as gym
from src.SAC_Continuous import SACContinuous
from src.rl_utils import ReplayBuffer, train_off_policy_agent



def main_walker():
    """
    """
    algorithm_name = 'SAC_correct'
    env_name = f'./data/BipedalWalkerV3/{algorithm_name}'
    # env = gym.make(env_name)
    env = gym.make("BipedalWalker-v3", hardcore=False, render_mode="rgb_array")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high[0]  # 动作最大值

    random.seed(0)
    np.random.seed(0)
    env.reset(seed=543, options={"low": -0.1, "high": 0.1})
    torch.manual_seed(0)

    actor_lr = 4e-4
    critic_lr = 4e-4
    alpha_lr = 4e-4
    num_episodes = 1000
    hidden_dim = 256
    gamma = 0.98
    tau = 0.01  # 软更新参数
    buffer_size = 100000
    minimal_size = 1000
    batch_size = 64
    target_entropy = -env.action_space.shape[0]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    replay_buffer = ReplayBuffer(buffer_size)
    agent = SACContinuous(state_dim, hidden_dim, action_dim, action_bound,
                          actor_lr, critic_lr, alpha_lr, target_entropy, tau,
                          gamma, device)
    
    # agent.load_model('BipedalWalker-v3\SACx.pth')

    _ = train_off_policy_agent(
        env, 
        agent, 
        num_episodes, 
        replay_buffer, 
        minimal_size, 
        batch_size, 
        trick_bool=False, 
        render_bool=True,
        env_name = env_name,
        algorithm_name = algorithm_name)


if __name__ == '__main__':
    #
    main_walker()
