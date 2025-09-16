#! -*- encoding:utf-8 -*-
"""
"""

import torch
import numpy as np
import gymnasium as gym
from src.SAC import SAC
from src.rl_utils import train_off_policy_agent, ReplayBuffer


def elem_adapter(transition_dict, device):
    """
    """
    states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float).to(device)
    actions = torch.tensor(np.array(transition_dict['actions'])).view(-1, 1).to(device)  # 动作不再是float类型
    rewards = torch.tensor(np.array(transition_dict['rewards']), dtype=torch.float).view(-1, 1).to(device)
    next_states = torch.tensor(np.array(transition_dict['next_states']), dtype=torch.float).to(device)
    dones = torch.tensor(np.array(transition_dict['dones']), dtype=torch.float).view(-1, 1).to(device)

    return states, actions, rewards, next_states, dones


def main_cart_pole():
    """
    """
    actor_lr = 1e-3
    critic_lr = 1e-2
    alpha_lr = 1e-2
    num_episodes = 200
    hidden_dim = 128
    gamma = 0.98
    tau = 0.005  # 软更新参数
    buffer_size = 10000
    minimal_size = 500
    batch_size = 64
    target_entropy = -1
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    algorithm_name = 'SAC'
    env_name = f'./data/CartPole/{algorithm_name}'
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    env.reset(seed=543, options={"low": -0.1, "high": 0.1})
    np.random.seed(0)
    torch.manual_seed(0)
    replay_buffer = ReplayBuffer(buffer_size)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = SAC(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, alpha_lr, target_entropy, tau, gamma, device)

    _ = train_off_policy_agent(
        env, 
        agent, 
        num_episodes, 
        replay_buffer, 
        minimal_size, 
        batch_size, 
        trick_bool=False, 
        render_bool=True,
        env_name=env_name,
        algorithm_name=algorithm_name, 
        adapter=elem_adapter, 
        device=device
    )



if __name__ == '__main__':
    #
    main_cart_pole()
