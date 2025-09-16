#! -*- encoding:utf-8 -*-
"""
"""
import torch
import numpy as np
import gymnasium as gym
from src.DQN import DQN, ReplayBuffer
from src.rl_utils import train_off_policy_agent


def elem_adapter(transition_dict, device):
    """
    """
    states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float).to(device)
    actions = torch.tensor(np.array(transition_dict['actions'])).view(-1, 1).to(device)
    rewards = torch.tensor(np.array(transition_dict['rewards']), dtype=torch.float).view(-1, 1).to(device)
    next_states = torch.tensor(np.array(transition_dict['next_states']), dtype=torch.float).to(device)
    dones = torch.tensor(np.array(transition_dict['dones']), dtype=torch.float).view(-1, 1).to(device)

    return states, actions, rewards, next_states, dones


def main_cart_pole():
    """
    """
    algorithm_name = 'DQN'
    env_name = f'./data/CartPole/{algorithm_name}'
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    env.reset(seed=543, options={"low": -0.1, "high": 0.1})

    lr = 2e-3
    num_episodes = 500
    hidden_dim = 128
    gamma = 0.98
    epsilon = 0.01
    target_update = 10
    buffer_size = 10000
    minimal_size = 500
    batch_size = 64
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    replay_buffer = ReplayBuffer(buffer_size)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon, target_update, device)

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
