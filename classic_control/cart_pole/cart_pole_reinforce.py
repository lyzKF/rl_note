#! -*- encoding:utf-8 -*-
"""
"""
import torch
import numpy as np
import gymnasium as gym
from src.REINFORCE import REINFORCE
from src.rl_utils import train_on_policy_agent


def elem_adapter(transition_dict, device):
    """
    """
    states = transition_dict['states']
    actions = transition_dict['actions']
    rewards = transition_dict['rewards']

    return states, actions, rewards, None, None


def main_cart_pole():
    """
    """
    learning_rate = 1e-3
    num_episodes = 1000
    hidden_dim = 128
    gamma = 0.98
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    algorithm_name = 'Reinforce'
    env_name = f'./data/CartPole/{algorithm_name}'
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    env.reset(seed=543, options={"low": -0.1, "high": 0.1})
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = REINFORCE(state_dim, hidden_dim, action_dim, learning_rate, gamma,device)

    _ = train_on_policy_agent(env, agent, num_episodes, render_bool=True, env_name=env_name,algorithm_name=algorithm_name, adapter=elem_adapter, device=device)


if __name__ == '__main__':
    #
    main_cart_pole()
