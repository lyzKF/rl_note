#! -*- encoding:utf-8 -*-
import gymnasium as gym
from itertools import count
import matplotlib.pyplot as plt

from deepRL_note.src.reinforce import Policy, PolicyGradient


def main_mountain_car():
    """
    """
    env = gym.make("MountainCar-v0",render_mode="human")
    env.reset(seed=543)

    policy = Policy(in_feat=env.observation_space.shape[0], hidden_feat=128, out_feat=env.action_space.n)
    RL = PolicyGradient(policy_net=policy, learning_rate=1e-2)
    
    for _ in range(1000):
        state, _ = env.reset()
        ep_reward= 0
        for _ in range(10000):
            action = RL.choose_action(state)
            state, reward, done, truncated, info = env.step(action)

            policy.rewards.append(reward)
            ep_reward += reward

            if done:
                print("hhh")
                break
            else:
                ep_reward=0

        if ep_reward !=0:
            policy_loss = RL.eposide_learning()
            print('running_reward {} | policy_loss:{:.03f}'.format(ep_reward, policy_loss))
        elif ep_reward==0:
             del policy.rewards[:]


if __name__ == '__main__':
    #
    main_mountain_car()
