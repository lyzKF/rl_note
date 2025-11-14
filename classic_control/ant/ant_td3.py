#! -*- encoding:utf-8 -*-
"""
"""
import argparse
import random
import gymnasium as gym
from tqdm import tqdm
import torch
import numpy as np
from src.TD3 import TD3
from src.rl_utils import save_video, plot_smooth_reward


'''Hyperparameter Setting'''
parser = argparse.ArgumentParser()
parser.add_argument('--dvc', type=str, default='cpu', help='running device: cuda or cpu')
parser.add_argument('--EnvIdex', type=int, default=4, help='PV1, Lch_Cv2, Humanv4, HCv4, BWv3, BWHv3')
parser.add_argument('--render', type=bool, default=False, help='Render or Not')
parser.add_argument('--Loadmodel', type=bool, default=False, help='Load pretrained model or Not')
parser.add_argument('--ModelIdex', type=int, default=30, help='which model to load')

parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--update_every', type=int, default=50, help='training frequency')
parser.add_argument('--Max_train_steps', type=int, default=int(1e3), help='Max training steps')
parser.add_argument('--save_interval', type=int, default=int(1e5), help='Model saving interval, in steps.')
parser.add_argument('--eval_interval', type=int, default=int(2e3), help='Model evaluating interval, in steps.')

parser.add_argument('--delay_freq', type=int, default=1, help='Delayed frequency for Actor and Target Net')
parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
parser.add_argument('--net_width', type=int, default=256, help='Hidden net width, s_dim-400-300-a_dim')
parser.add_argument('--a_lr', type=float, default=1e-4, help='Learning rate of actor')
parser.add_argument('--c_lr', type=float, default=1e-4, help='Learning rate of critic')
parser.add_argument('--batch_size', type=int, default=256, help='batch_size of training')
parser.add_argument('--explore_noise', type=float, default=0.15, help='exploring noise when interacting')
parser.add_argument('--explore_noise_decay', type=float, default=0.998, help='Decay rate of explore noise')
opt = parser.parse_args()
opt.dvc = torch.device(opt.dvc) # from str to torch.device



def main_walker():
    """
    """
    algorithm_name = 'TD3'
    env_name = f'./data/AntV5/{algorithm_name}'
    env = gym.make("Ant-v5", render_mode="rgb_array")

    random.seed(0)
    np.random.seed(0)
    env.reset(seed=543, options={"low": -0.1, "high": 0.1})
    torch.manual_seed(0)

    opt.state_dim = env.observation_space.shape[0]
    opt.action_dim = env.action_space.shape[0]
    opt.max_action = float(env.action_space.high[0])   #remark: action space【-max,max】
    opt.max_e_steps = env._max_episode_steps

    agent = TD3(**vars(opt)) # var: transfer argparse to dictionary

    # agent.load('td3','9')
    # total_scores = 0
    # s, info = env.reset()
    # done = False
    # while not done:
    #     # Take deterministic actions at test time
    #     act = agent.select_action(s, deterministic=True)
    #     s_next, r, dw, tr, info = env.step(act)
    #     done = (dw or tr)
    #     total_scores += r
    #     s = s_next
    # print(total_scores)


    return_list=[]
    minimal_size=1000
    num_episodes = opt.Max_train_steps
    for i in range(10):
        frame_list=[]
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:

            for i_episode in range(int(num_episodes/10)):
                s, _ = env.reset()  # Do not use opt.seed directly, or it can overfit to opt.seed
                done = False
                episode_return=0
                while not done:
                    # if i_episode < (10*opt.max_e_steps): a = env.action_space.sample() # warm up
                    # else: a = agent.select_action(s, deterministic=False)
                    a = agent.select_action(s, deterministic=False)
                    s_next, r, dw, tr, info = env.step(a)

                    done = (dw or tr)
                    episode_return += r
                    agent.replay_buffer.add(s, a, r, s_next, dw)
                    s = s_next

                    if agent.replay_buffer.size >= minimal_size:
                        agent.train()
                
                    if i_episode == int(num_episodes / 10) - 1:
                        rendered_image = env.render()
                        frame_list.append(rendered_image)

                return_list.append(episode_return)
                plot_smooth_reward(return_list, 100, env_name, algorithm_name)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({
                        'episode': '%d' % (num_episodes/10 * i + i_episode+1), 
                        'return': '%.3f' % np.mean(return_list[-10:])}
                )
                pbar.update(1)

        save_video(frame_list, env_name, f"{algorithm_name}_{i}", "gif")

    env.close()


if __name__ == '__main__':
    #
    main_walker()
