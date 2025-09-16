#! -*- encoding:utf-8 -*-
from tqdm import tqdm
import numpy as np
import torch
import collections
import random
import torch
import collections
import random
import matplotlib.pyplot as plt
import os
import imageio



class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity) 

    def add(self, state, action, reward, next_state, done): 
        self.buffer.append((state, action, reward, next_state, done)) 

    def sample(self, batch_size): 
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)

    def size(self): 
        return len(self.buffer)


def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0)) 
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))


def train_on_policy_agent(env, agent, num_episodes, render_bool=False, env_name="",algorithm_name="", adapter=None, device="cpu"):
    return_list = []
    for i in range(10):
        frame_list = []
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                state, _ = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, terminal, truncated, info = env.step(action)
                    done = False
                    if terminal or truncated:
                        done = True
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                    
                    if render_bool and i_episode == int(num_episodes / 10) - 1:
                        rendered_image = env.render()
                        frame_list.append(rendered_image)

                states, actions, rewards, next_states, dones = adapter(transition_dict, device=device)
                agent.update(states, actions, rewards, next_states, dones)

                return_list.append(episode_return)
                plot_smooth_reward(return_list, 100, env_name, algorithm_name)
                
                
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
        save_video(frame_list, env_name, f"{algorithm_name}_{i}", "gif")
    return return_list


def train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size, trick_bool=False, render_bool=False,env_name:str="",algorithm_name:str="", adapter=None, device="cpu"):
    """
    """
    max_reward = -float('inf')
    return_list = []
    
    for i in range(10):
        frame_list = []
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                state, _ = env.reset()
                done = False
                step_num=0
                while not done: #and step_num <= 800
                    # state:(24,) action: (4,)
                    action = agent.take_action(state)
                    next_state, reward, terminal, truncated, info = env.step(action)

                    if terminal or truncated:
                        done = True
                    
                    episode_return += reward
                    if trick_bool:
                        reward, done = trick(reward, done)

                    if render_bool and i_episode == int(num_episodes / 10) - 1:
                        
                        rendered_image = env.render()
                        frame_list.append(rendered_image)

                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state

                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {
                            'states': b_s, 
                            'actions': b_a, 
                            'next_states': b_ns, 
                            'rewards': b_r, 
                            'dones': b_d
                        }
                        states, actions, rewards, next_states, dones = adapter(transition_dict, device=device)
                        agent.update(states, actions, rewards, next_states, dones)

                    step_num+=1

                return_list.append(episode_return)
                plot_smooth_reward(return_list, 100, env_name, algorithm_name)

                if episode_return >= max_reward:
                    max_reward = episode_return
                    # agent.save_model(env_name, algorithm_name)
                    plot_smooth_reward(return_list, 100, env_name, algorithm_name)

                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({
                        'episode': '%d' % (num_episodes/10 * i + i_episode+1), 
                        'return': '%.3f' % np.mean(return_list[-10:])}
                    )
                pbar.update(1)

        save_video(frame_list, env_name, f"{algorithm_name}_{i}", "gif")
    return return_list


def compute_advantage(gamma, lmbda, td_delta):
    """
        GAE
        https://apxml.com/zh/courses/advanced-reinforcement-learning/chapter-3-advanced-policy-gradients-actor-critic/generalized-advantage-estimation
        https://zhuanlan.zhihu.com/p/1911110735150416080
    """
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(np.array(advantage_list), dtype=torch.float)


def save_video(frames, directory="./data", filename="video", mode="gif", fps=30):
    """
    """
    os.makedirs(directory, exist_ok=True)
    filename = f"{filename}.{mode}"
    filepath = os.path.join(directory, filename)

    # 创建视频写入器
    writer = imageio.get_writer(filepath, fps=fps)
    # 将所有帧写入视频
    for frame in frames:
        writer.append_data(frame)

    # 关闭视频写入器
    writer.close()


def plot_smooth_reward(rewards,
                       window_size=100,
                       directory="./data",
                       filename="smooth_reward_plot"):

    """
    """
    os.makedirs(directory, exist_ok=True)
    filename = f"{filename}.png"
    filepath = os.path.join(directory, filename)

    # 计算滑动窗口平均值
    smoothed_rewards = np.convolve(rewards, np.ones(window_size) / window_size, mode='valid')

    # 绘制原始奖励和平滑奖励曲线
    plt.plot(rewards, label='Raw Reward')
    plt.plot(smoothed_rewards, label='Smoothed Reward')

    # 设置图例、标题和轴标签
    plt.legend()
    plt.title('Smoothed Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')

    # 保存图像
    plt.savefig(filepath)

    # 关闭图像
    plt.close()


def trick(reward:float=0, done:bool=False):
    """
    """
    if reward <= -100:
        done = True
    else:
        done = False

    if reward <= -100:
        reward = -1

    return reward, done


def Action_adapter(a,max_action):
	#from [-1,1] to [-max,max]
	return  a*max_action


def evaluate_policy(env, max_action, agent, turns = 1):
    """
    """
    total_scores = 0
    for j in range(turns):
        s, info = env.reset()
        done = False
        frame_list = []
        while not done:
            # Take deterministic actions at test time
            a = agent.take_action(s)
            act = Action_adapter(a, max_action)
            s_next, r, dw, tr, info = env.step(act)
            rendered_image = env.render()
            frame_list.append(rendered_image)
            done = (dw or tr)
            total_scores += r
            s = s_next

    return int(total_scores/turns), frame_list









         
