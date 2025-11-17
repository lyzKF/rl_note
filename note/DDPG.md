## deep deterministic policy gradient
DDPG，全称是deep deterministic policy gradient，深度确定性策略梯度算法。DDPG是解决连续控制型问题的的一种算法，DDPG的主要修改是使用确定性策略，表示为$\mu(s;\theta^{\mu})$。actor网络不再输出动作的概率分布，而是直接将状态$s$映射到动作$a$：$$a = \mu(s;\theta^{\mu})$$其中，$\theta^{\mu}$为actor的参数，标准策略梯度定理涉及对从策略中采样的动作的期望。对于确定性策略，期望消失，目标函数的梯度可以通过链式法则推导出来，这利用了critic对actor所选动作的评估。

### actor网络和critic网络
更新的时候如果更新目标在不断变动，会造成更新困难。DDPG和DQN一样，用了固定网络(fix network)技术，就是先冻结住用来求target的网络。在更新之后，再把参数赋值到target网络。DDPG维护着两个主要的神经网络，以及它们各自的目标版本：actor, critic, Actor_target, cirtic_target。
1. actor网络$\mu(s;\theta^{\mu})$，接收状态$s$作为输入，并输出一个特定的连续动作$a$。其目标是通过最大化预期的未来奖励来学习最优策略。
2. critc网络，同时接收状态$s$和动作$a$作为输入，并输出相应的Q值(从状态$s$开始，执行动作$a$，然后遵循策略$\mu$的预期回报)。其目标是准确评估当前Actor策略的动作值函数。

![](/assets/img/ddpg.png "Deep Deterministic Policy Gradient")

#### critic网络训练
critic网络的训练和DQN中的Q网络训练相似，旨在最小化均方贝尔曼误差(MSBE)。从经验回放缓冲区$D$采样一个batch_size的数据，$(s_{i}, a_{i}, r_{i}, s_{i}^{'})$是其中的一个样本数据，我们计算目标值$y_{i}$，这里的$i$代表第几个样本：$$y_{i} = r_{i} + \gamma * Q^{'}(s_{i}^{'}, \mu^{'}(s_{i}^{'};\theta^{\mu^{'}}); \theta^{Q^{'}})$$

目标值计算使用**目标Actor网络**$\mu^{'}$来选择下一个动作$a^{'}=\mu^{'}(s_{i}^{'};\theta^{\mu^{'}})$，使用目标Critic网络$Q^{'}$来评估下一个动作值函数。目标网络和训练的网络解耦，显著提升了稳定性。

critic的损失函数是目标值$y_{i}$与Critic当前估计$Q_{i}(s_{i}, a_{i};\theta^{Q})$之间的方差：$$L(\theta^{Q}) = \frac{1}{N} \sum_{i}\bigg(y_{i} - Q(s_{i}, a_{i}; \theta^{Q})\bigg)$$
此损失通过对critic参数$\theta^{Q}$的梯度下降进行最小化。

#### actor网络训练
actor网络$\mu(s;\theta^{\mu})$使用确定性策略梯度进行更新，目标是调整Actor的参数$\theta^{\mu}$，根据当前的critic产生最大化预期Q值的动作。Actor目标函数的梯度使用经验回放缓冲区中的小批量数据进行近似：$$\nabla_{\theta^{\mu}} J(\theta^{\mu}) \approx \frac{1}{N} \sum_{i}\nabla_{a}Q\bigg(s_{i}, a_{i};\theta^{Q}\bigg)|_{s=s_{i}, a=\mu(s;\theta^{\mu})} * \nabla_{\theta^{\mu}}\mu(s_{i};\theta^{\mu})$$

这看起来很复杂，但直观地讲它意味着：
1. 对于batch中的每个状态$s_{i}$，找到当前Actor策略将输出的动作$a = \mu(s_{i};\theta^{\mu})$；
2. 询问critic，如果我们稍微改变动作$a$，Q值$Q(s_{i}, a_{i};\theta^{Q})$如何变化？
3. 计算actor参数$\theta^{mu}$的变化如何影响输出动作$a$?
4. 使用链式法则结合这些梯度，以确定改变$\theta^{\mu}$将如何影响Q值。
5. 使用梯度上升更新$\theta^{\mu}$

此更新使用Critic作为评估器，引导Actor做出更好的动作，而无需直接采样动作和估计期望，使其适用于连续空间。

#### 经验回放与目标网络
DDPG是一种off-policy算法。像DQN一样，它使用了经验回放和目标网络两种机制，以提升稳定性和样本效率：
1. 经验回放，样本$(s_{t}, a_{t}, r_{t+1}, s_{t+1})$存储在一个大型经验回放缓冲区$D$中。在训练期间，会从$D$中随机采样小批量数据。这打破了连续经验之间的时间相关性，使网络能从多样化的过去经验中学习，从而带来更稳定和高效的学习。
2. 目标网络，DDPG维护着独立的、缓慢更新副本的目标网络$\mu^{'}(s;\theta^{\mu^{'}})$和$Q^{'}(s,a;\theta^{Q^{'}})$；在每次主网络更新后，会使用一种称为Polyak平均的“软”更新规则：$$\theta^{'} \leftarrow \tau*\theta + (1-\tau)*\theta^{'}$$

#### 确定性策略中的探索
由于策略$\mu(s;\theta^{\mu})$是确定性的，如果在训练期间不加干预，它将始终为给定状态输出相同的动作。这会阻碍探索，为确保智能体充分探索环境，DDPG在actor的输出动作中添加噪声，仅限于训练期间：$$a_{t} = \mu(s, \theta^{\mu}) + \mathcal{N}_{t}$$
噪声$\mathcal{N}_{t}$可以是简单的高斯噪声，噪声通常会在训练过程中衰减，在评估或部署期间，此噪声会被关闭。