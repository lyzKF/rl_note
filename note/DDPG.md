## deep deterministic policy gradient
DDPG，全称是deep deterministic policy gradient，深度确定性策略梯度算法。DDPG是解决连续控制型问题的的一种算法，DDPG的主要修改是使用确定性策略，表示为$\mu(s;\theta^{\mu})$。actor网络不再输出动作的概率分布，而是直接将状态$s$映射到动作$a$：$$a = \mu(s;\theta^{\mu})$$其中，$\theta^{\mu}$为actor的参数，标准策略梯度定理涉及对从策略中采样的动作的期望。对于确定性策略，期望消失，目标函数的梯度可以通过链式法则推导出来，这利用了critic对actor所选动作的评估。

### actor网络和critic网络
更新的时候如果更新目标在不断变动，会造成更新困难。DDPG和DQN一样，用了固定网络(fix network)技术，就是先冻结住用来求target的网络。在更新之后，再把参数赋值到target网络。DDPG维护着两个主要的神经网络，以及它们各自的目标版本：actor, critic, Actor_target, cirtic_target。
1. actor网络$\mu(s;\theta^{\mu})$，接收状态$s$作为输入，并输出一个特定的连续动作$a$。其目标是通过最大化预期的未来奖励来学习最优策略。
2. critc网络，同时接收状态$s$和动作$a$作为输入，并输出相应的Q值(从状态$s$开始，执行动作$a$，然后遵循策略$\mu$的预期回报)。其目标是准确评估当前Actor策略的动作值函数。

![](/assets/img/ddpg.png "Deep Deterministic Policy Gradient")

