### actor-critic
标准策略梯度方法(例如reinforce)根据从整个回合中采样的回报$G_{t}$来更新策略参数$\theta$，梯度估计通常表示为$\nabla_{\theta}J(\theta) \approx E\bigg[\nabla_{\theta}\log\pi_{\theta}(a_{t}|s_{t})G_{t}\bigg]$。尽管是无偏的，但依赖完整的蒙特卡洛回报$G_{t}$会引入较大的方差，因为回报取决于轨迹中所有后续的动作和奖励。

![](/assets/img/actor_critic.png "Actor Critic")

actor-critic方法提供了一种有效的替代结构来降低这种高方差，学习策略引入了actor和critic，通常作为独立的神经网络实现：
- actor：负责学习和表示策略。它接收当前状态$s$作为输入，并输出动作的概率分布（对于随机策略）或一个特定动作（对于确定性策略）。actor的策略表示为$\pi_{\theta}(a|s)$，其参数为$\theta$，actor的目标是通过调整$\theta$来学习最优策略。

- critic：负责学习一个价值函数，以评估行动者选择的动作或遇到的状态。它接收状态$s$（有时也会有动作$a$）作为输入，并输出一个价值估计。常见的选择是状态价值函数$V_{\phi}(s)$或动作值函数$Q_{\phi}(s, a)$，其参数为$\phi$。critic的作用是不选择动作，而是对actor当前策略的好坏提供反馈。


actor-critic架构是actor决定做什么，critic评估做得如何。这种评估随后指导actor的更新。典型的流程如下：
- 动作选择：actor观察当前状态$s_{t}$，并根据其策略$\pi_{\theta}(a_{t}|s_{t})$，选择一个动作$a_{t}$。

- 环境交互：动作$a_{t}$在环境中执行，得到奖励$r_{t+1}$和下一个状态$s_{t+1}$。

- critic评估：critic基于数据$(s_{t}, a_{t}, r_{t+1}, s_{t+1})$来评估行动者的动作或由此产生的状态，一种常见的方法是计算时序差分误差：$\delta_{t} = r_{t+1} + \gamma * V_{\phi}(s_{t+1}) - V_{\phi}(s_{t})$。critic也可以评估动作值函数，误差为：$\delta_{t} = r_{t+1} + \gamma * Q_{\phi}(s_{t+1}, a_{t+1}) - Q_{\phi}(s_{t}, a_{t})$

- critic更新：critic更新参数$\phi$，目标是最小化$\delta_{t}$。通常通过梯度下降。目标是随着时间推移使其价值估计更加准确。

- actor更新：actor根据critic的评估方向更新其策略参数$\theta$，不再使用有噪声的蒙特卡洛回报$G_{t}$，而是使用评论者的反馈，通常是TD误差$\delta_{t}$。策略梯度更新变为如下形式：$\nabla_{\theta}J(\theta) \approx E\big[\nabla_{\theta}\log\pi_{\theta}(A_{t}|S_{t}) * \delta_{t}\big]$

这种交互循环使得actor和critic能够同时改进。critic提供更好的评估，actor根据这些评估产生更好的动作。与 REINFORCE相比，actor-critic框架的主要好处是方差降低。actor-critic方法可以在线学习，在每一步（或一小批步）之后更新actor和critic。

-------

#### 降低方差的基线
REINFORCE算法虽然是策略梯度的根本，但在实践中常因其梯度估计固有的高方差而遇到困难。策略更新依赖于在状态$s_{t}$中执行动作$a_{t}$后观察到的总回报$G_{t}$，由于$G_{t}$会根据后续的随机转移和动作而变化很大，所以产生的梯度估计$\nabla_{\theta}\log \pi_{\theta}(a_{t}|s_{t}) * G_{t}$噪声较大。这种噪声会减慢学习速度，并可能阻止策略稳定收敛到一个好的解决方案。

Actor-Critic方法通过在策略梯度计算中引入一个基线，其核心思想出乎意料地简单：从回报$G_{t}$中减去一个仅依赖于状态$s_{t}$的值，我们称之为$b(s_{t})$。修改后的策略梯度更新项变为：
$$\nabla_{\theta}\log \pi_{\theta}(a_{t}|s_{t}) * (G_{t} - b(s_{t}))$$

为什么有效呢？减去一个依赖于状态的基线并不会改变策略梯度的期望值，这意味着它不会在更新方向上引入偏差。期望梯度保持不变，但梯度估计的方差可以大幅降低。

- 如果$G_{t} > b(s_{t})$，动作$a_{t}$导致了优于平均水平的结果。该项为正，强化动作$a_{t}$。
- 如果$G_{t} < b(s_{t})$，动作$a_{t}$导致了劣于平均水平的结果。该项为正，抑制动作$a_{t}$。

通过将回报值围绕状态特定的平均值进行中心化，更新的幅度被缩小。通常选择哪种函数作为基线呢？
简单的常数基线是能有所帮助，具体函数：$$b = \frac{1}{N}{\sum_{i=1}^{N}r(\tau)}$$

一个效果更好的基线是状态值函数$V^{\pi_{\theta}}(s_{t})$，这个函数根据定义表示从状态$s_{t}$开始并遵循当前策略$\pi_{\theta}$所能获得的期望回报。使用$V(s_{t})$作为基线，更新项变为：
$$\nabla_{\theta}\log \pi_{\theta}(a_{t}|s_{t}) * (G_{t} - V(s_{t}))$$
其中$G_{t} - V(s_{t})$是优势函数$A(a_{t}, s_{t})$的一个估计，形式上：$$A(s_{t}, a_{t}) = Q(s_{t}, a_{t}) - V(s_{t})$$
其中$Q(s_{t}, a_{t})$是动作值函数，$G_{t}$是$Q(s_{t}, a_{t})$的蒙特卡洛样本估计，$G_{t} - V(s_{t})$作为优势$A(s_{t}, a_{t})$的样本估计。

理论上，在所有仅依赖于状态$s_{t}$的函数中，使用状态值函数$V(s_{t})$作为基线是最佳选择。

#### critic的出现
我们需要一种方法来估计$V(s_{t})$，将其用作基线。
- actor：负责学习和更新参数$\theta$，通常使用带优势估计的梯度上升：
$$\theta \leftarrow \theta + a * \nabla_{\theta}\log \pi_{\theta}(a_{t}|s_{t}) * A(s_{t}, a_{t})$$
- critic：负责学习状态值函数$V(s_{t};\phi)$，有时是动作值函数$Q(s_{t},a_{t};\phi)$，通常使用时序差分方法学习，critic的输出被actor用作基线$b(s_{t})$。

#### A2C and A3C
A2C以及其异步算法A3C都是AC算法架构中的高级延伸，他们通过引入优势函数显著地提升了稳定性和性能，从而降低基本策略梯度估计中存在的方差。策略梯度更新变为：
$$A(a,s) = Q(s, a) - V(s)$$
$$\nabla_{\theta}J(\theta) \approx E_{s\in d^{\pi}, a\in\pi_{\theta}}\big[\nabla_{\theta}\log\pi_{\theta}(a|s) * A(s,a)\big]$$

实际情况，我们不知道真正的$Q(a,s)$和$V(s)$，AC方法使用一个学习得到的critic $V_{\phi}(s)$来近似$V(s)$。在时间$t$时，优势函数的一个常用估计为时序差分(TD)误差：$$A^{`}(s_{t}, a_{t}) \approx r_{t+1} + \gamma*V_{\phi}(s_{t+1}) - V_{\phi}(s_{t} = \delta_{t})$$

这里，$r_{t+1} + \gamma*V_{\phi}(s_{t+1})$是$Q(s_{t}, a_{t})$的一个估计，而$V_{\phi}(s_{t})$是critic提供的基线。

critic通过最小化该误差进行训练，通常在TD误差上使用均方误差（MSE）损失：$$\mathcal{L}(\phi) = E\bigg[\big(r_{t+1} + \gamma * V_{\phi}(s_{t+1}) - V_{\phi}(s_{t})\big)^{2}\bigg]$$

A2C 和 A3C 的主要区别在于它们如何收集经验和应用这些更新。A2C 是优势演员-评论家方法的同步并行版本。其工作方式如下：
1. 并行执行actor：多个actor并行执行，每个actor在自己的环境中进行固定步数(例如T步)的交互。
2. 经验收集：每个actor收集一个轨迹片段$(s_{t}, a_{t}, r_{t+1}, s_{t+1})$，其中$t=1,...,T$。
3. 同步：所有actor暂停，并将他们收集到的数据汇集到一个batch中。
4. 优势函数计算：使用当前的critic网络$V_{\phi}$，为batch中的数据进行优势函数估算。
5. 梯度计算：使用此batch数据计算actor $\nabla_{\theta}$和critic $\nabla_{\phi}$的梯度。
6. 同步更新： 计算出的梯度（通常在并行actor中取平均）用于更新actor和critic网络。
7. 重复： actor使用更新后的网络参数继续交互。

A2C的一个特点是其同步性。所有actor在网络更新发生之前彼此等待。这通常使得GPU的运用更有效，因为GPU擅长同时处理大量数据。

#### GAE
目前存在多种估计$A(s_{t}, a_{t})$的方法，每种方法在偏差和方差之间都有各自的权衡，我们的目标是获得一个能够产生稳定且高效策略更新的估计值。其中一种方法是使用蒙特卡洛估计。我们将优势函数估计为总折扣回报$G_{t}$减去基线$V(s_{t})$：$$A_{t}^{MC} = G_{t} - V(s_{t}) = \bigg(\sum_{k=0}^{\infty}{\gamma^{k} * r_{r+k+1}} - V(s_{t})\bigg)$$

广义优势估计 (GAE) 提供了一种更精巧的方法来处理这种偏差-方差权衡。它引入了$\lambda$(其中$\lambda \in[0,1]$)，来控制偏差和方差之间的权重。GAE优势估计为多个时间步的TD误差$\delta_{t+l}$的指数加权平局值：$$A_{t}^{GAE} = \sum_{l=0}^{\infty}{(\gamma * \lambda)^{l} * \delta_{t+l}}$$

这里，$\gamma$是标准折扣因子，$是\lambda$用于调整权重。
1. $\lambda = 0$的情况下，$$A_{t}^{GAE} = \sum_{l=0}^{\infty}{(\gamma * \lambda)^{l} * \delta_{t+l}} = \delta_{t} = r_{t+1} + \gamma * V(s_{t+1}) - V(s_{t})$$，这正好是单步TD误差。它通常具有低方差。
2. $\lambda = 1$的情况下，$$A_{t}^{GAE} = \sum_{l=0}^{\infty}{(\gamma)^{l} * \delta_{t+l}} = \sum_{l=0}^{\infty}{\gamma^{l} * (r_{t+l+1} + \gamma * V(s_{t+l+1}) - V(s_{t+l}))} \approx \bigg(\sum_{l=0}^{\infty}{\gamma^{l} * r_{t+l+1}} - V(s_{t}))\bigg)$$，这基本就是无偏蒙特卡洛优势估计，它倾向于具有高方差。
3. $\lambda \in (0,1)$的情况下，会产生介于单步TD优势和蒙特卡洛优势之间的估计。

GAE 定义中的无穷和在计算上不实用，我们可以使用递归公式高效地计算它，通常从轨迹末尾或长度为$T$的经验批次的末尾开始反向计算。
一种稍微更精确的方法是从假设$A_{t}^{GAE} = 0$开始递归，反向迭代：
$$\delta_{t} = r_{t+1} + \gamma * V(s_{t+1}) - V(s_{t})$$
$$A_{t}^{GAE} = \delta_{t} + \gamma * \lambda * A_{t+1}^{GAE}$$
```
adv = 0
for t in reversed(range(T)):
    delta = rewards[t] + gamma * V[t+1] - V[t]
    adv = delta + gamma * lambda_ * adv
    advantages[t] = adv
```