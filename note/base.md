![](/assets/img/reinforce_learning.png "Reinforce Learning")

### 状态
所有状态的集合称为状态空间，表示为$S = \{s_{1}, s_{2}, ...\}$

### 动作
所有动作的集合称为动作空间，表示为$A = \{a_{1}, a_{2}, ...\}$。当执行一个动作时，智能体可以从一个状态转移到另一个状态，这样的过程称为状态转移。
$$s_{t}\overset{a_{t}}{\rightarrow} s_{t+1}$$

### 策略
策略(policy)会告诉agent在每一个状态下应该采取什么样的动作，策略可以通过条件概率描述。常用$\pi(a|s)$来表示在状态$s$采取动作$a$的概率。

### 奖励
在一个状态执行一个动作后，agent会获得奖励$r$，它是状态$s$和动作$a$的函数$r(s,a)$。奖励分为即时奖励和未来奖励，即时奖励是在初始状态执行动作后立刻获得的奖励；未来奖励是指离开初始状态后获得的奖励之和。

### 轨迹
一条轨迹(trajectory)指的是一个“状态-动作-奖励”的链条。
$$
\begin{CD}
    s_{t} @>{{a_{t}}}>> s_{t+1},r_{t+1} @>{{a_{t+1}}}>> s_{t+2},r_{t+2} @>{{a_{t+2}}}>> s_{t+3}, r_{t+3}
\end{CD}
$$

### 回报
沿着一条轨迹，agent会得到一系列的即时奖励，这些即时奖励之和被称为回报(return)。为了覆盖到轨迹无限长的情况，通常会引入折扣因子$\gamma \in (0,1)$。

agent的目标是最大化预期回报，即从时间步$t$开始的折现奖励的累积和。回报$G_{t}$定义为：
$$G_{t} = r_{t+1} + \gamma*r_{t+2} + .... = \sum_{t=0}^{\infty}{\gamma^{t}*r_{t+1}}$$
其中，$\gamma$接近0，表示重视近期奖励，$\gamma$接近1，表示重视远期奖励。

####

强化学习中的核心任务是找到一个策略$\pi$，使得从每个状态$s$开始的预期回报$E[G_{t}]$最大化。


### 参考内容
https://rail.eecs.berkeley.edu/deeprlcourse/

https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning