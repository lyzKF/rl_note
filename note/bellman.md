### 强化学习目标
强化学习的最终目标是寻找到最佳的策略，使得从每个状态$s$开始的预期回报$E[G_{t}]$最大化，我们需要关注评估不同策略的“优劣”。

### 贝尔曼方程
假设我们有一个固定的策略$\pi$，在此状态获取不同动作的概率$\pi(a|s)$，在遵循此策略的情况下，我们应该如何判定特定状态$s$的价值呢？

通常我们用状态值$V^{\pi}(s)$来量化，其定义为从状态$s$开始，遵循策略$\pi$所获得折扣回报的期望：
$$\begin{align}
V^{\pi}(s) = & E_{\pi}{[\sum_{t=0}^{\infty}{\gamma^{t} * r_{t+1}|S_{t}=s}]} \\
= & E_{\pi}{[G_{t}|S_{t}{=s}]}
\end{align}$$

其中$G_{t}$的定义如下：
$$
\begin{align}
G_{t} = & r_{t+1} + \gamma*r_{t+2} + .... = \sum_{t=0}^{\infty}{\gamma^{t}*r_{t+1}} \\
= & r_{t+1} + \gamma * G_{t+1}
\end{align}
$$
状态值$V^{\pi}(s)$，可以改写为：
$$
\begin{align}
V^{\pi}(s_{t}) = & E_{\pi}{[G_{t}|S_{t}{=s}]} \\
= & E_{\pi}{[r_{t+1} + \gamma * G_{t+1}|S_{t}{=s}]} \\
= & E_{\pi}{[r_{t+1}|S_{t}{=s}]} + E_{\pi}{[\gamma * G_{t+1}|S_{t}{=s}]} \\
\end{align}
$$
其中第一项是即时奖励的期望，在状态$s_{t}$下，agent首先根据$\pi(a_{t}|s_{t})$选择一个动作$a_{t}$，然后根据状态转移概率$p(s_{t+1}|s_{t},a_{t})$和环境生成下一个状态$s_{t+1}$和奖励$r_{t+1}$。
$$
\begin{align}
E_{\pi}{[r_{t+1}|S_{t}{=s}]} = \sum_{a}{\pi(a_{t}|s_{t})\sum_{s_{t+1},r_{t+1}}{p(s_{t+1}|s_{t},a_{t})r_{t+1}}}
\end{align}
$$
第二项是从下一个状态$s_{t+1}$开始折扣回报的期望：
$$
\begin{align}
E_{\pi}{[G_{t+1}|S_{t}{=s}]} = \sum_{a}{\pi(a_{t}|s_{t})\sum_{s_{t+1},r_{t+1}}{p(s_{t+1}|s_{t},a_{t})V^{\pi}(s_{t+1})}}
\end{align}
$$

贝尔曼期望方程为$V^{\pi}(s_{t})$提供了递归定义。它将状态值分解为期望的即时奖励和下一个状态的期望折扣值：
$$
\begin{align}
V^{\pi}(s_{t}) = \sum_{a}{\pi(a_{t}|s_{t})\sum_{s_{t+1},r_{t+1}}{p(s_{t+1}|s_{t},a_{t})(r_{t+1} + \gamma * V^{\pi}(s_{t+1}))}}
\end{align}
$$

类似的，我们定义动作值$Q^{\pi}(s_{t},a_{t})$，它表示从状态$s_{t}$开始，采取动作$a_{t}$，遵循策略$\pi$折扣回报的期望：
$$
\begin{align}
Q^{\pi}(s_{t},a_{t}) = & E_{\pi}{[G_{t}|S_{t}=s_{t}, A_{t}=a_{t}]} \\
= & E_{\pi}{[r_{t+1} + \gamma * G_{t+1}|S_{t}=s_{t}, A_{t}=a_{t}]} \\
= & E_{\pi}{[r_{t+1}|S_{t}{=s}]} + E_{\pi}{[\gamma * G_{t+1}|S_{t}=s_{t}, A_{t}=a_{t}]} \\
\end{align}
$$
$Q^{\pi}(s,a)$的贝尔曼期望方程是：
$$
\begin{align}
Q^{\pi}(s_{t},a_{t}) = \sum_{s_{t+1},r_{t+1}}{p(s_{t+1}|s_{t},a_{t})(r_{t+1} + \gamma * V^{\pi}(s_{t+1}))}
\end{align}
$$

注意$Q^{\pi}$和$V^{\pi}$的关系：
$$V^{\pi}(s_{t}) = \sum_{a}{\pi(a_{t}|s_{t})}{Q^{\pi}(s_{t},a_{t})}$$
$$Q^{\pi}(s_{t},a_{t}) = \sum_{s_{t+1},r_{t+1}}{p(s_{t+1}|s,a)[r_{t+1} + \gamma*V^{\pi}(s_{t+1})]}$$
对于给定的策略$\pi$和状态转移$p(s_{t+1}|s_{t},a_{t})$，贝尔曼期望方程定义了一个线性方程组。如果状态空间足够小，我们可以直接求解此系统以找到与策略$\pi$对应的状态值或动作值。

### 贝尔曼最优方程
在许多强化学习问题中，主要目标不仅仅是评估某个策略，而是找到最优策略，即获得最大累积奖励的策略。

最优状态值函数：通常用$V^{*}(s_{t})$表示，代表从状态$s_{t}$开始，遵循任意策略可达到的最大预期回报，即：$$V^{*}(s_{t}) = max_{\pi}{V^{\pi}(s_{t})}$$
状态$s_{t}$的最优价值必须等于在该状态下采取最优动作$a_{t}$，然后从产生的状态$s_{t+1}$继续最优行为所获得的预期回报
$$V^{*}(s_{t}) = max_{a_{t} \in \pi(a_{t}|s_{t})}{\sum_{s_{t+1},r_{t+1}}{p(s_{t+1}|s_{t},a_{t})(r_{t+1} + \gamma * V^{*}(s_{t+1}))}}$$


最优动作值函数：通常用$Q^{*}(s_{t},a_{t})$表示，代表从状态$s_{t}$开始，采取动作$a_{t}$后，遵循最优策略可达到的最大预期回报，即：$$Q^{*}(s_{t}, a_{t}) = max_{\pi}{Q^{\pi}(s_{t}, a_{t})}$$
在状态$s_{t}$采取动作$a_{t}$的最优价值是预期即时奖励$r_{t+1}$加上在下一个状态$s_{t+1}$采取最优可能动作$a_{t}$的折扣预期价值
$$Q^{*}(s_{t},a_{t}) = \sum_{s_{t+1},r_{t+1}}{p(s_{t+1}|s_{t},a_{t})(r_{t+1} + \gamma * max_{a_{t+1} \in \pi(a_{t+1}|s_{t+1})}Q^{*}(s_{t+1},a_{t+1}))}$$

### 寻找最优策略
如果我们有$Q^{*}(s_{t},a_{t})$，最优策略就是对每个状态中的$Q^{*}$采取贪婪行为：
$$\pi^{*}(s_{t}) = arg max_{a_{t} \in \pi(a_{t},s_{t})}{Q^{*}(s_{t},a_{t})}$$
这意味着在状态$s_{t}$，最优策略$\pi^{*}$，选择能使最优动作价值函数$Q^{*}(s_{t},a_{t})$最大化的动作$a_{t}$。如果只有$V^{*}(s_{t})$，仍然可以通过利用$p(s_{t+1}|s_{t}, a_{t})$来找到最优动作：
$$\pi^{*}(s_{t}) = arg max_{a_{t} \in \pi(a_{t},s_{t})}{\sum_{s_{t+1},r_{t+1}}{p(s_{t+1}|s_{t},a_{t})(r_{t+1} + \gamma * V^{*}(s_{t+1}))}}$$


### 参考文档
https://apxml.com/zh/courses/advanced-reinforcement-learning/chapter-1-rl-foundations-revisited/bellman-optimality-review

https://apxml.com/zh/courses/intro-to-reinforcement-learning/chapter-3-estimating-value-functions/bellman-optimality-equation