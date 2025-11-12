## Trust Region Policy Optimization
信赖域策略优化，一个基于策略梯度方法的算法，可以在离散的和连续的环境中使用。

### 算法原理
考虑一个带折扣的有限MDP过程$(S,A,P,r,\rho_{0}, \gamma)$。定义$\pi \in [0,1]$为在状态-动作下的随机策略，定义折扣回报函数$\eta(\pi)$为：$$\eta(\pi) = E_{s_{0},a_{0},...}[\sum_{t=0}^{\infty}{\gamma^{t}r(s_{t})}]$$
其中，$s_{0} \in \rho_{0}(s_{0})$，$a_{t} \in \pi(s_{t}|a_{t})$，$s_{t+1} \in P(s_{t+1} | s_{t}, a_{t})$

### 策略梯度的步长问题
根据策略梯度算法，可知其参数更新方程如下：
$$\theta_{new} = \theta_{old} + a * \nabla_{\theta}J(\theta)$$

策略梯度算法的问题之一就在于步长选取。若步长不合适，更新的参数所对应的策略是一个更不好的策略，当利用这个更不好的策略进行采样学习时，再次更新的参数会更差，因此很容易导致越学越差。


### TRPO的推导
合适的步长对于强化学习非常关键，通过选取一个合适的步长，使得在策略更新后，使得回报函数的值单调增。这是TRPO要解决的问题。对于这个思想，一个自然的想法便是，能否将新的策略对应的回报函数拆分为旧的策略对应的回报函数和其他值的和，若其他值这一项始终大于等于0，便能保证新的策略始终比旧的策略更好。Sham Kakade于2002年提出了以下等式，该等式便是TRPO算法的起点。
$$\eta(\tilde\pi) = \eta(\pi) + E_{s_{0},a_{0},... \in \tilde\pi}{\sum_{t=0}^{\infty}{\gamma^{t}A_{\pi}(s_{t}, a_{t})}}$$
这里用$\tilde\pi$表示新策略，用$\pi$表示旧策略，其中优势函数为：$$A_{\pi}(s,a) = Q_{\pi}(s,a) - V_{\pi}(s) = E_{s^{'} \in P(s^{'} | s, a)}[r(s) + \gamma * V_{\pi}(s^{'}) - V_{\pi}(s)]$$

证明如下：
$$E_{\tau \in \tilde\pi}[{\sum_{t=0}^{\infty}{\gamma^{t}A_{\pi}(s_{t}, a_{t})}}] = J()$$
$$
\begin{align}
 J()=& E_{\tau \in \tilde\pi}[{\sum_{t=0}^{\infty}{\gamma^{t}(r(s) + \gamma * V_{\pi}(s_{t+1}) - V_{\pi}(s_{t}))}}] \\
 = & E_{\tau \in \tilde\pi}[{\sum_{t=0}^{\infty}{\gamma^{t}r(s_{t})}} + {\sum_{t=0}^{\infty}{\gamma^{t}(\gamma * V_{\pi}(s_{t+1}) - V_{\pi}(s_{t}))}}] \\
 = & E_{\tau \in \tilde\pi}[{\sum_{t=0}^{\infty}{\gamma^{t}r(s_{t})}} + {\sum_{t=0}^{\infty}{\gamma^{t+1} * V_{\pi}}(s_{t+1})} - {\sum_{t=0}^{\infty}{\gamma^{t} * V_{\pi}(s_{t})}}] \\
 = & E_{\tau \in \tilde\pi}[{\sum_{t=0}^{\infty}{\gamma^{t}r(s_{t})}} + {\sum_{t=1}^{\infty}{\gamma^{t} * V_{\pi}}(s_{t})} - {\sum_{t=0}^{\infty}{\gamma^{t} * V_{\pi}(s_{t})}}] \\
 = & E_{\tau \in \tilde\pi}[{\sum_{t=0}^{\infty}{\gamma^{t}r(s_{t})}}] + E_{s_{0}}[-V_{\pi}(s_{0})]  \\
 = & E_{\tau \in \tilde\pi}[{\sum_{t=0}^{\infty}{\gamma^{t}r(s_{t})}}] + E_{s_{0}}[-V_{\pi}(s_{0})] \\
 = & \eta(\tilde\pi) - \eta(\pi) \\
\end{align}
$$

我们对新旧策略回报差进行转化。优势函数的期望可以写成如下式：
$$\eta(\tilde\pi) = \eta(\pi) + \sum_{t=0}^{\infty}{\sum_{s}P(s_{t}=s|\tilde\pi){\sum_{a}\tilde\pi(a|s)}\gamma^{t}A_{\pi}(s_{t}, a_{t})}$$

其中，$P(s_{t}=s|\tilde\pi)$是第$t$步出现$s$的概率，${\sum_{a}\tilde\pi(a|s)}$是状态$s$时，对动作进行求和。我们定义$\rho_{\pi}(s)$如下：
$$\rho_{\pi}(s) = P(s_{0}=s) + \gamma * P(s_{1}=s) + \gamma^{2} * P(s_{2}=s) + ...$$
则：
$$\eta(\tilde\pi) = \eta(\pi) + {\sum_{s}\rho_{\tilde\pi}(s){\sum_{a}\tilde\pi(a|s)}A_{\pi}(s_{t}, a_{t})}$$
此时状态$s$的分布由新的策略产生，对新的策略严重依赖。

------
#### 技巧1
TRPO的第一个技巧对状态分布进行处理。我们忽略状态分布的变化，依然采用旧的策略所对应的状态分布。这个技巧是对原代价函数的第一次近似。其实，当新旧参数很接近时，我们将用旧的状态分布代替新的状态分布也是合理的。
原来的代价函数如下：
$$L_{\pi}(\tilde\pi) = \eta(\pi) + {\sum_{s}\rho_{\pi}(s){\sum_{a}\tilde\pi(a|s)}A_{\pi}(s_{t}, a_{t})}$$
这时的动作a是由新的策略$\tilde\pi$产生，新的策略是带$\theta$是参数的，这个参数是未知的，因此无法用来产生动作。

#### 重要性采样

替代回报函如下：
$$L_{\pi}(\tilde\pi) = \eta(\pi) + E_{s \in \rho_{\theta_{old}}, a \in \pi_{\theta_{old}}}{[\frac{\tilde\pi(a|s)}{\pi_{\theta_{old}}(a|s)}A_{\pi_{\theta_{old}}}(s,a)]}$$

#### 确定步长
引入第二个重量级的不等式：$$\eta(\tilde\pi) \geqslant L_{\pi}(\tilde\pi) - C * D_{KL}^{max}{(\pi, \tilde\pi)}$$
受惩罚系数$C$的影响，更新步长会很小，导致更新很慢。一个解决方法是将KL散度作为惩罚项的极值问题，转化为KL散度作为约束条件的优化问题，即：
最大化：$$E_{s \in \rho_{\theta_{old}}, a \in \pi_{\theta_{old}}}{[\frac{\tilde\pi(a|s)}{\pi_{\theta_{old}}(a|s)}A_{\pi_{\theta_{old}}}(s,a)]}$$
且受限于：
$$D_{KL}^{max}{(\theta_{old}, \theta)} \leqslant \delta$$
因为有无穷多的状态，因此约束条件$D_{KL}^{max}{(\theta_{old}, \theta)}$有无穷多个，问题不可解。

#### TRPO最终优化
在约束条件中，利用平均KL散度代替最大KL散度，即：
$$D_{KL}^{\rho_{\theta_{old}}}{(\theta_{old}, \theta)} \leqslant \delta$$
同时：
$$s \in \rho_{\theta_{old}} \rightarrow s \in \pi_{old} $$
最终TRPO问题化简为：
$$E_{s \in \pi_{\theta_{old}}, a \in \pi_{\theta_{old}}}{[\frac{\pi_{\theta}(a|s)}{\pi_{\theta_{old}}(a|s)}A_{\pi_{\theta_{old}}}(s,a)]}$$
$$E_{s \in \pi_{\theta_{old}}}[D_{KL}{(\pi_{\theta_{old}}(\cdot|s)||\pi_{\theta}(\cdot|s))}] \leqslant \delta$$


## 参考文档
arxiv: https://arxiv.org/pdf/1502.05477 

https://zhuanlan.zhihu.com/p/605886935

https://hrl.boyuai.com/chapter/2/trpo%E7%AE%97%E6%B3%95