蒙特卡洛、TD、动态规划的关系？
DQN的几个变种以及各自解决了那些问题？
深度强化学习中的DQN和A3C区别与联系？
策略梯度的推导过程？
策略梯度和actor-critic的关系与对比？
A3C和DDPG区别和共同点？
value-based和policy-based关系？
off-policy和on-policy的好与坏？
表格式到函数近似的理解？
Actor-Critic的优点？
Actor和Critic两者的区别？
advantage(优势函数)推导过程，如何计算？
DPG、DDPG、D3PG、D4PG之间的区别？
强化学习是什么？和有监督学习的异同？SL靠的是样本标签训练模型，RL依靠的是什么？
强化学习用来解决什么问题？
强化学习的损失函数是什么？
为什么最优值函数就等同最优策略
强化学习和动态规划的关系；
简述TD算法
蒙特卡洛和时间差分的对比：MC和TD分别是无偏估计吗，为什么？MC、TD谁的方差大，为什么？
简述Q-Learning，写出其Q(s,a)更新公式
简述值函数逼近的想法？
RL的马尔科夫性质? [imath]t+1[/imath]时的状态仅与[imath]t[/imath]时的状态有关，而与更早之前的历史状态无关。
RL与监督学习和无监督学习的区别
RL不同于其它学习算法的原因?
Model-based和model-free的区别？
确定性策略和 随机性策略的区别与联系？
on-policy 和off-policy的区别与联系？
重要性采样的推导过程、作用？
Q-learning是off-policy的方法，为什么不使用重要性采样？
有哪些方法可以使得RL训练稳定?
写出贝尔曼期望方程和贝尔曼最优方程?
贝尔曼期望方程和贝尔曼最优方程什么时候用?
策略梯度算法的目标函数和策略梯度计算?
DQN的原理？
DQN和Sarsa的区别？
为什么使用优势函数？
常见的平衡探索与利用的方法？
TD3如何解决过估计？
TD3和DDPG的区别？
多臂老虎机和强化学习算法的差别？
多臂老虎机算法的分类？
有那几种Bandit算法？
简述UCB算法 （Upper Confidence Bound)？
简述重要性采样，Thompson sampling采样？
什么是强化学习？
强化学习和监督学习、无监督学习的区别是什么？
强化学习适合解决什么样子的问题？
强化学习的损失函数（loss function）是什么？和深度学习的损失函数有何关系？
POMDP是什么？马尔科夫过程是什么？马尔科夫决策过程是什么？里面的“马尔科夫”体现了什么性质？
贝尔曼方程的具体数学表达式是什么？
最优值函数和最优策略为什么等价？
值迭代和策略迭代的区别？
如果不满足马尔科夫性怎么办？当前时刻的状态和它之前很多很多个状态都有关之间关系？
求解马尔科夫决策过程都有哪些方法？有模型用什么方法？动态规划是怎么回事？
简述动态规划(DP)算法？
简述蒙特卡罗估计值函数(MC)算法。
简述时间差分(TD)算法。
简述动态规划、蒙特卡洛和时间差分的对比（共同点和不同点）
MC和TD分别是无偏估计吗？
MC、TD谁的方差大，为什么？
简述on-policy和off-policy的区别
简述Q-Learning，写出其Q(s,a)更新公式。它是on-policy还是off-policy，为什么？
写出用第n步的值函数更新当前值函数的公式（1-step，2-step，n-step的意思）。当n的取值变大时，期望和方差分别变大、变小？
TD（λ）方法：当λ=0时实际上与哪种方法等价，λ=1呢？
写出蒙特卡洛、TD和TD（λ）这三种方法更新值函数的公式？
value-based和policy-based的区别是什么？
DQN的两个关键trick分别是什么？
阐述目标网络和experience replay的作用？
手工推导策略梯度过程？
描述随机策略和确定性策略的特点？
不打破数据相关性，神经网络的训练效果为什么就不好？
画出DQN玩Flappy Bird的流程图。在这个游戏中，状态是什么，状态是怎么转移的？奖赏函数如何设计，有没有奖赏延迟问题？
DQN都有哪些变种？引入状态奖励的是哪种？
简述double DQN原理？
策略梯度方法中基线baseline如何确定？
什么是DDPG，并画出DDPG框架结构图？
Actor-Critic两者的区别是什么？
actor-critic框架中的critic起了什么作用？
DDPG是on-policy还是off-policy，为什么？
是否了解过D4PG算法？简述其过程
简述A3C算法？A3C是on-policy还是off-policy，为什么？
A3C算法是如何异步更新的？是否能够阐述GA3C和A3C的区别？
简述A3C的优势函数？
什么是重要性采样？
为什么TRPO能保证新策略的回报函数单调不减？
TRPO是如何通过优化方法使每个局部点找到让损失函数非增的最优步长来解决学习率的问题；
如何理解利用平均KL散度代替最大KL散度？
简述PPO算法？与TRPO算法有何关
简述DPPO和PPO的关系？
强化学习如何用在推荐系统中？
推荐场景中奖赏函数如何设计？
场景中状态是什么，当前状态怎么转移到下一状态？
自动驾驶和机器人的场景如何建模成强化学习问题？MDP各元素对应真实场景中的哪些变量？
强化学习需要大量数据，如何生成或采集到这些数据？
是否用某种DRL算法玩过Torcs游戏？具体怎么解决？
是否了解过奖励函数的设置(reward shaping)？
强化学习中如何处理归一化？
强化学习如何观察收敛曲线？
强化学习如何如何确定收敛？
影响强化学习算法收敛的因素有哪些，如何调优？
强化学习的损失函数（loss function）是什么？和深度学习的损失函数有何关系？
多智能体强化学习算法有哪些？
简述Model Based Learning？有什么新的进展？比如World Model？Dream？MuZero?
简述Meta Reinforcement Learning?
为什么Reptile应用的效果并不好？
Meta RL不好应用的原因有哪些？
简述Meta Gradient Reinforcement Learning？
简述Imitation Learning？GAIL? Deepminic?
简述DRL的一些最新改进？R2D3？LASER？
简述Multi-Agent Reinforcement Learning？ 比如MADDPG比较早的，思想是什么？和一般的DRL有什么区别？
简述seed rl? 对于大规模分布式强化学习，还有更好的提高throughput的方法吗？
简述AI-GAs? 你对这个理论有什么看法？
简述Out-of-Distributon Generalization? Modularity?
DRL要实现足够的泛化Generalization有哪些做法？Randomization？
简述Neural-Symbolic Learning的方法？怎么看待？
简述unsupervised reinforcement learning？Diversity is all you need？
简述offline reinforcement learning？
简述Multi-Task Reinforcement Learning？ Policy Distillation？
简述sim2real? 有哪些方法？
对于drl在机器人上的应用怎么看？
简述go-explore?
对于hard exploration的问题，要怎么处理？
简述Transformer？能否具体介绍一下实现方法？
简述Pointer Network？和一般的Attention有什么不同？
什么是Importance Sampling? 为什么PPO和IMPALA要使用？两者在使用方式上有何不同？能否结合？
PPO在实现上是怎么采样的？
为什么使用Gumbel-max? 能否解释一下Gumbel-max 及Gumbel Softmax？
是否了解SAC？ SAC的Policy是什么形式？
SAC的Policy能实现Multi-Modal吗？
是否了解IMPALA？能否解释一下V-Trace？rho和c的作用是什么？
PPO里使用的GAE是怎么实现的？能否写出计算过程？
是否理解Entropy，KL divergence和Mutual Information的含义？
AlphaStar的scatter connection？怎么实现的？
对于多个entity的observation，你会怎么预处理？神经网络要怎么构建？
AlphaStar的League，能否解释一下？如何让agent足够diverse？
Inverse RL 能否解决奖励问题，如何解决的？
分层强化学习的原理是什么 ？
简述分层强化学习中基于目标的(goal-reach)和基于目标的(goal-reach）的区别与联系？
请简述IQL（independent Q-learning算法过程？
是否了解[imath]\alpha-Rank[/imath]算法？
请简述QMIX算法？
简述模仿学习与强化学习的区别、联系？
简述MADDPG算法的过程和伪代码？
多智能体之间如何通信、如何竞争？
你熟悉的多智能体环境有哪些？
你做过的强化学习项目有哪些，遇到的难点有哪些？
请简述造成强化学习inefficient的原因？
sarsa的公式以及和Q-leaning的区别？
是否了解RLlib?Coach？
Ray怎么做梯度并行运算的?
A3C中多线程如何更新梯度？
GA3C算法的queue如何实现？请简述
强化学习的动作、状态以及奖励如何定义的，指标有哪些，包括状态和动作的维度是多少，那些算法效果比较好？、
DQN的trick有哪些？
PPO算法中的clip如何实现的？
简述一些GAE过程？
MADDPG如何解决离散action的？
强化学习在机器人的局限性有哪些 ？
强化学习中如何解决高纬度输入输出问题？
是否了解过奖励函数的设置(reward shaping)？
基于值函数方法的算法有哪些？其损失函数是什么？（MSE）
写出用第n步的值函数更新当前值函数的公式（1-step，2-step，n-step的意思）。当n的取值变大时，期望和方差分别变大、变小？
TD(λ)方法：当λ=0时实际上与哪种方法等价，λ=1呢？
为什么Policy中输出的动作需要sample，而不是直接使用呢？
是否用某种DRL算法玩过Torcs游戏？具体怎么解决？
为什么连续动作环境下使用DDPG的表现还没有直接动作离散化后Q-learning表现好？
PPO算法中的损失函由那些组成？
深度强化学习中奖励函数如何设置？如何Reward Shapping？
你在强化学习模型调试中，有哪些调优技巧？
简述PPO、DPPO算法？
简述PER算法、HER算法？
离散action和连续action在处理上有什么相似和不同的地方？
Baseline为什么可以直接减去一个值而对策略迭代没什么影响？
TRPO的优化目标是什么？
TRPO求逆矩阵的方法是什么？
PPO相比于TRPO的改进是什么？
PPO处理连续动作和离散动作的区别？
PPO的actor损失函数怎么算？
Advantage大于0或者小于0时clip的范围？
有没有用过分布式ppo？一般怎么做底层通信？
Vtrace算法了解吗？IMPALA相比于A3C的优势？
GAE了解吗？两个参数哪个控制偏差哪个控制方差？
详细介绍下GAE怎么计算的。
常用的探索方法有哪些？
知道softQ吗？
强化学习做过图像输入的吗？
自博弈算法完全随机开始和有预训练模型的区别？
介绍纳什均衡
介绍蒙特卡洛搜索树