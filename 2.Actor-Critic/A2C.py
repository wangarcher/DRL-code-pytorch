import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter


# Actor 网络(策略网络): 状态 s → 动作概率分布 π(a|s)
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_width):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_width)  # 输入层 → 隐藏层
        self.l2 = nn.Linear(hidden_width, action_dim)  # 隐藏层 → 动作维(输出 logits)

    def forward(self, s):
        s = F.relu(self.l1(s))  # 隐藏层 + ReLU 激活
        # softmax 把 logits 归一化为概率分布: π_i = e^z_i / Σ_j e^z_j, dim=1 沿动作维
        a_prob = F.softmax(self.l2(s), dim=1)
        return a_prob


# Critic 网络(价值网络): 状态 s → V(s), 估计从 s 出发的期望折扣回报
class Critic(nn.Module):
    def __init__(self, state_dim, hidden_width):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_width)  # 输入层 → 隐藏层
        self.l2 = nn.Linear(hidden_width, 1)  # 隐藏层 → 1 维输出 V(s)

    def forward(self, s):
        s = F.relu(self.l1(s))  # 隐藏层 + ReLU 激活
        v_s = self.l2(s)  # 线性输出 V(s), 不加激活(价值可正可负)
        return v_s


class A2C(object):
    """A2C(Advantage Actor-Critic) 智能体
    核心: 策略梯度定理 ∇J = E[γ^t · ∇log π(a|s) · A(s,a)]
    其中优势 A(s,a) 用单步 TD 误差近似: A ≈ δ = r + γV(s') - V(s)
    特点: 每与环境交互一步就立即学习一次(单步更新, 无 buffer, 纯 on-policy)
    """

    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_width = 64  # 隐藏层神经元数量
        self.lr = 5e-4  # 学习率
        self.GAMMA = 0.99  # 折扣因子 γ
        self.I = 1  # 折扣累乘器, 表示策略梯度定理中的 γ^t(每 learn 一次乘一次 γ)

        self.actor = Actor(state_dim, action_dim, self.hidden_width)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)

        self.critic = Critic(state_dim, self.hidden_width)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

    def choose_action(self, s, deterministic):
        """按策略选择动作: 训练时随机采样(探索), 评估时贪心取最大概率动作"""
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)  # 单状态加 batch 维: (state_dim,)→(1, state_dim)
        prob_weights = self.actor(s).detach().numpy().flatten()  # 前向得概率分布, 断梯度转 numpy 压平成一维
        if deterministic:  # 评估时用确定性策略
            a = np.argmax(prob_weights)  # 直接选概率最大的动作
            return a
        else:  # 训练时用随机策略(保留探索)
            a = np.random.choice(range(self.action_dim), p=prob_weights)  # 按概率分布 π(a|s) 随机采样动作
            return a

    def learn(self, s, a, r, s_, dw):
        """单步学习: 用一步 transition (s,a,r,s_,dw) 同时更新 actor 和 critic
        actor: 沿策略梯度 γ^t · log π(a|s) · δ 的方向抬升/压低动作概率
        critic: TD 学习, 让 V(s) 逼近 td_target = r + γ(1-dw)V(s')
        """
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)  # 加 batch 维
        s_ = torch.unsqueeze(torch.tensor(s_, dtype=torch.float), 0)  # 加 batch 维
        v_s = self.critic(s).flatten()  # V(s): critic 对当前状态的价值估计
        v_s_ = self.critic(s_).flatten()  # V(s'): critic 对下一状态的价值估计

        with torch.no_grad():  # td_target 是回归"标签", 不参与梯度回传
            # TD 目标: td_target = r + γ·(1-dw)·V(s')
            # (1-dw) 的作用: dw=True(死亡/获胜,真终止)时砍掉 bootstrap 项 V(s'), 否则死局被"凭空续命";
            # 到达 max_episode_steps 被截断时 dw=False, 环境实际仍在继续, V(s') 保留
            td_target = r + self.GAMMA * (1 - dw) * v_s_

        # 更新 actor
        log_pi = torch.log(self.actor(s).flatten()[a])  # log π(a|s): 采样到的动作 a 的对数概率
        # actor 损失 = -γ^t · δ · log π(a|s)
        # 其中 δ = td_target - V(s) 是 TD 误差, 即优势 A(s,a) 的单步估计:
        #   δ>0(实际比预期好) → 抬高该动作概率; δ<0(比预期差) → 压低该动作概率
        # δ.detach(): 优势只当"权重"用, 梯度只从 log_pi 流向 actor(这步不更新 critic)
        actor_loss = -self.I * ((td_target - v_s).detach()) * log_pi  # 只对 log_pi 求导
        self.actor_optimizer.zero_grad()  # 清空梯度
        actor_loss.backward()  # 反向传播
        self.actor_optimizer.step()  # 更新 actor 参数

        # 更新 critic
        # critic 损失 = δ² = (td_target - V(s))², 标准 TD 回归(此处梯度只从 v_s 流向 critic)
        critic_loss = (td_target - v_s) ** 2  # 只对 v(s) 求导
        self.critic_optimizer.zero_grad()  # 清空梯度
        critic_loss.backward()  # 反向传播
        self.critic_optimizer.step()  # 更新 critic 参数

        self.I *= self.GAMMA  # I ← γ·I, 累乘表示策略梯度定理中的 γ^t(每步折扣一次)


def evaluate_policy(env, agent):
    """评估函数: 用确定性策略跑 times 个回合, 返回平均回合奖励"""
    times = 3  # 评估 3 个回合取平均, 降低单回合随机性
    evaluate_reward = 0
    for _ in range(times):
        s = env.reset()  # 重置评估环境
        done = False
        episode_reward = 0
        while not done:
            a = agent.choose_action(s, deterministic=True)  # 评估用确定性策略(argmax)
            s_, r, done, _ = env.step(a)
            episode_reward += r  # 累计回合奖励
            s = s_
        evaluate_reward += episode_reward

    return int(evaluate_reward / times)  # 返回平均奖励(取整)


if __name__ == '__main__':
    env_name = ['CartPole-v0', 'CartPole-v1']  # 可选环境
    env_index = 0  # 当前选 CartPole-v0(改这里换环境)
    env = gym.make(env_name[env_index])  # 训练环境
    env_evaluate = gym.make(env_name[env_index])  # 评估环境(训练评估分开, 互不干扰)
    number = 9  # 实验编号(区分重复实验)
    # 设置随机种子(环境/动作空间/numpy/torch), 保证实验可复现
    seed = 0
    env.seed(seed)
    env.action_space.seed(seed)
    env_evaluate.seed(seed)
    env_evaluate.action_space.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    state_dim = env.observation_space.shape[0]  # 状态维度
    action_dim = env.action_space.n  # 离散动作数
    max_episode_steps = env._max_episode_steps  # 每回合最大步数(区分"真终止"和"到时截断"用)
    print("state_dim={}".format(state_dim))
    print("action_dim={}".format(action_dim))
    print("max_episode_steps={}".format(max_episode_steps))

    agent = A2C(state_dim, action_dim)
    writer = SummaryWriter(log_dir='runs/A2C/A2C_env_{}_number_{}_seed_{}'.format(env_name[env_index], number, seed))  # 建 TensorBoard 日志

    max_train_steps = 3e5  # 总训练步数
    evaluate_freq = 1e3  # 每 1000 步评估一次
    evaluate_rewards = []  # 记录每次评估的平均奖励
    evaluate_num = 0  # 已评估次数
    total_steps = 0  # 已交互总步数

    while total_steps < max_train_steps:  # 主循环: 直到总步数耗尽
        episode_steps = 0
        s = env.reset()  # 新回合初始状态
        done = False
        agent.I = 1  # 每回合重置折扣累乘器 γ^t = 1(回合内 t 从 0 重新计)
        while not done:  # 回合内循环
            episode_steps += 1
            a = agent.choose_action(s, deterministic=False)  # 训练时随机采样(探索)
            s_, r, done, _ = env.step(a)

            # 区分两类"终止":
            # 死亡/获胜(done 且未到步数上限) → dw=True: 没有下一状态 s'
            # 达到 max_episode_steps → dw=False: 只是仿真被截断, 实际还有 s'
            if done and episode_steps != max_episode_steps:
                dw = True
            else:
                dw = False

            agent.learn(s, a, r, s_, dw)  # A2C 特点: 每步立即学习(无 buffer, 纯 on-policy)
            s = s_  # 状态前移

            # 每 evaluate_freq 步评估一次
            if (total_steps + 1) % evaluate_freq == 0:
                evaluate_num += 1
                evaluate_reward = evaluate_policy(env_evaluate, agent)
                evaluate_rewards.append(evaluate_reward)
                print("evaluate_num:{} \t evaluate_reward:{} \t".format(evaluate_num, evaluate_reward))
                writer.add_scalar('step_rewards_{}'.format(env_name[env_index]), evaluate_reward, global_step=total_steps)  # 写 TensorBoard 曲线
                # 保存奖励数据
                if evaluate_num % 10 == 0:
                    np.save('./data_train/A2C_env_{}_number_{}_seed_{}.npy'.format(env_name[env_index], number, seed), np.array(evaluate_rewards))

            total_steps += 1
