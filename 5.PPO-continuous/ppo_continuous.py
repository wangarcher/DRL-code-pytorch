import torch
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torch.nn as nn
from torch.distributions import Beta, Normal


# Trick 8: 正交初始化
# 作用: 把层权重 W 初始化为随机正交矩阵 × gain, 即满足 W·W^T = gain²·I
# 性质: 正交矩阵保持向量范数不变, 因此 Var(Wx) = gain²·Var(x)
#       gain=1 时信号方差逐层不变 → 前向激活不爆炸、反向梯度不消失
# 同时把偏置 b 置 0, 避免初始输出带偏移
def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)  # 权重 → 正交矩阵 × gain
    nn.init.constant_(layer.bias, 0)  # 偏置 → 0


class Actor_Beta(nn.Module):
    """Beta 分布策略网络(连续动作, 默认方案)
    用 Beta(alpha, beta) 分布作为策略 π(a|s), 支撑集天然在 (0,1) 内:
    - 不会采出越界动作, 无需 clamp 截断(避免高斯截断导致的概率模型不自洽问题)
    网络结构: state → fc1 → fc2 → 两个输出层(alpha_layer / beta_layer)
    """

    def __init__(self, args):
        super(Actor_Beta, self).__init__()
        self.fc1 = nn.Linear(args.state_dim, args.hidden_width)  # 输入层 → 隐藏层
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)  # 隐藏层 → 隐藏层
        self.alpha_layer = nn.Linear(args.hidden_width, args.action_dim)  # 输出 Beta 分布参数 α, 每个动作维度一个
        self.beta_layer = nn.Linear(args.hidden_width, args.action_dim)  # 输出 Beta 分布参数 β, 每个动作维度一个
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]  # Trick10: 用 True/False 当下标选激活函数 (False→ReLU, True→Tanh)

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)  # gain=1: 方差保持不变
            orthogonal_init(self.fc2)  # gain=1: 方差保持不变
            # 输出层 gain=0.01: 把 α、β 的输出压小, 初始化时 α≈β≈1 → 分布接近均匀分布
            # → 初始策略接近均匀随机, 保持探索, 梯度温和
            orthogonal_init(self.alpha_layer, gain=0.01)
            orthogonal_init(self.beta_layer, gain=0.01)

    def forward(self, s):
        """前向传播: 状态 s → 分布参数 (alpha, beta)
        数据流: s → 激活(fc1) → 激活(fc2) → 两个线性输出层 → softplus 变换
        """
        s = self.activate_func(self.fc1(s))  # 第一隐藏层 + 激活
        s = self.activate_func(self.fc2(s))  # 第二隐藏层 + 激活
        # alpha 和 beta 必须大于 1, 所以先用 softplus 激活再 +1:
        #   softplus(x) = log(1+e^x) > 0  → 保证输出恒正
        #   +1.0 保证 α, β > 1:
        #     α,β>1 时 Beta 分布单峰(钟形), 边界处密度为 0, 训练稳定
        #     若 α<1 或 β<1, 密度在边界 0 或 1 处趋于无穷(U 形), 训练不稳定
        alpha = F.softplus(self.alpha_layer(s)) + 1.0
        beta = F.softplus(self.beta_layer(s)) + 1.0
        return alpha, beta

    def get_dist(self, s):
        """构造 Beta 分布对象, 供采样和训练使用
        action_dim 维动作 = action_dim 个相互独立的一维 Beta 分布
        """
        alpha, beta = self.forward(s)  # 网络前向得到分布参数
        # 组装 Beta 分布: p(a) = a^(α-1)·(1-a)^(β-1) / B(α,β), a∈(0,1)
        # 其中 B(α,β) 是归一化常数(Beta 函数), 保证密度积分为 1
        dist = Beta(alpha, beta)
        return dist

    def mean(self, s):
        """计算 Beta 分布的均值(评估时作为确定性动作)
        均值公式: E[a] = α / (α+β), 天然落在 (0,1) 内
        """
        alpha, beta = self.forward(s)
        mean = alpha / (alpha + beta)  # Beta 分布的均值
        return mean


class Actor_Gaussian(nn.Module):
    """高斯分布策略网络(连续动作, 经典基线)
    用 N(mean, std) 作为策略 π(a|s):
    - mean 由网络输出, tanh 压缩到 [-max_action, max_action]
    - log_std 是可学习参数(与状态无关, 所有状态共享)
    缺点: 高斯无界, 采样后需要 clamp 截断, 边界处概率模型不自洽
    """

    def __init__(self, args):
        super(Actor_Gaussian, self).__init__()
        self.max_action = args.max_action  # 动作边界绝对值
        self.fc1 = nn.Linear(args.state_dim, args.hidden_width)  # 输入层 → 隐藏层
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)  # 隐藏层 → 隐藏层
        self.mean_layer = nn.Linear(args.hidden_width, args.action_dim)  # 输出高斯均值, 每个动作维度一个
        # log_std 定义为可学习参数(nn.Parameter 会随反向传播自动更新)
        # 初始为 0 → 初始 std = e^0 = 1
        self.log_std = nn.Parameter(torch.zeros(1, args.action_dim))  # 用 nn.Parameter 自动训练 log_std
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]  # Trick10: 用 True/False 当下标选激活函数 (False→ReLU, True→Tanh)

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)  # gain=1: 方差保持不变
            orthogonal_init(self.fc2)  # gain=1: 方差保持不变
            # 输出层 gain=0.01: 初始均值接近 0 → 初始策略接近对称随机, 保持探索
            orthogonal_init(self.mean_layer, gain=0.01)

    def forward(self, s):
        """前向传播: 状态 s → 均值 mean
        tanh 把线性输出压到 (-1,1), 再乘 max_action 缩放到 (-max_action, max_action)
        """
        s = self.activate_func(self.fc1(s))  # 第一隐藏层 + 激活
        s = self.activate_func(self.fc2(s))  # 第二隐藏层 + 激活
        mean = self.max_action * torch.tanh(self.mean_layer(s))  # [-1,1]→[-max_action,max_action]
        return mean

    def get_dist(self, s):
        """构造高斯分布对象, 供采样和训练使用"""
        mean = self.forward(s)  # 网络输出均值
        log_std = self.log_std.expand_as(mean)  # 把 log_std 扩展成和 mean 相同的形状 (B, action_dim)
        # 训练 log_std 而不是 std 的原因: std = exp(log_std) 恒大于 0,
        # 保证标准差恒为正, 避免直接训练 std 可能出现负数的非法情况
        std = torch.exp(log_std)  # std = e^(log_std) > 0
        # 高斯分布密度: p(a) = (1/√(2πσ²))·exp(-(a-μ)²/(2σ²)), a∈(-∞,+∞)
        dist = Normal(mean, std)  # 得到高斯分布
        return dist


class Critic(nn.Module):
    """价值网络: 状态 s → V(s), 估计状态价值(从 s 出发的期望折扣回报 E[Σγ^t·r_t])
    输出维度为 1, 是回归任务: 最后一层不加激活、gain 用默认 1.0
    """

    def __init__(self, args):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(args.state_dim, args.hidden_width)  # 输入层 → 隐藏层
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)  # 隐藏层 → 隐藏层
        self.fc3 = nn.Linear(args.hidden_width, 1)  # 隐藏层 → 1 维输出 V(s)
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]  # Trick10: 用 True/False 当下标选激活函数 (False→ReLU, True→Tanh)

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)  # gain=1: 方差保持不变
            orthogonal_init(self.fc2)  # gain=1: 方差保持不变
            orthogonal_init(self.fc3)  # 价值回归输出层, gain=1(无需像 actor 输出层那样压小)

    def forward(self, s):
        """前向传播: s → V(s), 三层 MLP, 输出对状态价值的估计"""
        s = self.activate_func(self.fc1(s))  # 第一隐藏层 + 激活
        s = self.activate_func(self.fc2(s))  # 第二隐藏层 + 激活
        v_s = self.fc3(s)  # 线性输出 V(s), 不加激活(价值可正可负, 量级不受限)
        return v_s


class PPO_continuous():
    """PPO 连续动作版智能体: 组合 actor + critic, 实现采样(choose_action)/评估(evaluate)/更新(update)/学习率衰减(lr_decay)"""

    def __init__(self, args):
        self.policy_dist = args.policy_dist  # 策略分布类型: "Beta" 或 "Gaussian"
        self.max_action = args.max_action  # 动作边界绝对值
        self.batch_size = args.batch_size  # 一次更新使用的经验总量(on-policy rollout 大小)
        self.mini_batch_size = args.mini_batch_size  # 每次梯度计算使用的小批样本数
        self.max_train_steps = args.max_train_steps  # 总训练步数(学习率衰减用)
        self.lr_a = args.lr_a  # actor 学习率
        self.lr_c = args.lr_c  # critic 学习率
        self.gamma = args.gamma  # 折扣因子 γ: 未来奖励的折扣权重
        self.lamda = args.lamda  # GAE 参数 λ: 多步 TD 误差的加权衰减系数
        self.epsilon = args.epsilon  # PPO clip 参数 ε: 新旧策略比值允许的偏离范围
        self.K_epochs = args.K_epochs  # 一批数据重复训练的轮数
        self.entropy_coef = args.entropy_coef  # 熵正则系数: 鼓励探索, 防止策略过早收敛
        self.set_adam_eps = args.set_adam_eps
        self.use_grad_clip = args.use_grad_clip
        self.use_lr_decay = args.use_lr_decay
        self.use_adv_norm = args.use_adv_norm

        # 按超参选择策略分布: Beta(默认, 天然有界)或 Gaussian(经典基线), 二者接口一致可一键切换
        if self.policy_dist == "Beta":
            self.actor = Actor_Beta(args)
        else:
            self.actor = Actor_Gaussian(args)
        self.critic = Critic(args)

        # actor / critic 各自独立的 Adam 优化器(学习率可分开调)
        if self.set_adam_eps:  # Trick 9: 设置 Adam epsilon=1e-5
            # eps 是 Adam 更新式 θ ← θ - η·m̂/(√v̂+ε) 分母中的防除零常数
            # RL 中梯度常接近 0, 默认 1e-8 太小会把噪声也放大成标准步长;
            # 调大到 1e-5 后, 梯度小时步长 ∝ 梯度大小, 自动抑制末期噪声
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a, eps=1e-5)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c, eps=1e-5)
        else:
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)

    def evaluate(self, s):  # 评估策略时, 直接使用分布的均值(确定性动作, 不探索)
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)  # 单个状态加 batch 维: (state_dim,) → (1, state_dim)
        if self.policy_dist == "Beta":
            # Beta 分布: 动作 = 均值 α/(α+β) ∈ (0,1)
            # 注意: main 中还要做区间映射 action = 2·(a-0.5)·max_action, 把 (0,1) 映射到 (-max,max)
            a = self.actor.mean(s).detach().numpy().flatten()  # detach 切断梯度 → 转 numpy → 压平成 (action_dim,)
        else:
            # 高斯分布: forward 已经输出 tanh 压缩后的均值, 直接用
            a = self.actor(s).detach().numpy().flatten()
        return a

    def choose_action(self, s):
        """训练时按策略分布随机采样动作(保留探索), 并记录采样时刻的 log π_old(a|s) 供更新时计算比率"""
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)  # 单个状态加 batch 维: (state_dim,) → (1, state_dim)
        if self.policy_dist == "Beta":
            with torch.no_grad():  # 采样阶段只"用"网络不训练, 关闭梯度记录省内存
                dist = self.actor.get_dist(s)  # 由网络输出构造 Beta 分布
                a = dist.sample()  # 按概率分布随机采样动作, 天然落在 (0,1) 内, 无越界问题
                a_logprob = dist.log_prob(a)  # 记录对数概率密度 log π_old(a|s), 存入 buffer 后冻结, 更新时用于计算 ratio
        else:
            with torch.no_grad():  # 采样阶段只"用"网络不训练, 关闭梯度记录省内存
                dist = self.actor.get_dist(s)  # 由网络输出构造高斯分布
                a = dist.sample()  # 按概率分布随机采样动作(高斯无界)
                a = torch.clamp(a, -self.max_action, self.max_action)  # 截断到 [-max,max](Beta 分支不需要这步, 这正是 Beta 存在的意义)
                a_logprob = dist.log_prob(a)  # 记录对数概率密度 log π_old(a|s)(注意: 用的是截断前分布的密度, 这是高斯的固有缺陷)
        return a.numpy().flatten(), a_logprob.numpy().flatten()  # 转成一维 numpy 数组返回 (动作, 对数概率)

    def update(self, replay_buffer, total_steps):
        """PPO 核心更新: ① 用 GAE 计算优势 → ② K 轮小批量更新 actor/critic → ③(外部)学习率衰减"""
        s, a, a_logprob, r, s_, dw, done = replay_buffer.numpy_to_tensor()  # 从 buffer 取出整批训练数据; a_logprob 是采样时刻冻结的 log π_old
        """
            使用 GAE(广义优势估计)计算优势函数
            'dw=True' 表示死亡或获胜: 没有下一个状态 s', bootstrap 项必须清零
            'done=True' 表示回合终止(死亡/获胜/达到最大步数): 计算 adv 时, 若 done=True 则 gae=0(切断回合间优势传递)
        """
        adv = []
        gae = 0
        with torch.no_grad():  # adv 和 v_target 只是"标签", 不参与梯度回传
            vs = self.critic(s)  # V(s_t): critic 对当前状态的价值估计
            vs_ = self.critic(s_)  # V(s_{t+1}): critic 对下一状态的价值估计
            # TD 误差: δ_t = r_t + γ·(1-dw)·V(s_{t+1}) - V(s_t)
            # (1-dw) 的作用: dw=True(真终止)时砍掉 bootstrap 项 V(s'), 否则死局会被"凭空续命", 价值严重高估;
            # 若只是达到 max_episode_steps 被截断(dw=False), 环境实际仍在继续, V(s') 应保留
            deltas = r + self.gamma * (1.0 - dw) * vs_ - vs
            # 从后往前递推 GAE: A_t = δ_t + γλ·(1-d_t)·A_{t+1}
            # 展开即多步 TD 的指数加权混合: A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...
            # (1-d) 的作用: 遇到回合边界就把来自下一回合的 A_{t+1} 清零, 避免两个回合的优势互相污染
            for delta, d in zip(reversed(deltas.flatten().numpy()), reversed(done.flatten().numpy())):
                gae = delta + self.gamma * self.lamda * gae * (1.0 - d)
                adv.insert(0, gae)  # 每次插到队头, 把顺序翻转回正常时间序
            adv = torch.tensor(adv, dtype=torch.float).view(-1, 1)  # 转回 tensor, 形状 (batch_size, 1)
            v_target = adv + vs  # 价值回归目标 = A_t + V(s_t), 即 TD(λ) 回归目标(比蒙特卡洛回报方差更低)
            if self.use_adv_norm:  # Trick 1: 优势归一化
                adv = ((adv - adv.mean()) / (adv.std() + 1e-5))  # 标准化为零均值单位方差, 统一"得分"尺度, 稳定更新幅度
                # 注意: v_target 用的是归一化之前的 adv, 二者不能错位

        # 优化策略 K 轮(epoch):
        for _ in range(self.K_epochs):
            # 随机采样且不重复: 把 batch_size 条数据随机切成若干个大小为 mini_batch_size 的小批
            # 'False' 表示最后不足 mini_batch_size 的一组也继续训练
            for index in BatchSampler(SubsetRandomSampler(range(self.batch_size)), self.mini_batch_size, False):
                dist_now = self.actor.get_dist(s[index])  # 用当前(新)策略重新构造分布, 梯度从这里流向 actor
                dist_entropy = dist_now.entropy().sum(1, keepdim=True)  # 策略熵 H = -Σ_a π log π, 多维动作各维独立、逐维求和, 形状(mini_batch_size X 1)
                a_logprob_now = dist_now.log_prob(a[index])  # 新策略下旧动作的对数概率密度 log π_new(a|s), 形状(mini_batch_size X action_dim)
                # 重要性采样比率: r_t = π_new(a|s)/π_old(a|s) = exp(log π_new - log π_old)
                # 多维连续动作空间下, 各维独立分布的联合概率 = 各维概率之积,
                # log 域里乘积变求和, 所以要对 action_dim 维求和, 形状(mini_batch_size X 1)
                ratios = torch.exp(a_logprob_now.sum(1, keepdim=True) - a_logprob[index].sum(1, keepdim=True))  # shape(mini_batch_size X 1)

                surr1 = ratios * adv[index]  # 未截断的代理目标 r_t·A_t(只对 a_logprob_now 计算梯度)
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv[index]  # 把比值截断到 [1-ε, 1+ε] 再乘优势
                # PPO-Clip 目标(取负号变最小化), 外加熵正则项:
                # actor_loss = -E[ min(r_t·A_t, clip(r_t, 1±ε)·A_t) ] - c·H(π)
                # clip 的直觉: A>0(好动作)时, r_t 超过 1+ε 后不再奖励"过度强化";
                #             A<0(差动作)时, r_t 低于 1-ε 后不再鼓励"过度贬低", 保护旧策略防止塌缩
                actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy  # Trick 5: 策略熵(保持探索, 防止过早收敛)
                # 更新 actor
                self.optimizer_actor.zero_grad()  # 清空历史梯度
                actor_loss.mean().backward()  # 对小批取均值后反向传播
                if self.use_grad_clip:  # Trick 7: 梯度裁剪
                    # 把全部参数梯度的整体 L2 范数缩放到 ≤0.5, 防止离群大梯度把策略带崩
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.optimizer_actor.step()  # 优化器更新 actor 参数

                v_s = self.critic(s[index])  # critic 前向: 当前对 V(s) 的预测
                # critic 损失 = MSE: 让 V(s) 逼近 TD(λ) 回归目标 v_target, 即 L = E[(v_target - V(s))²]
                critic_loss = F.mse_loss(v_target[index], v_s)
                # 更新 critic
                self.optimizer_critic.zero_grad()  # 清空历史梯度
                critic_loss.backward()  # 反向传播
                if self.use_grad_clip:  # Trick 7: 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)  # 梯度范数缩放到 ≤0.5
                self.optimizer_critic.step()  # 优化器更新 critic 参数

        if self.use_lr_decay:  # Trick 6: 学习率衰减
            self.lr_decay(total_steps)  # 一批数据更新完毕后, 按当前总步数衰减学习率

    def lr_decay(self, total_steps):
        """线性学习率衰减: lr_now = lr_0 · (1 - t/T)
        训练后期步长自动变小, 避免大步长破坏已接近收敛的策略
        """
        lr_a_now = self.lr_a * (1 - total_steps / self.max_train_steps)  # actor 当前学习率(随训练进度线性降到 0)
        lr_c_now = self.lr_c * (1 - total_steps / self.max_train_steps)  # critic 当前学习率(随训练进度线性降到 0)
        for p in self.optimizer_actor.param_groups:  # 直接改写优化器参数组里的学习率
            p['lr'] = lr_a_now
        for p in self.optimizer_critic.param_groups:
            p['lr'] = lr_c_now
