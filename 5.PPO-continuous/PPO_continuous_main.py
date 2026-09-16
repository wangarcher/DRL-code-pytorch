import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import gym
import argparse
from normalization import Normalization, RewardScaling
from replaybuffer import ReplayBuffer
from ppo_continuous import PPO_continuous


def evaluate_policy(args, env, agent, state_norm):
    """评估函数: 用确定性策略(分布均值)在评估环境里跑 times 个回合, 返回平均回合奖励
    评估要点: ① 用均值动作不探索 ② 状态归一化 update=False(冻结统计量, 保持与训练一致的尺度)
    """
    times = 3  # 评估回合数(取平均以降低单回合随机性)
    evaluate_reward = 0
    for _ in range(times):
        s = env.reset()  # 重置评估环境, 回到初始状态
        if args.use_state_norm:
            s = state_norm(s, update=False)  # 评估期间不更新均值/方差, 只做变换 (Trick 2)
        done = False
        episode_reward = 0
        while not done:
            a = agent.evaluate(s)  # 评估时用确定性策略(分布均值)
            if args.policy_dist == "Beta":
                # Beta 分布支撑集是 (0,1), 环境动作范围是 [-max,max], 做线性映射:
                # action = 2·(a-0.5)·max_action, 即 0→-max, 0.5→0, 1→+max
                action = 2 * (a - 0.5) * args.max_action  # [0,1]->[-max,max]
            else:
                action = a  # Gaussian 分支: forward 已用 tanh·max_action 压到 [-max,max], 直接用
            s_, r, done, _ = env.step(action)  # 环境执行动作, 返回下一状态/奖励/终止标志
            if args.use_state_norm:
                s_ = state_norm(s_, update=False)  # 同样只变换、不更新统计量
            episode_reward += r  # 累计本回合奖励
            s = s_  # 状态滚动前移
        evaluate_reward += episode_reward

    return evaluate_reward / times  # 返回 times 个回合的平均奖励


def main(args, env_name, number, seed):
    """主训练流程: 建环境 → 设随机种子 → 采样循环(攒满 batch 就 update) → 周期性评估并记录"""
    env = gym.make(env_name)  # 训练环境
    env_evaluate = gym.make(env_name)  # 评估环境(训练评估分开, 避免互相干扰状态)
    # 设置随机种子(环境/动作空间/numpy/torch), 保证实验可复现
    env.seed(seed)
    env.action_space.seed(seed)
    env_evaluate.seed(seed)
    env_evaluate.action_space.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    args.state_dim = env.observation_space.shape[0]  # 状态维度(如 HalfCheetah 为 17)
    args.action_dim = env.action_space.shape[0]  # 动作维度(连续, 如 6)
    args.max_action = float(env.action_space.high[0])  # 动作上界(Beta 映射 / Gaussian tanh 缩放都用它)
    args.max_episode_steps = env._max_episode_steps  # 每回合最大步数(区分"真终止"和"到时截断"用)
    print("env={}".format(env_name))
    print("state_dim={}".format(args.state_dim))
    print("action_dim={}".format(args.action_dim))
    print("max_action={}".format(args.max_action))
    print("max_episode_steps={}".format(args.max_episode_steps))

    evaluate_num = 0  # 已评估次数
    evaluate_rewards = []  # 每次评估的平均奖励列表
    total_steps = 0  # 已交互的总步数

    replay_buffer = ReplayBuffer(args)  # on-policy 缓冲区(攒满 batch_size 条就更新并清零)
    agent = PPO_continuous(args)  # PPO 智能体(内含 actor + critic + 优化器)

    # 建 TensorBoard 日志, 目录含环境名/策略分布/编号/种子, 便于区分实验
    writer = SummaryWriter(log_dir='runs/PPO_continuous/env_{}_{}_number_{}_seed_{}'.format(env_name, args.policy_dist, number, seed))

    state_norm = Normalization(shape=args.state_dim)  # Trick 2: 状态归一化 (在线统计均值方差)
    if args.use_reward_norm:  # Trick 3: 奖励归一化(减均值除方差, 二选一)
        reward_norm = Normalization(shape=1)
    elif args.use_reward_scaling:  # Trick 4: 奖励缩放(只除以折扣回报的 std, 二选一)
        reward_scaling = RewardScaling(shape=1, gamma=args.gamma)

    while total_steps < args.max_train_steps:  # 主循环: 直到总步数耗尽
        s = env.reset()  # 新回合初始状态
        if args.use_state_norm:
            s = state_norm(s)  # 训练时归一化会同时更新统计量
        if args.use_reward_scaling:
            reward_scaling.reset()  # 每回合重置折扣累计回报 R=0(Trick 4 的回合边界)
        episode_steps = 0
        done = False
        while not done:  # 回合内循环
            episode_steps += 1
            a, a_logprob = agent.choose_action(s)  # 按当前策略分布采样动作及其 log π_old(a|s)
            if args.policy_dist == "Beta":
                action = 2 * (a - 0.5) * args.max_action  # [0,1]->[-max,max] 区间映射
            else:
                action = a
            s_, r, done, _ = env.step(action)  # 与环境交互一步

            if args.use_state_norm:
                s_ = state_norm(s_)  # 下一状态归一化
            if args.use_reward_norm:
                r = reward_norm(r)  # 奖励归一化 (r-μ)/(σ+ε)
            elif args.use_reward_scaling:
                r = reward_scaling(r)  # 奖励缩放 r/(std(R)+ε), R=γR+r 递推

            # 区分两类"终止":
            # 死亡/获胜(done 且未到步数上限) → dw=True: 没有下一个状态 s', bootstrap 必须清零
            # 达到 max_episode_steps → dw=False: 环境其实还在继续, V(s') 应保留(只是仿真被截断)
            if done and episode_steps != args.max_episode_steps:
                dw = True
            else:
                dw = False

            # 注意: 环境执行的是映射后的 'action', 但 buffer 存的是原始 'a'(Beta 时 a∈(0,1)),
            # 因为更新时 log_prob 是按 (0,1) 支撑集的分布算的, 必须与采样时一致
            replay_buffer.store(s, a, a_logprob, r, s_, dw, done)
            s = s_  # 状态前移
            total_steps += 1

            # buffer 攒满 batch_size 条 → 触发一次 PPO 更新, 然后清零重新采样(on-policy)
            if replay_buffer.count == args.batch_size:
                agent.update(replay_buffer, total_steps)
                replay_buffer.count = 0

            # 每 evaluate_freq 步评估一次策略
            if total_steps % args.evaluate_freq == 0:
                evaluate_num += 1
                evaluate_reward = evaluate_policy(args, env_evaluate, agent, state_norm)  # 评估环境跑 3 回合取平均
                evaluate_rewards.append(evaluate_reward)
                print("evaluate_num:{} \t evaluate_reward:{} \t".format(evaluate_num, evaluate_reward))
                writer.add_scalar('step_rewards_{}'.format(env_name), evaluate_rewards[-1], global_step=total_steps)  # 写 TensorBoard 曲线
                # 保存奖励曲线数据
                if evaluate_num % args.save_freq == 0:
                    np.save('./data_train/PPO_continuous_{}_env_{}_number_{}_seed_{}.npy'.format(args.policy_dist, env_name, number, seed), np.array(evaluate_rewards))


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for PPO-continuous")
    parser.add_argument("--max_train_steps", type=int, default=int(3e6), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=5e3, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=20, help="Save frequency")
    parser.add_argument("--policy_dist", type=str, default="Gaussian", help="Beta or Gaussian")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--hidden_width", type=int, default=64, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr_a", type=float, default=3e-4, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=3e-4, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=10, help="PPO parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_state_norm", type=bool, default=True, help="Trick 2:state normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=True, help="Trick 4:reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=float, default=True, help="Trick 10: tanh activation function")

    args = parser.parse_args()

    env_name = ['BipedalWalker-v3', 'HalfCheetah-v2', 'Hopper-v2', 'Walker2d-v2']  # 可选训练环境列表
    env_index = 1  # 当前选 HalfCheetah-v2 (改这里换环境)
    main(args, env_name=env_name[env_index], number=1, seed=10)  # number 是实验编号(区分重复实验), seed 是随机种子
