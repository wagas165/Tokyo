#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import csv
import json
import time
import datetime
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # 非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import multiprocessing
from tqdm import tqdm  # 导入进度条库

# Gym & Stable-Baselines3
import gym
from gym import spaces
from gym.utils import seeding
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

# --------------------- 全局配置 ----------------------
file_path = "data/end_copy.json"  # 场馆和比赛 JSON
baseline_csv_path = "data/daily_overall_EI_delta.csv"  # baseline CSV
NUM = 50  # 节点数
MAX_POP = 1e12

# SEIJRD 参数
paras = {
    "beta": 0.155,  # S->E  传播率(由 E 造成)
    "sigma_i": 0.0299,  # E->I
    "sigma_j": 0.0156,  # E->J
    "gamma_i": 0.079,  # I->R
    "gamma_j": 0.031,  # J->R
    "mu_i": 1.95e-5,  # I->D
    "mu_j": 0.00025,  # J->D
    "rho": 1e-4  # 场馆内接触传染概率
}


# =============== 1. 数据处理 / 比赛 / 场馆 ===============
def process_competitions(data_file):
    slot_mapping = {
        0: (8, 11),
        1: (13, 17),
        2: (19, 22)
    }
    year = 2021
    earliest_date = None
    for venue in data_file:
        for t in venue["time"]:
            event_date = datetime.date(year, t["month"], t["day"])
            if earliest_date is None or event_date < earliest_date:
                earliest_date = event_date

    comps = []
    for venue in data_file:
        vid = venue["venue_id"]
        capacity = int(venue["capacity"].replace(',', ''))
        for t in venue["time"]:
            event_date = datetime.date(year, t["month"], t["day"])
            days_diff = (event_date - earliest_date).days
            slot = t["slot"]  # 0/1/2
            start_hour, end_hour = slot_mapping[slot]
            start_tick = days_diff * 24 + start_hour
            end_tick = days_diff * 24 + end_hour
            comps.append((vid, start_tick, end_tick, capacity, slot))
    return comps


def process_places(data_file):
    places = {}
    for d in data_file:
        vid = d["venue_id"]
        place_info = {
            "venue_id": vid,
            "venue_name": d["venue_name"],
            "capacity": int(d["capacity"].replace(',', ''))
        }
        places[vid] = Place(place_info)
    return places


# ================= 2. Place, SEIJRD, Env =================
class Place:
    """
    audience_from => shape=(node_num, 6)，表示各节点的 [S,E,I,J,R,D]
    """

    def __init__(self, data: Dict):
        self.id = data["venue_id"]
        self.name = data["venue_name"]
        self.capacity = data["capacity"]
        self.agenda = []
        self.endtime = -1
        self.audience_from = []

    def infect(self):
        """
        在场馆内做一次基于 E 的额外传播:
          prob = 1 - (1 - rho)^E_tot
          new_inf = S_tot * prob
        """
        if len(self.audience_from) == 0:
            return
        seijrd_sum = np.sum(self.audience_from, axis=0)  # [S,E,I,J,R,D]
        E_tot = seijrd_sum[1]
        S_tot = seijrd_sum[0]
        if S_tot <= 0:
            return
        try:
            prob = 1 - (1 - paras["rho"]) ** E_tot
        except OverflowError:
            prob = 1.0
        prob = np.clip(prob, 0, 1)
        new_inf = S_tot * prob
        frac_sus = self.audience_from[:, 0] / (S_tot + 1e-12)
        new_inf_each = frac_sus * new_inf

        self.audience_from[:, 0] -= new_inf_each
        self.audience_from[:, 1] += new_inf_each


class SEIJRD_Node:
    """
    单节点, 6 状态: [S, E, I, J, R, D]
    """

    def __init__(self, state, total_nodes):
        self.state = state.astype(np.float64)
        self.from_node = np.zeros((total_nodes, 6))
        self.total_nodes = total_nodes

    def update_seijrd(self, dt=0.01, times=100):
        # 把流入人口加在一起
        from_sum = np.sum(self.from_node, axis=0)
        seijrd_tot = self.state + from_sum
        # N = S+E+I+J+R (不含D)
        N = np.sum(seijrd_tot[:5])
        if N < 1e-8:
            self.from_node[:] = 0
            return

        S, E, I, J, R, D_ = seijrd_tot
        beta = paras["beta"]
        sigma_i = paras["sigma_i"]
        sigma_j = paras["sigma_j"]
        gamma_i = paras["gamma_i"]
        gamma_j = paras["gamma_j"]
        mu_i = paras["mu_i"]
        mu_j = paras["mu_j"]

        dS = - beta * S * E / N
        dE = beta * S * E / N - sigma_i * E - sigma_j * E
        dI = sigma_i * E - gamma_i * I - mu_i * I
        dJ = sigma_j * E - gamma_j * J - mu_j * J
        dR = gamma_i * I + gamma_j * J
        dD = mu_i * I + mu_j * J

        # 按本节点自有 vs. 流入者的占比来分配导数
        frac_self = np.sum(self.state[:5]) / N
        self.state += np.array([dS, dE, dI, dJ, dR, dD]) * frac_self * dt * times

        from_pop_each = np.sum(self.from_node[:, :5], axis=1)
        for idx in range(self.total_nodes):
            if from_pop_each[idx] < 1e-12:
                continue
            ratio_ij = from_pop_each[idx] / N
            self.from_node[idx] += np.array([dS, dE, dI, dJ, dR, dD]) * ratio_ij * dt * times

        self.state = np.clip(self.state, 0, MAX_POP)
        self.from_node = np.clip(self.from_node, 0, MAX_POP)


class SEIJRD_Network:
    """
    包含 node_num 个 SEIJRD_Node
    """

    def __init__(self, num_nodes, states):
        self.node_num = num_nodes
        self.nodes = {}
        for i in range(num_nodes):
            self.nodes[i] = SEIJRD_Node(states[i], num_nodes)

        self.A = np.random.random((num_nodes, num_nodes))
        self.A /= np.sum(self.A, axis=1, keepdims=True)
        self.delta_time = 1 / 2400

    def morning_commuting(self):
        for i in range(self.node_num):
            arr = self.nodes[i].state.copy()
            for j in range(self.node_num):
                if i == j:
                    continue
                a_ij = self.A[i][j]
                move_pop = np.minimum(arr * a_ij, self.nodes[i].state)
                self.nodes[j].from_node[i] = move_pop
                self.nodes[i].state -= move_pop
            self.nodes[i].state = np.clip(self.nodes[i].state, 0, MAX_POP)

    def evening_commuting(self):
        for i in range(self.node_num):
            for j in range(self.node_num):
                if i == j:
                    continue
                self.nodes[j].state += self.nodes[i].from_node[j]
                self.nodes[j].state = np.clip(self.nodes[j].state, 0, MAX_POP)
            self.nodes[i].from_node = np.zeros((self.node_num, 6))

    def update_network(self):
        for i in range(self.node_num):
            self.nodes[i].update_seijrd(self.delta_time, 100)


class Env:
    """
    整体环境：持有 SEIJRD_Network, 场馆, 比赛日程
    capacity_strategy => (n_venues*3,) 的动作
    """

    def __init__(self):
        self.places = {}
        self.competitions = []
        self.network = None
        self.capacity_strategy = None
        self.current_tick = 0

    def init_env(self):
        data_json = json.load(open(file_path, 'r', encoding='utf-8'))
        self.competitions = process_competitions(data_json)
        self.places = process_places(data_json)

        for comp in self.competitions:
            vid = comp[0]
            self.places[vid].agenda.append(comp)
        for p in self.places.values():
            p.agenda.sort(key=lambda x: x[1])

        # 初始状态: [S,E,I,J,R,D]
        init_states = []
        for i in range(NUM):
            # 演示: S=92200, E=800, 其余=0
            init_states.append(np.array([92200, 800, 0, 0, 0, 0]))
        init_states = np.array(init_states)
        self.network = SEIJRD_Network(NUM, init_states)
        self.current_tick = 0

    def check_competition_start(self):
        for pid, place in self.places.items():
            if not place.agenda:
                continue
            comp = place.agenda[0]
            if comp[1] == self.current_tick:
                place.audience_from = np.zeros((self.network.node_num, 6))
                place.agenda.pop(0)
                place.endtime = comp[2]

                capacity = comp[3]
                slot_id = comp[4]

                ratio = 1.0
                if self.capacity_strategy is not None:
                    n_v = len(self.capacity_strategy) // 3
                    a_2d = self.capacity_strategy.reshape((n_v, 3))
                    ratio = a_2d[pid - 1, slot_id]
                actual_cap = capacity * ratio

                # 抽调 [S,E,I,J,R] => D 不移动
                N_list = [np.sum(self.network.nodes[i].state[:5]) for i in range(self.network.node_num)]
                sumN = np.sum(N_list) if np.sum(N_list) > 0 else 1.0
                for i in range(self.network.node_num):
                    frac = (self.network.nodes[i].state[:5] / sumN) * actual_cap
                    frac = np.minimum(frac, self.network.nodes[i].state[:5])
                    moved_6 = np.zeros(6)
                    moved_6[:5] = frac
                    self.network.nodes[i].state[:5] -= frac
                    place.audience_from[i] = moved_6

                place.infect()

    def check_competition_end(self):
        for pl in self.places.values():
            if pl.endtime == self.current_tick:
                for i in range(self.network.node_num):
                    # 把 [S,E,I,J,R] 加回去
                    self.network.nodes[i].state[:5] += pl.audience_from[i][:5]
                pl.audience_from = []
                pl.endtime = -1


# ============== 3. 自定义Gym 环境 =============

class MultiVenueSEIREnv(gym.Env):
    """
    每个step = 1天(24小时)，跑指定天数 (默认20天，对应7/21~8/9).
    每天结束后计算 newE/newI, 并与 baseline 比较得到 reward.
    """

    def __init__(self, days=20, n_venues=41,
                 alpha=3e-8, beta=8.5e-7, gamma=1e-9):
        super(MultiVenueSEIREnv, self).__init__()
        self.days = days
        self.n_venues = n_venues
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.seir_env = Env()
        self.seir_env.init_env()

        # 动作空间 => (n_venues*3,)
        self.action_space = spaces.Box(low=0.0, high=1.0,
                                       shape=(n_venues * 3,),
                                       dtype=np.float32)
        # 观测空间 => 6*NUM
        self.observation_space = spaces.Box(low=0.0, high=MAX_POP,
                                            shape=(6 * NUM,),
                                            dtype=np.float32)

        # baseline CSV => day -> (DeltaI, DeltaE)
        self.baseline_deltaI = {}
        self.baseline_deltaE = {}
        self._load_baseline()

        self.current_day = 0
        self.prev_E, self.prev_I = self._get_total_EI()
        self.state = self._get_observation()

        self.np_random = None
        self.seed_value = None

    def _load_baseline(self):
        with open(baseline_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                d = int(row["Day"])
                dE = float(row["DeltaE"])
                dI = float(row["DeltaI"])
                self.baseline_deltaE[d] = dE
                self.baseline_deltaI[d] = dI

    def _get_total_EI(self):
        E, I = 0.0, 0.0
        for i in range(NUM):
            st = self.seir_env.network.nodes[i].state
            E += st[1]  # E=索引1
            I += st[2]  # I=索引2
        return E, I

    def _get_observation(self):
        obs_list = []
        for i in range(NUM):
            st = self.seir_env.network.nodes[i].state
            obs_list.extend(np.clip(st, 0, MAX_POP))
        return np.array(obs_list, dtype=np.float32)

    def seed(self, seed=None):
        self.np_random, seed_ = seeding.np_random(seed)
        self.seed_value = seed_
        return [seed_]

    def reset(self):
        self.seir_env = Env()
        self.seir_env.init_env()
        self.current_day = 0
        self.prev_E, self.prev_I = self._get_total_EI()
        self.state = self._get_observation()
        return self.state

    def step(self, action):
        self.seir_env.capacity_strategy = action
        start_tick = self.current_day * 24
        end_tick = (self.current_day + 1) * 24

        for hour in range(start_tick, end_tick):
            self.seir_env.current_tick = hour
            self.seir_env.check_competition_start()
            if hour % 24 == 8:
                self.seir_env.network.morning_commuting()
            self.seir_env.network.update_network()
            if hour % 24 == 18:
                self.seir_env.network.evening_commuting()
            self.seir_env.check_competition_end()

        cur_E, cur_I = self._get_total_EI()
        daily_newE = cur_E - self.prev_E
        daily_newI = cur_I - self.prev_I
        self.prev_E, self.prev_I = cur_E, cur_I

        # 收入 => sum(ratio)* capacity * alpha
        a_2d = action.reshape((self.n_venues, 3))
        total_ratio_capacity = 0.0
        for pid, place in self.seir_env.places.items():
            sum_slot_ratio = np.sum(a_2d[pid - 1])
            total_ratio_capacity += sum_slot_ratio * place.capacity
        daily_revenue = total_ratio_capacity * self.alpha

        # 与 baseline 差值
        day_idx = self.current_day
        baseI = self.baseline_deltaI.get(day_idx, 0.0)
        baseE = self.baseline_deltaE.get(day_idx, 0.0)
        exceedI = max(0.0, daily_newI - baseI)
        exceedE = max(0.0, daily_newE - baseE)

        # 以 daily_revenue - (beta*exceedI + gamma*exceedE) 作为 reward
        reward = daily_revenue - (self.beta * exceedI + self.gamma * exceedE)

        self.current_day += 1
        done = (self.current_day >= self.days)
        self.state = self._get_observation()

        info = {
            "newI": daily_newI,
            "newE": daily_newE,
            "day": self.current_day
        }
        return self.state, reward, done, info


# ============== 4. 继续训练已有模型的函数 ==============

class ContinueTrainingCallback(BaseCallback):
    def __init__(self, exp_id=0, version=1, total_timesteps=100000, log_interval=500, verbose=1):
        super(ContinueTrainingCallback, self).__init__(verbose)
        self.exp_id = exp_id
        self.version = version
        self.total_timesteps = total_timesteps
        self.log_interval = log_interval
        self.start_time = None
        self.pbar = None

    def _on_training_start(self):
        self.start_time = time.time()
        # 创建进度条
        self.pbar = tqdm(total=self.total_timesteps, desc=f"[Exp {self.exp_id} v{self.version}]")
        print(f"[Exp {self.exp_id} v{self.version}] 继续训练开始...")

    def _on_step(self):
        steps = self.model.num_timesteps
        # 更新进度条
        if self.pbar is not None:
            self.pbar.n = steps
            self.pbar.refresh()

        if steps % self.log_interval == 0:
            elapsed = time.time() - self.start_time
            fraction = steps / self.total_timesteps
            eta = elapsed / fraction - elapsed if fraction > 0 else 0
            policy_loss = self.model.logger.name_to_value.get("train/policy_loss", None)
            value_loss = self.model.logger.name_to_value.get("train/value_loss", None)
            pl_str = f"{policy_loss:.2e}" if policy_loss else "N/A"
            vl_str = f"{value_loss:.2e}" if value_loss else "N/A"
            # 更新进度条描述
            if self.pbar is not None:
                self.pbar.set_postfix({
                    "loss_p": pl_str,
                    "loss_v": vl_str,
                    "eta": f"{eta / 60:.1f}m"
                })
        return True

    def _on_training_end(self):
        # 关闭进度条
        if self.pbar is not None:
            self.pbar.close()


def make_env_for_continuing(env_id, days, n_venues, alpha, beta, gamma):
    def _init():
        e = MultiVenueSEIREnv(days=days, n_venues=n_venues,
                              alpha=alpha, beta=beta, gamma=gamma)
        e.seed(env_id + int(time.time() % 10000))  # 添加时间戳以确保不同进程有不同随机种子
        return e

    return _init


def continue_training_all_versions(exp_id, alpha, beta, gamma, envs_per_exp=17, days=20, n_venues=41):
    """
    为一个实验训练多个版本 - 避免嵌套进程

    参数:
    - exp_id: 实验ID
    - alpha, beta, gamma: 模型参数
    - envs_per_exp: 每个实验分配的环境数
    """
    print(f"实验 {exp_id} 开始训练多个版本 (α={alpha}, β={beta}, γ={gamma})...")

    # 版本配置
    versions = [1, 2, 3]
    learning_rates = [5e-5, 1e-5, 5e-6]
    total_timesteps_list = [300000, 600000, 800000]

    # 原始模型路径
    original_model_path = f"./results/exp_{exp_id}_A2C_a{alpha}_b{beta}_g{gamma}.zip"

    if not os.path.exists(original_model_path):
        print(f"错误: 找不到原始模型 {original_model_path}")
        return None

    # 为每个版本依次训练
    model_paths = []
    for v_idx, version in enumerate(versions):
        print(f"\n----- 实验 {exp_id} 版本 {version} 开始训练 -----")

        # 创建环境
        env_fns = [make_env_for_continuing(i, days, n_venues, alpha, beta, gamma)
                   for i in range(envs_per_exp)]

        # 使用 SubprocVecEnv 进行并行训练
        vec_env = SubprocVecEnv(env_fns)

        # 创建日志目录
        logdir = f"./logs/exp_{exp_id}_continued_v{version}_A2C_a{alpha}_b{beta}_g{gamma}"
        os.makedirs(logdir, exist_ok=True)

        # 加载模型 - 第一个版本从原始模型加载，后续版本从前一个版本加载
        if v_idx == 0:
            print(f"加载原始模型: {original_model_path}")
            model = A2C.load(original_model_path, env=vec_env, device='cpu')  # 强制使用CPU
        else:
            prev_model_path = model_paths[-1]
            print(f"加载前一版本模型: {prev_model_path}")
            model = A2C.load(prev_model_path, env=vec_env, device='cpu')  # 强制使用CPU

        # 调整学习率
        learning_rate = learning_rates[v_idx]
        model.learning_rate = learning_rate
        print(f"学习率已调整为: {learning_rate}")

        # 训练步数
        total_timesteps = total_timesteps_list[v_idx]

        # 创建回调
        callback = ContinueTrainingCallback(
            exp_id=exp_id,
            version=version,
            total_timesteps=total_timesteps,
            log_interval=500
        )

        # 继续训练
        print(f"开始训练 exp_{exp_id} v{version}，步数={total_timesteps}...")
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            tb_log_name=f"continued_training_v{version}",
            reset_num_timesteps=False,
            progress_bar=True  # SB3内置进度条
        )

        # 保存模型
        new_model_filename = f"exp_{exp_id}_A2C_continued_v{version}_a{alpha}_b{beta}_g{gamma}.zip"
        save_path = os.path.join("./results", new_model_filename)
        model.save(save_path)
        model_paths.append(save_path)

        # 关闭环境
        vec_env.close()

        print(f"✅ 实验 {exp_id} 第 {version} 版训练完成，模型已保存: {save_path}")

        # 立即可视化
        visualize_continued_model(exp_id, version, "A2C", alpha, beta, gamma)

    print(f"\n✅✅ 实验 {exp_id} 所有版本训练完成!")
    return model_paths


def visualize_continued_model(exp_id, version, algo="A2C", alpha=1.0e-8, beta=8.5e-7, gamma=1e-9):
    """
    可视化续训模型的结果
    """
    # 续训模型路径
    model_path = f"./results/exp_{exp_id}_A2C_continued_v{version}_a{alpha}_b{beta}_g{gamma}.zip"

    if not os.path.exists(model_path):
        print(f"错误: 找不到续训模型 {model_path}")
        return

    print(f"开始评估和可视化 exp_{exp_id} v{version}...")

    # 加载模型
    model = A2C.load(model_path, device='cpu')  # 强制使用CPU

    # 创建评估环境
    env = MultiVenueSEIREnv(days=20, n_venues=41, alpha=alpha, beta=beta, gamma=gamma)
    os.makedirs("./visualizations", exist_ok=True)

    # 收集评估数据
    episode_data = {
        "day": [],
        "new_I": [],
        "new_E": [],
        "reward": [],
        "actions": []
    }

    obs = env.reset()
    day = 0
    done = False

    # 使用进度条显示评估进度
    pbar = tqdm(total=20, desc=f"评估 exp_{exp_id} v{version}")

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, rew, done, info = env.step(action)

        episode_data["day"].append(day)
        episode_data["new_I"].append(info["newI"])
        episode_data["new_E"].append(info["newE"])
        episode_data["reward"].append(rew)
        episode_data["actions"].append(action)
        day += 1
        pbar.update(1)

    pbar.close()
    df = pd.DataFrame(episode_data)

    # 1) 绘制 daily newI & newE
    plt.figure(figsize=(10, 6))
    plt.plot(df["day"], df["new_I"], 'r-', label="daily newI")
    plt.plot(df["day"], df["new_E"], 'y-', label="daily newE")
    plt.title(f"Model {exp_id} (续训v{version}) - daily newInfections & newExposures")
    plt.xlabel("Day")
    plt.ylabel("Population")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"./visualizations/exp_{exp_id}_continued_v{version}_newIE.png")

    # 2) Heatmap of actions
    actions_array = np.array(episode_data["actions"])
    plt.figure(figsize=(12, 8))
    sns.heatmap(actions_array.T, cmap="YlOrRd",
                xticklabels=range(df.shape[0]),
                yticklabels=range(actions_array.shape[1]))
    plt.title(f"Model {exp_id} (续训v{version}) - Action Heatmap")
    plt.xlabel("Days")
    plt.ylabel("Action Index")
    plt.savefig(f"./visualizations/exp_{exp_id}_continued_v{version}_policy.png")

    # 3) 保存数据
    df.to_csv(f"./visualizations/exp_{exp_id}_continued_v{version}_data.csv", index=False)

    plt.close('all')
    print(f"✅ 完成续训模型 {exp_id}v{version} 的可视化 => './visualizations/'")


# ================ 5. 主函数 - 单层进程模型 ================

if __name__ == "__main__":
    # 忽略警告
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning)

    # 创建必要的目录
    os.makedirs("./results", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    os.makedirs("./visualizations", exist_ok=True)

    # 第二轮实验配置
    base_experiments = [
        dict(exp_id=2, alpha=1.0e-8, beta=8.241e-07, gamma=1.0e-9),
        dict(exp_id=5, alpha=1.0e-8, beta=8.966e-07, gamma=1.0e-9),
        dict(exp_id=10, alpha=1.0e-8, beta=1.017e-06, gamma=1.0e-9),
        dict(exp_id=11, alpha=1.0e-8, beta=1.041e-06, gamma=1.0e-9),
        dict(exp_id=12, alpha=1.0e-8, beta=1.066e-06, gamma=1.0e-9),
        dict(exp_id=17, alpha=1.0e-8, beta=1.186e-06, gamma=1.0e-9),
        dict(exp_id=22, alpha=1.0e-8, beta=1.307e-06, gamma=1.0e-9),
        dict(exp_id=26, alpha=1.0e-8, beta=1.403e-06, gamma=1.0e-9),
    ]

    # 计算每个实验的并行环境数
    num_experiments = len(base_experiments)
    total_cores = 140
    versions_per_exp = 3  # 每个实验训练3个版本

    # 每个实验分配的并行环境数 - 确保充分利用所有CPU
    envs_per_exp = total_cores // num_experiments
    print(f"将训练 {num_experiments} 个实验，每个实验 {versions_per_exp} 个版本，每个实验使用 {envs_per_exp} 个并行环境")

    # 创建进程池，为每个实验分配一个进程
    all_processes = []
    for exp in base_experiments:
        p = multiprocessing.Process(
            target=continue_training_all_versions,
            args=(
                exp["exp_id"],
                exp["alpha"],
                exp["beta"],
                exp["gamma"],
                envs_per_exp,
                20,
                41
            )
        )
        all_processes.append(p)

    # 使用主进度条显示整体训练进度
    pbar_main = tqdm(total=len(all_processes), desc="实验总进度")

    # 启动所有进程
    for p in all_processes:
        p.start()

    # 等待所有进程完成并更新进度条
    active_processes = len(all_processes)
    while active_processes > 0:
        completed = sum(1 for p in all_processes if not p.is_alive())
        if completed > pbar_main.n:
            pbar_main.n = completed
            pbar_main.refresh()
        active_processes = sum(1 for p in all_processes if p.is_alive())
        time.sleep(5)  # 每5秒检查一次

    # 确保所有进程都已结束
    for p in all_processes:
        p.join()

    pbar_main.n = len(all_processes)
    pbar_main.close()
    print("\n===== 所有实验的所有版本训练完成! =====")

    # 打印结果总结
    print("\n====== 续训实验总结 ======")
    for i, exp in enumerate(base_experiments):
        print(f"{i + 1}. 实验ID: {exp['exp_id']}, α={exp['alpha']}, β={exp['beta']}, γ={exp['gamma']}")
        for v in range(1, 4):
            model_path = f"./results/exp_{exp['exp_id']}_A2C_continued_v{v}_a{exp['alpha']}_b{exp['beta']}_g{exp['gamma']}.zip"
            status = "✅ 已完成" if os.path.exists(model_path) else "❌ 未完成"
            print(f"   - v{v}: {status}")
    print("=========================")