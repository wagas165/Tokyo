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
matplotlib.use("Agg")  # 非交互式后端，适用于服务器环境
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import multiprocessing

# Gym & Stable-Baselines3
import gym
from gym import spaces
from gym.utils import seeding
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback


# --------------------- 全局配置 ----------------------
file_path = "data/end_copy.json"           # 场馆和比赛 JSON
baseline_csv_path = "data/daily_overall_EI_delta.csv"  # baseline CSV
NUM = 50         # 节点数
MAX_POP = 1e12

# SEIJRD 参数, 将旧的 (beta,gamma,sigma,mu) 拆分成 6 状态模型
# 参考新版 main.py, 并保留 "E 为传染源" 的动力学
paras = {
    "beta": 0.155,        # S->E  传播率(由 E 造成)
    "sigma_i": 0.0299,    # E->I
    "sigma_j": 0.0156,    # E->J
    "gamma_i": 0.079,     # I->R
    "gamma_j": 0.031,     # J->R
    "mu_i": 1.95e-5,      # I->D
    "mu_j": 0.00025,      # J->D
    "rho": 1e-4           # 场馆内接触传染概率
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
            start_tick = days_diff*24 + start_hour
            end_tick   = days_diff*24 + end_hour
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
            prob = 1 - (1 - paras["rho"])**E_tot
        except OverflowError:
            prob = 1.0
        prob = np.clip(prob, 0, 1)
        new_inf = S_tot * prob
        frac_sus = self.audience_from[:,0] / (S_tot+1e-12)
        new_inf_each = frac_sus * new_inf

        self.audience_from[:,0] -= new_inf_each
        self.audience_from[:,1] += new_inf_each

class SEIJRD_Node:
    """
    单节点, 6 状态: [S, E, I, J, R, D]
    """
    def __init__(self, state, total_nodes):
        self.state = state.astype(np.float64)
        self.from_node = np.zeros((total_nodes,6))
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
        beta     = paras["beta"]
        sigma_i  = paras["sigma_i"]
        sigma_j  = paras["sigma_j"]
        gamma_i  = paras["gamma_i"]
        gamma_j  = paras["gamma_j"]
        mu_i     = paras["mu_i"]
        mu_j     = paras["mu_j"]

        dS = - beta * S * E / N
        dE = beta * S * E / N - sigma_i*E - sigma_j*E
        dI = sigma_i*E - gamma_i*I - mu_i*I
        dJ = sigma_j*E - gamma_j*J - mu_j*J
        dR = gamma_i*I + gamma_j*J
        dD = mu_i*I + mu_j*J

        # 按本节点自有 vs. 流入者的占比来分配导数
        frac_self = np.sum(self.state[:5]) / N
        self.state += np.array([dS,dE,dI,dJ,dR,dD]) * frac_self * dt * times

        from_pop_each = np.sum(self.from_node[:,:5], axis=1)
        for idx in range(self.total_nodes):
            if from_pop_each[idx] < 1e-12:
                continue
            ratio_ij = from_pop_each[idx]/N
            self.from_node[idx] += np.array([dS,dE,dI,dJ,dR,dD]) * ratio_ij * dt * times

        self.state     = np.clip(self.state,     0, MAX_POP)
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

        self.A = np.random.random((num_nodes,num_nodes))
        self.A /= np.sum(self.A, axis=1, keepdims=True)
        self.delta_time = 1/2400

    def morning_commuting(self):
        for i in range(self.node_num):
            arr = self.nodes[i].state.copy()
            for j in range(self.node_num):
                if i==j:
                    continue
                a_ij = self.A[i][j]
                move_pop = np.minimum(arr*a_ij, self.nodes[i].state)
                self.nodes[j].from_node[i] = move_pop
                self.nodes[i].state -= move_pop
            self.nodes[i].state = np.clip(self.nodes[i].state, 0, MAX_POP)

    def evening_commuting(self):
        for i in range(self.node_num):
            for j in range(self.node_num):
                if i==j:
                    continue
                self.nodes[j].state += self.nodes[i].from_node[j]
                self.nodes[j].state = np.clip(self.nodes[j].state, 0, MAX_POP)
            self.nodes[i].from_node = np.zeros((self.node_num,6))

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
            init_states.append(np.array([92200,800,0,0,0,0]))
        init_states = np.array(init_states)
        self.network = SEIJRD_Network(NUM, init_states)
        self.current_tick = 0

    def check_competition_start(self):
        for pid, place in self.places.items():
            if not place.agenda:
                continue
            comp = place.agenda[0]
            if comp[1] == self.current_tick:
                place.audience_from = np.zeros((self.network.node_num,6))
                place.agenda.pop(0)
                place.endtime = comp[2]

                capacity = comp[3]
                slot_id  = comp[4]

                ratio = 1.0
                if self.capacity_strategy is not None:
                    n_v = len(self.capacity_strategy)//3
                    a_2d = self.capacity_strategy.reshape((n_v,3))
                    ratio = a_2d[pid-1, slot_id]
                actual_cap = capacity*ratio

                # 抽调 [S,E,I,J,R] => D 不移动
                N_list = [np.sum(self.network.nodes[i].state[:5]) for i in range(self.network.node_num)]
                sumN = np.sum(N_list) if np.sum(N_list)>0 else 1.0
                for i in range(self.network.node_num):
                    frac = (self.network.nodes[i].state[:5]/sumN)*actual_cap
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
        self.beta  = beta
        self.gamma = gamma

        self.seir_env = Env()
        self.seir_env.init_env()

        # 动作空间 => (n_venues*3,)
        self.action_space = spaces.Box(low=0.0, high=1.0,
                                       shape=(n_venues*3,),
                                       dtype=np.float32)
        # 观测空间 => 6*NUM
        self.observation_space = spaces.Box(low=0.0, high=MAX_POP,
                                            shape=(6*NUM,),
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
            E += st[1]   # E=索引1
            I += st[2]   # I=索引2
        return E,I

    def _get_observation(self):
        obs_list = []
        for i in range(NUM):
            st = self.seir_env.network.nodes[i].state
            obs_list.extend(np.clip(st,0,MAX_POP))
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
        end_tick   = (self.current_day+1) * 24

        for hour in range(start_tick, end_tick):
            self.seir_env.current_tick = hour
            self.seir_env.check_competition_start()
            if hour%24 == 8:
                self.seir_env.network.morning_commuting()
            self.seir_env.network.update_network()
            if hour%24 == 18:
                self.seir_env.network.evening_commuting()
            self.seir_env.check_competition_end()

        cur_E, cur_I = self._get_total_EI()
        daily_newE = cur_E - self.prev_E
        daily_newI = cur_I - self.prev_I
        self.prev_E, self.prev_I = cur_E, cur_I

        # 收入 => sum(ratio)* capacity * alpha
        a_2d = action.reshape((self.n_venues,3))
        total_ratio_capacity = 0.0
        for pid, place in self.seir_env.places.items():
            sum_slot_ratio = np.sum(a_2d[pid-1])
            total_ratio_capacity += sum_slot_ratio * place.capacity
        daily_revenue = total_ratio_capacity * self.alpha

        # 与 baseline 差值
        day_idx = self.current_day
        baseI = self.baseline_deltaI.get(day_idx, 0.0)
        baseE = self.baseline_deltaE.get(day_idx, 0.0)
        exceedI = max(0.0, daily_newI - baseI)
        exceedE = max(0.0, daily_newE - baseE)

        # 以 daily_revenue - (beta*exceedI + gamma*exceedE) 作为 reward
        reward = daily_revenue - (self.beta*exceedI + self.gamma*exceedE)

        self.current_day += 1
        done = (self.current_day >= self.days)
        self.state = self._get_observation()

        info = {
            "newI": daily_newI,
            "newE": daily_newE,
            "day": self.current_day
        }
        return self.state, reward, done, info


# ========== 基线和定常策略的跑法（给 compare_capacity_strategies 用） ==========

def run_baseline(env):
    """
    在给定的 env 上执行 baseline 策略:
      - venue_id in [3,5,24,37,40] => 0.5, 否则 => 0
    直到 done
    返回 df: [day, new_I, reward]
    （此处 reward 暂未返回, 也可根据需求补充, 示例中只主要对 newI 做对比）
    """
    # 为确保从头模拟，这里重新 reset env
    env.reset()

    baseline_action_2d = np.zeros((env.n_venues,3), dtype=np.float32)
    special_ids = [3,5,24,37,40]
    for vid in special_ids:
        baseline_action_2d[vid-1] = 0.5

    records = {
        "day": [],
        "new_I": [],
        "reward": []
    }
    done = False
    day = 0
    obs = env.state  # 或 env.reset() 后的state
    while not done:
        action = baseline_action_2d.reshape(-1)
        obs, rew, done, info = env.step(action)
        records["day"].append(day)
        records["new_I"].append(info["newI"])
        records["reward"].append(rew)
        day += 1
    df = pd.DataFrame(records)
    return df

def run_constant_ratio(env, ratio):
    """
    在给定 env 上执行固定策略(全场馆所有时段都用相同 ratio).
    直到 done.
    返回 [day, new_I, reward].
    """
    env.reset()
    const_action_2d = np.ones((env.n_venues,3), dtype=np.float32)*ratio

    records = {
        "day": [],
        "new_I": [],
        "reward": []
    }
    done = False
    day = 0
    obs = env.state
    while not done:
        action = const_action_2d.reshape(-1)
        obs, rew, done, info = env.step(action)
        records["day"].append(day)
        records["new_I"].append(info["newI"])
        records["reward"].append(rew)
        day += 1
    df = pd.DataFrame(records)
    return df


# ========== 主要可视化函数（替换旧的） ==========

def compare_capacity_strategies(env, model, exp_id, algo, alpha, beta, gamma):
    """
    画一张2x2图:
      (左上) daily newI
      (右上) cumulative newI
      (左下) daily reward
      (右下) cumulative reward
    """
    # baseline
    baseline_df = run_baseline(env)
    baseline_df["cum_new_I"]  = baseline_df["new_I"].cumsum()
    baseline_df["cum_reward"] = baseline_df["reward"].cumsum()

    # 定常策略
    ratio_list = [0.0, 0.3, 0.5, 1.0]
    ratio_data = {}
    for r in ratio_list:
        cdf = run_constant_ratio(env, r)
        cdf["cum_new_I"]  = cdf["new_I"].cumsum()
        cdf["cum_reward"] = cdf["reward"].cumsum()
        ratio_data[r] = cdf

    # RL 策略
    env.reset()
    rl_dict = {"day":[],"new_I":[],"reward":[]}
    obs= env.state
    done=False
    day=0
    while not done:
        action, _= model.predict(obs, deterministic=True)
        obs, rew, done, info= env.step(action)
        rl_dict["day"].append(day)
        rl_dict["new_I"].append(info["newI"])
        rl_dict["reward"].append(rew)
        day+=1
    rl_df= pd.DataFrame(rl_dict)
    rl_df["cum_new_I"]  = rl_df["new_I"].cumsum()
    rl_df["cum_reward"] = rl_df["reward"].cumsum()

    # 绘图
    fig, axes = plt.subplots(2,2, figsize=(12,10))
    ax1, ax2, ax3, ax4 = axes.flatten()

    # 左上: daily newI
    ax1.plot(baseline_df["day"], baseline_df["new_I"], label="Baseline", lw=2)
    for r,dfc in ratio_data.items():
        ax1.plot(dfc["day"], dfc["new_I"], label=f"ratio {r}", alpha=0.8)
    ax1.plot(rl_df["day"], rl_df["new_I"], label="RL", lw=2)
    ax1.set_title("Daily New Infections")
    ax1.set_xlabel("Day")
    ax1.set_ylabel("newI")
    ax1.legend()
    ax1.grid(True)

    # 右上: cumulative newI
    ax2.plot(baseline_df["day"], baseline_df["cum_new_I"], label="Baseline", lw=2)
    for r,dfc in ratio_data.items():
        ax2.plot(dfc["day"], dfc["cum_new_I"], label=f"ratio {r}", alpha=0.8)
    ax2.plot(rl_df["day"], rl_df["cum_new_I"], label="RL", lw=2)
    ax2.set_title("Cumulative New Infections")
    ax2.set_xlabel("Day")
    ax2.set_ylabel("cum newI")
    ax2.legend()
    ax2.grid(True)

    # 左下: daily reward
    ax3.plot(baseline_df["day"], baseline_df["reward"], label="Baseline", lw=2)
    for r,dfc in ratio_data.items():
        ax3.plot(dfc["day"], dfc["reward"], label=f"ratio {r}", alpha=0.8)
    ax3.plot(rl_df["day"], rl_df["reward"], label="RL", lw=2)
    ax3.set_title("Daily Reward")
    ax3.set_xlabel("Day")
    ax3.set_ylabel("Reward")
    ax3.legend()
    ax3.grid(True)

    # 右下: cumulative reward
    ax4.plot(baseline_df["day"], baseline_df["cum_reward"], label="Baseline", lw=2)
    for r,dfc in ratio_data.items():
        ax4.plot(dfc["day"], dfc["cum_reward"], label=f"ratio {r}", alpha=0.8)
    ax4.plot(rl_df["day"], rl_df["cum_reward"], label="RL", lw=2)
    ax4.set_title("Cumulative Reward")
    ax4.set_xlabel("Day")
    ax4.set_ylabel("Reward")
    ax4.legend()
    ax4.grid(True)

    fig.suptitle(f"Strategy Comparison (Exp {exp_id}, {algo}, α={alpha}, β={beta}, γ={gamma})", fontsize=14)
    plt.tight_layout(rect=[0,0.03,1,0.97])

    os.makedirs("./visualizations", exist_ok=True)
    out_png = f"./visualizations/exp_{exp_id}_{algo}_strategy_comparison.png"
    plt.savefig(out_png)
    plt.close(fig)

    print(f"✅ Created multi-metric strategy comparison figure for Exp {exp_id}, saved at: {out_png}")


def visualize_model_results(exp_id, algo, alpha, beta, gamma):
    """
    Evaluate model and generate visualization results:
      - plot daily newI vs day,
      - heatmap for actions,
      - final compare_capacity_strategies => 2x2 figure
    """
    # 训练时保存模型时的命名
    model_path = f"./results/exp_{exp_id}_{algo}_a{alpha}_b{beta}_g{gamma}.zip"

    # 加载对应算法的模型
    if algo == "A2C":
        from stable_baselines3 import A2C
        model = A2C.load(model_path)
    else:
        print("Only A2C covered here, you can expand to PPO/TD3/SAC if needed.")
        return

    # 生成一个环境，用来收集 step-wise 数据
    env = MultiVenueSEIREnv(days=20, n_venues=41, alpha=alpha, beta=beta, gamma=gamma)
    os.makedirs("./visualizations", exist_ok=True)

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
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, rew, done, info = env.step(action)

        episode_data["day"].append(day)
        episode_data["new_I"].append(info["newI"])
        episode_data["new_E"].append(info["newE"])
        episode_data["reward"].append(rew)
        episode_data["actions"].append(action)
        day += 1

    df = pd.DataFrame(episode_data)

    # 1) daily newI & newE
    plt.figure(figsize=(10,6))
    plt.plot(df["day"], df["new_I"], 'r-', label="daily newI")
    plt.plot(df["day"], df["new_E"], 'y-', label="daily newE")
    plt.title(f"Model {exp_id} ({algo}) - daily newInfections & newExposures")
    plt.xlabel("Day")
    plt.ylabel("Population")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"./visualizations/exp_{exp_id}_newIE.png")

    # 2) Heatmap of actions
    actions_array = np.array(episode_data["actions"])
    plt.figure(figsize=(12,8))
    sns.heatmap(actions_array.T, cmap="YlOrRd",
                xticklabels=range(df.shape[0]),
                yticklabels=range(actions_array.shape[1]))
    plt.title(f"Model {exp_id} {algo} - Action Heatmap")
    plt.xlabel("Days")
    plt.ylabel("Action Index")
    plt.savefig(f"./visualizations/exp_{exp_id}_policy.png")

    # 3) Save data
    df.to_csv(f"./visualizations/exp_{exp_id}_data.csv", index=False)

    # 4) Compare different capacity strategies => 2x2 figure
    compare_capacity_strategies(env, model, exp_id, algo, alpha, beta, gamma)

    plt.close('all')
    print(f"✅ Completed model {exp_id} visualization => './visualizations/'")


# ============== 4. 并行 A2C 训练(示例) ==============
class MyProgressCallback(BaseCallback):
    def __init__(self, exp_id=0, total_timesteps=3000, log_interval=500, verbose=1):
        super(MyProgressCallback, self).__init__(verbose)
        self.exp_id = exp_id
        self.total_timesteps = total_timesteps
        self.log_interval = log_interval
        self.start_time = None

    def _on_training_start(self):
        self.start_time = time.time()

    def _on_step(self):
        steps = self.model.num_timesteps
        if steps%self.log_interval ==0:
            elapsed = time.time()-self.start_time
            fraction = steps/self.total_timesteps
            eta = elapsed/fraction - elapsed if fraction>0 else 0
            policy_loss = self.model.logger.name_to_value.get("train/policy_loss", None)
            value_loss  = self.model.logger.name_to_value.get("train/value_loss", None)
            pl_str = f"{policy_loss:.2e}" if policy_loss else "N/A"
            vl_str = f"{value_loss:.2e}" if value_loss else "N/A"
            print(f"[Exp {self.exp_id}] steps={steps}/{self.total_timesteps} ({fraction*100:.1f}%), "
                  f"policy_loss={pl_str}, value_loss={vl_str}, ETA={eta/60:.1f}m")
        return True

def make_env(env_id, days, n_venues, alpha, beta, gamma):
    def _init():
        e = MultiVenueSEIREnv(days=days, n_venues=n_venues,
                              alpha=alpha, beta=beta, gamma=gamma)
        e.seed(env_id)
        return e
    return _init

def train_experiment(exp_id, gpu_id, alpha, beta, gamma,
                     total_timesteps=3000, days=20, n_venues=41):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    num_envs = 4
    env_fns = [make_env(i, days, n_venues, alpha, beta, gamma) for i in range(num_envs)]
    vec_env = SubprocVecEnv(env_fns)

    logdir = f"./logs/exp_{exp_id}_A2C_a{alpha}_b{beta}_g{gamma}"
    callback = MyProgressCallback(exp_id=exp_id, total_timesteps=total_timesteps, log_interval=500)

    model = A2C("MlpPolicy", vec_env, verbose=1, tensorboard_log=logdir)
    model.learn(total_timesteps=total_timesteps,
                callback=callback,
                progress_bar=True)

    # 注意命名 => 与 visualize_model_results() 中保持一致
    model_filename = f"exp_{exp_id}_A2C_a{alpha}_b{beta}_g{gamma}.zip"
    model.save(os.path.join("./results", model_filename))
    vec_env.close()
    print(f"[Exp {exp_id}] done. alpha={alpha}, beta={beta}, gamma={gamma}, GPU={gpu_id}.")

def run_experiments():
    """
    可以自行增减配置, 这里示例几组
    """
    # experiments = [
    #     dict(exp_id=1, gpu_id=0, alpha=1e-8, beta=1e-6, gamma=1e-9),
    #     dict(exp_id=2, gpu_id=1, alpha=1e-8, beta=8.5e-7, gamma=1e-9),
    #     dict(exp_id=3, gpu_id=2, alpha=1e-8, beta=1.1e-6, gamma=1e-9),
    #     dict(exp_id=4, gpu_id=3, alpha=1e-8, beta=1.1e-6, gamma=1e-9),
    #     dict(exp_id=5, gpu_id=4, alpha=1e-8, beta=9.2e-7, gamma=1e-9),
    #     dict(exp_id=6, gpu_id=5, alpha=1e-8, beta=1.0e-6, gamma=1e-9),
    #     dict(exp_id=7, gpu_id=6, alpha=1e-8, beta=8.7e-7, gamma=1e-9),
    #     dict(exp_id=8, gpu_id=7, alpha=1e-8, beta=1.05e-6, gamma=1e-9),
    # ]
    experiments = [
        dict(exp_id=1, gpu_id=0, alpha=1.0e-8, beta=8.000e-07, gamma=1.0e-9),  # ratio=80.0
        dict(exp_id=2, gpu_id=1, alpha=1.0e-8, beta=8.241e-07, gamma=1.0e-9),  # ratio=82.4
        dict(exp_id=3, gpu_id=2, alpha=1.0e-8, beta=8.483e-07, gamma=1.0e-9),  # ratio=84.8
        dict(exp_id=4, gpu_id=3, alpha=1.0e-8, beta=8.724e-07, gamma=1.0e-9),  # ratio=87.2
        dict(exp_id=5, gpu_id=4, alpha=1.0e-8, beta=8.966e-07, gamma=1.0e-9),  # ratio=89.7
        dict(exp_id=6, gpu_id=5, alpha=1.0e-8, beta=9.207e-07, gamma=1.0e-9),  # ratio=92.1
        dict(exp_id=7, gpu_id=6, alpha=1.0e-8, beta=9.448e-07, gamma=1.0e-9),  # ratio=94.5
        dict(exp_id=8, gpu_id=7, alpha=1.0e-8, beta=9.690e-07, gamma=1.0e-9),  # ratio=96.9
        dict(exp_id=9, gpu_id=0, alpha=1.0e-8, beta=9.931e-07, gamma=1.0e-9),  # ratio=99.3
        dict(exp_id=10, gpu_id=1, alpha=1.0e-8, beta=1.017e-06, gamma=1.0e-9),  # ratio=101.7
        dict(exp_id=11, gpu_id=2, alpha=1.0e-8, beta=1.041e-06, gamma=1.0e-9),  # ratio=104.1
        dict(exp_id=12, gpu_id=3, alpha=1.0e-8, beta=1.066e-06, gamma=1.0e-9),  # ratio=106.6
        dict(exp_id=13, gpu_id=4, alpha=1.0e-8, beta=1.090e-06, gamma=1.0e-9),  # ratio=109.0
        dict(exp_id=14, gpu_id=5, alpha=1.0e-8, beta=1.114e-06, gamma=1.0e-9),  # ratio=111.4
        dict(exp_id=15, gpu_id=6, alpha=1.0e-8, beta=1.138e-06, gamma=1.0e-9),  # ratio=113.8
        dict(exp_id=16, gpu_id=7, alpha=1.0e-8, beta=1.162e-06, gamma=1.0e-9),  # ratio=116.2
        dict(exp_id=17, gpu_id=0, alpha=1.0e-8, beta=1.186e-06, gamma=1.0e-9),  # ratio=118.6
        dict(exp_id=18, gpu_id=1, alpha=1.0e-8, beta=1.210e-06, gamma=1.0e-9),  # ratio=121.0
        dict(exp_id=19, gpu_id=2, alpha=1.0e-8, beta=1.234e-06, gamma=1.0e-9),  # ratio=123.4
        dict(exp_id=20, gpu_id=3, alpha=1.0e-8, beta=1.259e-06, gamma=1.0e-9),  # ratio=125.9
        dict(exp_id=21, gpu_id=4, alpha=1.0e-8, beta=1.283e-06, gamma=1.0e-9),  # ratio=128.3
        dict(exp_id=22, gpu_id=5, alpha=1.0e-8, beta=1.307e-06, gamma=1.0e-9),  # ratio=130.7
        dict(exp_id=23, gpu_id=6, alpha=1.0e-8, beta=1.331e-06, gamma=1.0e-9),  # ratio=133.1
        dict(exp_id=24, gpu_id=7, alpha=1.0e-8, beta=1.355e-06, gamma=1.0e-9),  # ratio=135.5
        dict(exp_id=25, gpu_id=0, alpha=1.0e-8, beta=1.379e-06, gamma=1.0e-9),  # ratio=137.9
        dict(exp_id=26, gpu_id=1, alpha=1.0e-8, beta=1.403e-06, gamma=1.0e-9),  # ratio=140.3
        dict(exp_id=27, gpu_id=2, alpha=1.0e-8, beta=1.428e-06, gamma=1.0e-9),  # ratio=142.8
        dict(exp_id=28, gpu_id=3, alpha=1.0e-8, beta=1.452e-06, gamma=1.0e-9),  # ratio=145.2
        dict(exp_id=29, gpu_id=4, alpha=1.0e-8, beta=1.476e-06, gamma=1.0e-9),  # ratio=147.6
        dict(exp_id=30, gpu_id=5, alpha=1.0e-8, beta=1.500e-06, gamma=1.0e-9),  # ratio=150.0
    ]

    experiments = [
        dict(exp_id=2, gpu_id=1, alpha=1.0e-8, beta=8.241e-07, gamma=1.0e-9),
        dict(exp_id=5, gpu_id=4, alpha=1.0e-8, beta=8.966e-07, gamma=1.0e-9),
        dict(exp_id=10, gpu_id=1, alpha=1.0e-8, beta=1.017e-06, gamma=1.0e-9),  # ratio=101.7
        dict(exp_id=11, gpu_id=2, alpha=1.0e-8, beta=1.041e-06, gamma=1.0e-9),  # ratio=104.1
        dict(exp_id=12, gpu_id=3, alpha=1.0e-8, beta=1.066e-06, gamma=1.0e-9),  # ratio=106.6
        dict(exp_id=17, gpu_id=0, alpha=1.0e-8, beta=1.186e-06, gamma=1.0e-9),  # ratio=118.6
        dict(exp_id=22, gpu_id=5, alpha=1.0e-8, beta=1.307e-06, gamma=1.0e-9),
        dict(exp_id=26, gpu_id=1, alpha=1.0e-8, beta=1.403e-06, gamma=1.0e-9),
    ]

    procs=[]
    for cfg in experiments:
        p = multiprocessing.Process(
            target=train_experiment,
            args=(cfg["exp_id"], cfg["gpu_id"],
                  cfg["alpha"], cfg["beta"], cfg["gamma"],
                  1000000, 20, 41)  # 20 天
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()
    print("All experiments finished.")

    # =========== 训练完毕后, 测试 & 可视化 =============
    for cfg in experiments:
        exp_id= cfg["exp_id"]
        alpha= cfg["alpha"]
        beta = cfg["beta"]
        gamma= cfg["gamma"]
        visualize_model_results(exp_id, "A2C", alpha, beta, gamma)


# ============= 5. main入口：baseline对比、训练、可视化 =============
if __name__ == "__main__":
    # 1) 并行训练 => run_experiments() 里会自动可视化
    run_experiments()

    # 如果你只想测试/可视化某个单独模型, 可手动调用:
    # visualize_model_results(exp_id=1, algo="A2C", alpha=3e-8, beta=8.5e-7, gamma=1e-9)
