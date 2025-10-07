#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
main_RL_task2_910.py

任务2：将 407 版的动力学方程替换为“717版”的 **双β** 动力学，并修正所有相关依赖、缺失函数与可视化流程。
功能包含：
1) 数据预处理（场馆/比赛档期）
2) SEIJRD 网络（E、I、J 均为传染态；场馆内附加传播；通勤节律 8/18 点）
3) 多场馆-容量决策环境（动作维度 = n_venues*3，不同时段容量比例）
4) A2C 训练与模型保存
5) 基线（固定场馆半开）与固定比例策略对比
6) 完整可视化：日新增 I、累计 I、日奖励、累计奖励
7) 兼容旧入口：precompute_baseline / visualize_model_results / run_experiments
"""

import os
import csv
import json
import time
import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# gym & sb3
import gym
from gym import spaces
from gym.utils import seeding
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv

# --------------------- 路径 & 全局 ----------------------
os.makedirs("./models", exist_ok=True)
os.makedirs("./results", exist_ok=True)
os.makedirs("./visualizations", exist_ok=True)

FILE_PATH_JSON = "data/end_copy.json"                     # 场馆/赛事排期
BASELINE_CSV    = "data/daily_overall_EI_delta.csv"       # 逐日基线 ΔE / ΔI
MAX_POP = 1e12
# 网络节点数量（抽象人群节点 / 不是 23 区），与旧代码一致保持 50
NUM = 50  # :contentReference[oaicite:4]{index=4}

# ============== “717版” 双β 动力学参数（E、I+J 传染） ==============
# 与你们 717 版一致：beta_1 (E 造成的暴露)，beta_2 (I+J 造成的暴露)
DYN = {
    "beta_1": 0.0614,
    "beta_2": 0.0696,
    "sigma_i": 0.01378,  # E -> I
    "sigma_j": 0.03953,  # E -> J
    "gamma_i": 0.0496,   # I -> R
    "gamma_j": 0.0376,   # J -> R
    "mu_i": 2e-5,        # I -> D
    "mu_j": 0.00027,     # J -> D
    "rho": 8.62e-05      # 场馆内接触概率（E+I+J 传播） :contentReference[oaicite:5]{index=5}
}

# --------------------- 赛事/场馆预处理 ----------------------
def process_competitions(data_json: List[Dict]) -> List[Tuple[int,int,int,int,int]]:
    """
    读取 JSON 赛事时间，映射为 (venue_id, start_tick, end_tick, capacity, slot)
    slot: 0->(8,11), 1->(13,17), 2->(19,22)
    """
    slot_mapping = {0: (8, 11), 1: (13, 17), 2: (19, 22)}
    year = 2021
    earliest = None
    for venue in data_json:
        for t in venue["time"]:
            d = datetime.date(year, t["month"], t["day"])
            if earliest is None or d < earliest:
                earliest = d

    comps = []
    for venue in data_json:
        vid = venue["venue_id"]
        cap = int(venue["capacity"].replace(",", ""))
        for t in venue["time"]:
            d = datetime.date(year, t["month"], t["day"])
            days = (d - earliest).days
            slot = t["slot"]
            sh, eh = slot_mapping[slot]
            start_tick = days * 24 + sh
            end_tick   = days * 24 + eh
            comps.append((vid, start_tick, end_tick, cap, slot))
    return comps  # :contentReference[oaicite:6]{index=6}


def process_places(data_json: List[Dict]) -> Dict[int, "Place"]:
    """生成 place 字典：{venue_id -> Place}"""
    places: Dict[int, Place] = {}
    for d in data_json:
        vid = d["venue_id"]
        info = {
            "venue_id": vid,
            "venue_name": d["venue_name"],
            "capacity": int(d["capacity"].replace(",", ""))
        }
        places[vid] = Place(info)
    return places  # :contentReference[oaicite:7]{index=7}


# --------------------- Place / SEIJRD Node & Network ----------------------
class Place:
    """
    audience_from: shape=(node_num, 6) => [S, E, I, J, R, D] 来自各节点进入场馆的人群
    """
    def __init__(self, data: Dict):
        self.id   = data["venue_id"]
        self.name = data["venue_name"]
        self.capacity = data["capacity"]
        self.agenda: List[Tuple[int,int,int,int,int]] = []
        self.endtime: int = -1
        self.audience_from: np.ndarray = np.empty((0, 6), dtype=np.float64)

    def infect(self):
        """
        场馆内一次“接触式”额外传播：
        令 E+I+J 为传染者数，prob = 1 - (1 - rho)^(E+I+J)；
        将总新感染按各节点 S 占比分摊到 audience_from。
        """
        if len(self.audience_from) == 0:
            return
        seijrd_sum = np.sum(self.audience_from, axis=0)  # [S,E,I,J,R,D]
        S_tot = seijrd_sum[0]
        infectious = seijrd_sum[1] + seijrd_sum[2] + seijrd_sum[3]
        if S_tot <= 0 or infectious <= 0:
            return
        try:
            prob = 1 - (1 - DYN["rho"]) ** infectious
        except OverflowError:
            prob = 1.0
        prob = np.clip(prob, 0.0, 1.0)

        new_inf = S_tot * prob
        frac_s = self.audience_from[:, 0] / (S_tot + 1e-12)
        inc = frac_s * new_inf
        self.audience_from[:, 0] -= inc
        self.audience_from[:, 1] += inc  # S->E  :contentReference[oaicite:8]{index=8}


class SEIJRD_Node:
    """
    单节点 SEIJRD：采用“717双β”动力学。
    """
    def __init__(self, state: np.ndarray, total_nodes: int):
        self.state = state.astype(np.float64)  # [S,E,I,J,R,D]
        self.from_node = np.zeros((total_nodes, 6), dtype=np.float64)

    def update(self, dt=1/2400, times=100):
        # 显式 Euler 多次迭代
        for _ in range(times):
            S, E, I, J, R, D_ = self.state
            N = S + E + I + J + R
            if N < 1e-9:
                continue

            beta1 = DYN["beta_1"]
            beta2 = DYN["beta_2"]
            dS = - beta1 * S * E / N - beta2 * S * (I + J) / N
            dE = + beta1 * S * E / N + beta2 * S * (I + J) / N \
                 - DYN["sigma_i"] * E - DYN["sigma_j"] * E
            dI = DYN["sigma_i"] * E - DYN["gamma_i"] * I - DYN["mu_i"] * I
            dJ = DYN["sigma_j"] * E - DYN["gamma_j"] * J - DYN["mu_j"] * J
            dR = DYN["gamma_i"] * I + DYN["gamma_j"] * J
            dD = DYN["mu_i"] * I + DYN["mu_j"] * J

            self.state[0] += dS * dt
            self.state[1] += dE * dt
            self.state[2] += dI * dt
            self.state[3] += dJ * dt
            self.state[4] += dR * dt
            self.state[5] += dD * dt

        self.state = np.clip(self.state, 0, MAX_POP)  # :contentReference[oaicite:9]{index=9}


class SEIJRD_Network:
    def __init__(self, num_nodes: int, init_states: np.ndarray):
        self.node_num = num_nodes
        self.nodes = {i: SEIJRD_Node(init_states[i], num_nodes) for i in range(num_nodes)}
        # 随机通勤矩阵（逐行归一），与旧版保持一致用随机 A
        A = np.random.random((num_nodes, num_nodes))
        self.A = A / np.sum(A, axis=1, keepdims=True)
        self.delta_time = 1/2400

    def morning_commuting(self):
        """
        早 8 点外出：把 [S,E,I,J,R] 按 A 分发到去向节点，D 不流动。
        """
        for i in range(self.node_num):
            SEIJR = self.nodes[i].state[:5].copy()
            for j in range(self.node_num):
                if i == j:  # 自环保留
                    continue
                move = np.minimum(SEIJR * self.A[i, j], self.nodes[i].state[:5])
                self.nodes[j].from_node[i, :5] = move
                self.nodes[i].state[:5] -= move
        # 入站累加
        for j in range(self.node_num):
            inflow = np.sum(self.nodes[j].from_node[:, :5], axis=0)
            self.nodes[j].state[:5] += inflow

    def evening_commuting(self):
        """
        晚 18 点返程：撤回早上记录的 from_node。
        """
        for i in range(self.node_num):
            for j in range(self.node_num):
                if i == j:
                    continue
                # j->i 回家
                self.nodes[i].state += self.nodes[j].from_node[i]
        for k in range(self.node_num):
            self.nodes[k].from_node[:] = 0.0

    def update_network(self):
        for i in range(self.node_num):
            self.nodes[i].update(dt=self.delta_time, times=100)  # :contentReference[oaicite:10]{index=10}


# --------------------- 环境（容量比例控制） ----------------------
class Env:
    """
    保存场馆/比赛 & 网络。capacity_strategy: (n_venues*3,) 每场馆三时段的容量比例。
    """
    def __init__(self):
        self.places: Dict[int, Place] = {}
        self.competitions: List[Tuple[int,int,int,int,int]] = []
        self.network: Optional[SEIJRD_Network] = None
        self.capacity_strategy: Optional[np.ndarray] = None
        self.current_tick = 0

    def init_env(self):
        data_json = json.load(open(FILE_PATH_JSON, "r", encoding="utf-8"))
        self.competitions = process_competitions(data_json)
        self.places = process_places(data_json)
        # 把赛事排进各场馆队列
        for comp in self.competitions:
            self.places[comp[0]].agenda.append(comp)
        for p in self.places.values():
            p.agenda.sort(key=lambda x: x[1])

        # 初始状态（示例与旧版一致）
        init = np.array([[92200, 800, 0, 0, 0, 0]] * NUM, dtype=np.float64)
        self.network = SEIJRD_Network(NUM, init)
        self.current_tick = 0

    def check_competition_start(self):
        for vid, place in self.places.items():
            if not place.agenda:
                continue
            comp = place.agenda[0]
            if comp[1] == self.current_tick:
                place.audience_from = np.zeros((self.network.node_num, 6), dtype=np.float64)
                place.agenda.pop(0)
                place.endtime = comp[2]
                capacity = comp[3]
                slot_id  = comp[4]

                ratio = 1.0
                if self.capacity_strategy is not None:
                    n_v = len(self.capacity_strategy) // 3
                    a2d = self.capacity_strategy.reshape(n_v, 3)
                    # 注意 JSON 的 venue_id 从 1 开始
                    ratio = a2d[vid - 1, slot_id] if (0 <= vid - 1 < n_v) else 1.0
                actual_cap = capacity * ratio

                # 把 [S,E,I,J,R] 按各节点人口占比分配到场馆
                N5 = np.array([np.sum(self.network.nodes[i].state[:5]) for i in range(self.network.node_num)])
                totalN = np.sum(N5) if np.sum(N5) > 0 else 1.0
                for i in range(self.network.node_num):
                    frac = (self.network.nodes[i].state[:5] / totalN) * actual_cap
                    frac = np.minimum(frac, self.network.nodes[i].state[:5])
                    moved6 = np.zeros(6)
                    moved6[:5] = frac
                    self.network.nodes[i].state[:5] -= frac
                    place.audience_from[i] = moved6

                # 进馆后立刻一次接触传播
                place.infect()

    def check_competition_end(self):
        for pl in self.places.values():
            if pl.endtime == self.current_tick:
                for i in range(self.network.node_num):
                    self.network.nodes[i].state[:5] += pl.audience_from[i][:5]
                pl.audience_from = np.empty((0, 6), dtype=np.float64)
                pl.endtime = -1


class MultiVenueSEIREnv(gym.Env):
    """
    每个 step = 1 天（24 小时）。观察量是各节点 SEIJRD 拼接。
    奖励：日收入（与容量成正比） - 惩罚(超基线的 ΔI 与 ΔE)。
    """
    metadata = {"render.modes": []}

    def __init__(self, days=20, n_venues=41, alpha=3e-8, beta=8.5e-7, gamma=1e-9):
        super().__init__()
        self.days = days
        self.n_venues = n_venues
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        # 内部 SEIJRD 环境
        self.seir_env = Env()
        self.seir_env.init_env()

        # 行为空间：各场馆*3时段的容量比例（0~1）
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(n_venues * 3,), dtype=np.float32)
        # 观测空间：NUM * 6
        self.observation_space = spaces.Box(low=0.0, high=MAX_POP, shape=(NUM * 6,), dtype=np.float32)

        # 载入基线 ΔE/ΔI
        self.baseline_deltaE = {}
        self.baseline_deltaI = {}
        self._load_baseline_csv()

        self.current_day = 0
        self.prev_E, self.prev_I = self._get_total_EI()
        self.state = self._get_obs()
        self.np_random = None

    # ---------- baseline 读取 ----------
    def _load_baseline_csv(self):
        if os.path.exists(BASELINE_CSV):
            with open(BASELINE_CSV, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    d = int(row["Day"])
                    self.baseline_deltaE[d] = float(row["DeltaE"])
                    self.baseline_deltaI[d] = float(row["DeltaI"])
        else:
            # 若不存在，默认 0
            for d in range(self.days):
                self.baseline_deltaE[d] = 0.0
                self.baseline_deltaI[d] = 0.0

    # ---------- 工具 ----------
    def _get_total_EI(self) -> Tuple[float, float]:
        E, I = 0.0, 0.0
        for i in range(NUM):
            st = self.seir_env.network.nodes[i].state
            E += st[1]; I += st[2]
        return E, I

    def _get_obs(self) -> np.ndarray:
        obs = []
        for i in range(NUM):
            st = np.clip(self.seir_env.network.nodes[i].state, 0, MAX_POP)
            obs.extend(st)
        return np.array(obs, dtype=np.float32)

    def seed(self, seed=None):
        self.np_random, seed_ = seeding.np_random(seed)
        return [seed_]

    # ---------- RL 接口 ----------
    def reset(self):
        self.seir_env = Env()
        self.seir_env.init_env()
        self.current_day = 0
        self.prev_E, self.prev_I = self._get_total_EI()
        self.state = self._get_obs()
        return self.state

    def step(self, action: np.ndarray):
        # 当天 24 小时循环
        self.seir_env.capacity_strategy = np.asarray(action, dtype=np.float32)
        start_tick = self.current_day * 24
        end_tick   = (self.current_day + 1) * 24

        for hour_tick in range(start_tick, end_tick):
            self.seir_env.current_tick = hour_tick
            self.seir_env.check_competition_start()

            if hour_tick % 24 == 8:
                self.seir_env.network.morning_commuting()

            # 每小时更新动力学
            self.seir_env.network.update_network()

            if hour_tick % 24 == 18:
                self.seir_env.network.evening_commuting()

            self.seir_env.check_competition_end()

        # 计算日新增 E/I
        cur_E, cur_I = self._get_total_EI()
        daily_newE = cur_E - self.prev_E
        daily_newI = cur_I - self.prev_I
        self.prev_E, self.prev_I = cur_E, cur_I

        # 收入：容量*比例*alpha（按三个 slot 求和）
        a2d = action.reshape((self.n_venues, 3))
        total_ratio_capacity = 0.0
        for vid, place in self.seir_env.places.items():
            # 三个时段容量叠加
            total_ratio_capacity += np.sum(a2d[vid - 1]) * place.capacity
        daily_revenue = total_ratio_capacity * self.alpha

        # 惩罚：超过基线的 ΔI, ΔE
        baseI = self.baseline_deltaI.get(self.current_day, 0.0)
        baseE = self.baseline_deltaE.get(self.current_day, 0.0)
        exceedI = max(0.0, daily_newI - baseI)
        exceedE = max(0.0, daily_newE - baseE)

        reward = daily_revenue - (self.beta * exceedI + self.gamma * exceedE)

        # 前进一天
        self.current_day += 1
        done = (self.current_day >= self.days)
        self.state = self._get_obs()
        info = {"newI": float(daily_newI), "newE": float(daily_newE), "day": int(self.current_day)}
        return self.state, float(reward), done, info


# --------------------- 运行若干策略进行对比 ----------------------
def run_baseline(env: MultiVenueSEIREnv) -> pd.DataFrame:
    """
    基线策略：某些场馆 0.5，其他 0。与旧实现保持一致的示意基线。
    """
    env.reset()
    baseline_2d = np.zeros((env.n_venues, 3), dtype=np.float32)
    special_ids = [3, 5, 24, 37, 40]  # 可与历史固定基线一致
    for vid in special_ids:
        baseline_2d[vid - 1] = 0.5

    rec = {"day": [], "new_I": [], "reward": []}
    done = False; day = 0; obs = env.state
    while not done:
        action = baseline_2d.reshape(-1)
        obs, rew, done, info = env.step(action)
        rec["day"].append(day)
        rec["new_I"].append(info["newI"])
        rec["reward"].append(rew)
        day += 1
    return pd.DataFrame(rec)  # :contentReference[oaicite:11]{index=11}


def run_constant_ratio(env: MultiVenueSEIREnv, ratio: float) -> pd.DataFrame:
    """所有场馆、所有时段使用同一比例 ratio"""
    env.reset()
    const_2d = np.ones((env.n_venues, 3), dtype=np.float32) * ratio

    rec = {"day": [], "new_I": [], "reward": []}
    done = False; day = 0; obs = env.state
    while not done:
        action = const_2d.reshape(-1)
        obs, rew, done, info = env.step(action)
        rec["day"].append(day)
        rec["new_I"].append(info["newI"])
        rec["reward"].append(rew)
        day += 1
    return pd.DataFrame(rec)  # :contentReference[oaicite:12]{index=12}


def compare_capacity_strategies(env: MultiVenueSEIREnv, model: A2C,
                                exp_id: int, version: int,
                                alpha: float, beta: float, gamma: float,
                                output_dir="./visualizations"):
    """
    生成 2×2 对比图：日新增 I、累计 I、日奖励、累计奖励
    """
    print(f"开始对比不同容量策略 (实验{exp_id} v{version})...")

    # baseline
    base_df = run_baseline(env)
    base_df["cum_new_I"] = base_df["new_I"].cumsum()
    base_df["cum_reward"] = base_df["reward"].cumsum()

    # 固定比例若干档
    ratio_data = {}
    for r in [0.0, 0.3, 0.5, 1.0]:
        df = run_constant_ratio(env, r)
        df["cum_new_I"] = df["new_I"].cumsum()
        df["cum_reward"] = df["reward"].cumsum()
        ratio_data[r] = df

    # RL 策略
    env.reset()
    rl_rec = {"day": [], "new_I": [], "reward": []}
    obs = env.state; done = False; day = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, rew, done, info = env.step(action)
        rl_rec["day"].append(day)
        rl_rec["new_I"].append(info["newI"])
        rl_rec["reward"].append(rew)
        day += 1
    rl_df = pd.DataFrame(rl_rec)
    rl_df["cum_new_I"] = rl_df["new_I"].cumsum()
    rl_df["cum_reward"] = rl_df["reward"].cumsum()

    # 绘图
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    ax1, ax2, ax3, ax4 = axes.flatten()

    ax1.plot(base_df["day"], base_df["new_I"], label="Baseline", lw=2)
    for r, d in ratio_data.items():
        ax1.plot(d["day"], d["new_I"], label=f"ratio {r}", alpha=0.8)
    ax1.plot(rl_df["day"], rl_df["new_I"], label="RL", lw=2, color="red")
    ax1.set_title("Daily New Infections"); ax1.set_xlabel("Day"); ax1.set_ylabel("newI"); ax1.legend(); ax1.grid(True)

    ax2.plot(base_df["day"], base_df["cum_new_I"], label="Baseline", lw=2)
    for r, d in ratio_data.items():
        ax2.plot(d["day"], d["cum_new_I"], label=f"ratio {r}", alpha=0.8)
    ax2.plot(rl_df["day"], rl_df["cum_new_I"], label="RL", lw=2, color="red")
    ax2.set_title("Cumulative New Infections"); ax2.set_xlabel("Day"); ax2.set_ylabel("cum newI"); ax2.legend(); ax2.grid(True)

    ax3.plot(base_df["day"], base_df["reward"], label="Baseline", lw=2)
    for r, d in ratio_data.items():
        ax3.plot(d["day"], d["reward"], label=f"ratio {r}", alpha=0.8)
    ax3.plot(rl_df["day"], rl_df["reward"], label="RL", lw=2, color="red")
    ax3.set_title("Daily Reward"); ax3.set_xlabel("Day"); ax3.set_ylabel("Reward"); ax3.legend(); ax3.grid(True)

    ax4.plot(base_df["day"], base_df["cum_reward"], label="Baseline", lw=2)
    for r, d in ratio_data.items():
        ax4.plot(d["day"], d["cum_reward"], label=f"ratio {r}", alpha=0.8)
    ax4.plot(rl_df["day"], rl_df["cum_reward"], label="RL", lw=2, color="red")
    ax4.set_title("Cumulative Reward"); ax4.set_xlabel("Day"); ax4.set_ylabel("Reward"); ax4.legend(); ax4.grid(True)

    fig.suptitle(f"Strategy Comparison (Exp {exp_id} v{version}, α={alpha:.1e}, β={beta:.1e}, γ={gamma:.1e})", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    os.makedirs(output_dir, exist_ok=True)
    out_png = f"{output_dir}/exp_{exp_id}_continued_v{version}_strategy_comparison.png"
    plt.savefig(out_png)
    plt.close(fig)
    print(f"✅ 策略对比图已保存: {out_png}")  # :contentReference[oaicite:13]{index=13}


# --------------------- 兼容旧入口：基线预计算 + 可视化 ----------------------
def precompute_baseline(days: int, n_venues: int, alpha: float):
    """
    兼容老代码的接口：返回两个 dict
      - base_infection_dict[day] = (DeltaE_base + DeltaI_base)
      - base_revenue_dict[day]  = baseline 策略的日收入
    若 CSV 缺失则以 0 代替。
    """
    # 读取 ΔE、ΔI
    dE, dI = {}, {}
    if os.path.exists(BASELINE_CSV):
        with open(BASELINE_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                d = int(row["Day"])
                dE[d] = float(row["DeltaE"]); dI[d] = float(row["DeltaI"])
    base_infection = {d: dE.get(d, 0.0) + dI.get(d, 0.0) for d in range(days)}

    # 用一个 env 跑“baseline 容量=0.5（少数场馆）”估计收入
    env_tmp = MultiVenueSEIREnv(days=days, n_venues=n_venues, alpha=alpha, beta=0.0, gamma=0.0)
    base_df = run_baseline(env_tmp)
    base_revenue = {i: float(base_df["reward"].iloc[i]) for i in range(len(base_df))}
    return base_infection, base_revenue  # （留给需要此接口的外层调用使用）


def visualize_model_results(exp_id: int, alpha: float, beta: float, gamma: float, version: int = 3):
    """
    兼容老入口：读取模型并画对比图。若模型不存在则先训练再画。
    """
    model_path = f"./results/exp_{exp_id}_A2C_continued_v{version}_a{alpha}_b{beta}_g{gamma}.zip"
    if os.path.exists(model_path):
        model = A2C.load(model_path, device="cpu")
    else:
        # 没有就现场训练一个轻量模型
        model = train_and_save_model(exp_id, alpha, beta, gamma,
                                     total_timesteps=200_000, version=version)

    env = MultiVenueSEIREnv(days=20, n_venues=41, alpha=alpha, beta=beta, gamma=gamma)
    compare_capacity_strategies(env, model, exp_id, version, alpha, beta, gamma)


# --------------------- 训练 & 批量实验 ----------------------
def train_and_save_model(exp_id: int, alpha: float, beta: float, gamma: float,
                         total_timesteps: int = 1_000_000, version: int = 3) -> A2C:
    """
    训练 A2C 并保存，文件名对齐旧习惯。
    """
    env_maker = lambda: MultiVenueSEIREnv(days=20, n_venues=41, alpha=alpha, beta=beta, gamma=gamma)
    vec_env = DummyVecEnv([env_maker])
    model = A2C("MlpPolicy", vec_env, device="cpu", verbose=0,
                learning_rate=7e-4, n_steps=5, gamma=0.99, ent_coef=0.01,
                vf_coef=0.5, max_grad_norm=0.5)
    model.learn(total_timesteps=total_timesteps)
    out = f"./results/exp_{exp_id}_A2C_continued_v{version}_a{alpha}_b{beta}_g{gamma}.zip"
    model.save(out)
    vec_env.close()
    print(f"[exp {exp_id}] A2C saved => {out}")
    return model


def run_experiments():
    """
    批量跑实验（与旧主程序习惯相近），默认只做可视化。
    """
    exp_ids = [2, 5, 10, 11, 12, 17, 22, 26]
    exp_params = {
        2:  {"alpha": 1.0e-8, "beta": 8.241e-07, "gamma": 1.0e-9},
        5:  {"alpha": 1.0e-8, "beta": 8.966e-07, "gamma": 1.0e-9},
        10: {"alpha": 1.0e-8, "beta": 1.017e-06, "gamma": 1.0e-9},
        11: {"alpha": 1.0e-8, "beta": 1.041e-06, "gamma": 1.0e-9},
        12: {"alpha": 1.0e-8, "beta": 1.066e-06, "gamma": 1.0e-9},
        17: {"alpha": 1.0e-8, "beta": 1.186e-06, "gamma": 1.0e-9},
        22: {"alpha": 1.0e-8, "beta": 1.307e-06, "gamma": 1.0e-9},
        26: {"alpha": 1.0e-8, "beta": 1.403e-06, "gamma": 1.0e-9},
    }
    version = 3

    for eid in exp_ids:
        a = exp_params[eid]["alpha"]
        b = exp_params[eid]["beta"]
        g = exp_params[eid]["gamma"]
        print(f"\n===== 处理实验 {eid} v{version} =====")
        # 先可视化（若模型不存在会在 visualize 内部触发训练）
        visualize_model_results(eid, a, b, g, version)


# --------------------- CLI ----------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Task2 - MultiVenue RL with dual-β SEIJRD")
    parser.add_argument("--mode", choices=["train", "viz", "all"], default="viz",
                        help="train=只训练; viz=只可视化(缺模型会轻量训练); all=训练后可视化")
    parser.add_argument("--exp-id", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=1.0e-8)
    parser.add_argument("--beta", type=float, default=8.241e-07)
    parser.add_argument("--gamma", type=float, default=1.0e-9)
    parser.add_argument("--timesteps", type=int, default=300000)
    args = parser.parse_args()

    if args.mode == "train":
        train_and_save_model(args.exp_id, args.alpha, args.beta, args.gamma,
                             total_timesteps=args.timesteps)
    elif args.mode == "viz":
        visualize_model_results(args.exp_id, args.alpha, args.beta, args.gamma)
    elif args.mode == "all":
        m = train_and_save_model(args.exp_id, args.alpha, args.beta, args.gamma,
                                 total_timesteps=args.timesteps)
        env = MultiVenueSEIREnv(days=20, n_venues=41, alpha=args.alpha, beta=args.beta, gamma=args.gamma)
        compare_capacity_strategies(env, m, args.exp_id, 3, args.alpha, args.beta, args.gamma)
    else:
        run_experiments()
