#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import json
import time
import datetime
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # 非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import multiprocessing
from multiprocessing import Pool
import torch
import gym
from gym import spaces
from gym.utils import seeding

# Stable-Baselines3
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor

# ----------- 保证目录存在 -----------
os.makedirs("./models", exist_ok=True)
os.makedirs("./logs", exist_ok=True)
os.makedirs("./results", exist_ok=True)
os.makedirs("./visualizations", exist_ok=True)
os.makedirs("./tensorboard", exist_ok=True)

# ----------- 硬件与全局参数 -----------
NUM_GPUS = 8
NUM_CPU_CORES = 140
MAX_POP = 1e12

# ----------- 老代码参数 -----------
backup_paras = {
    "beta": 0.155,
    "gamma_i": 0.079,
    "gamma_j": 0.031,
    "sigma_i": 0.0299,
    "sigma_j": 0.0156,
    "mu_i": 1.95e-5,
    "mu_j": 0.00025,
    "rho": 8.48e-5,  # 接触传染概率
    "initial_infect": 500,  # 初始感染人数
}

DEFAULT_IS_STATES = False
MIN_SELF_ACTIVITY = 0.3

file_path = "data/end_copy.json"


# ================== CSV 读取辅助 ==================
def Input_Population(fp="data/tokyo_population.csv"):
    df = pd.read_csv(fp, header=None,
                     names=["code", "ward_jp", "ward_en", "dummy", "population"])
    df["population"] = df["population"].astype(int)
    return df["population"].tolist()


def Input_Matrix(fp="data/tokyo_commuting_flows_with_intra.csv"):
    df = pd.read_csv(fp, index_col=0)
    df = df.apply(pd.to_numeric, errors='coerce')
    return df.values


def Input_Intial(fp="data/SEIJRD.csv"):
    df = pd.read_csv(fp, index_col=None)
    df = df.astype(float)
    return df.values


# ================== 老代码 process 函数 ==================
def process_competitions(data_file):
    slot_mapping = {0: (8, 11), 1: (13, 17), 2: (19, 22)}
    year = 2021
    earliest = None
    for venue in data_file:
        for t in venue["time"]:
            edate = datetime.date(year, t["month"], t["day"])
            if earliest is None or edate < earliest:
                earliest = edate
    comps = []
    for venue in data_file:
        vid = venue["venue_id"]
        cap = int(venue["capacity"].replace(',', ''))
        for t in venue["time"]:
            edate = datetime.date(year, t["month"], t["day"])
            diff = (edate - earliest).days
            slot = t["slot"]
            sh, eh = slot_mapping[slot]
            st_tick = diff * 24 + sh
            ed_tick = diff * 24 + eh
            comps.append((vid, st_tick, ed_tick, cap))
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


class Place:
    def __init__(self, data: Dict):
        self.id = data["venue_id"]
        self.name = data["venue_name"]
        self.capacity = data["capacity"]
        self.agenda = []
        self.endtime = None
        self.audience_from = []

    def infect(self, paras):
        """
        E+I+J 皆可传染. if S_tot<1e-3 => return
        """
        if len(self.audience_from) == 0:
            return
        arr = np.sum(self.audience_from, axis=0)  # shape(6,)
        inf_num = arr[1] + arr[2] + arr[3]
        if inf_num <= 0:
            return
        prob = 1 - (1 - paras["rho"]) ** inf_num
        prob = np.clip(prob, 0, 1)

        S_tot = arr[0]
        if S_tot < 1e-3:
            return
        new_inf = S_tot * prob

        ratio = self.audience_from[:, 0] / (S_tot + 1e-12)
        new_inf_each = ratio * new_inf
        self.audience_from[:, 0] -= new_inf_each
        self.audience_from[:, 1] += new_inf_each


# ================== 老代码 SEIR_Node + SEIR_Network ==================
class SEIR_Node:
    def __init__(self, state, total_nodes):
        self.state = state.astype(float)
        self.total_nodes = total_nodes
        self.from_node = np.zeros((total_nodes, 6))
        self.to_node = np.zeros((total_nodes, 5))

    def update_seir(self, delta_time=1 / 2400, times=100, paras=backup_paras):
        """
        dS,dE,dI,dJ,dR,dD => self.state += ...
        同时更新 from_node
        """
        S, E, I, J, R, D_ = self.state
        N = np.sum(self.state)
        if N > 0:
            dS = - paras["beta"] * S * E / N
            dE = paras["beta"] * S * E / N - paras["sigma_i"] * E - paras["sigma_j"] * E
            dI = paras["sigma_i"] * E - paras["gamma_i"] * I - paras["mu_i"] * I
            dJ = paras["sigma_j"] * E - paras["gamma_j"] * J - paras["mu_j"] * J
            dR = paras["gamma_i"] * I + paras["gamma_j"] * J
            dD = paras["mu_i"] * I + paras["mu_j"] * J
        else:
            dS = dE = dI = dJ = dR = dD = 0.0

        for _ in range(times):
            self.state += np.array([dS, dE, dI, dJ, dR, dD]) * delta_time
            # 更新 from_node
            Nvalue = np.sum(self.from_node, axis=1)  # shape:(total_nodes,)
            if N > 0:
                ratio = (Nvalue / N)[:, None]
            else:
                ratio = np.zeros((self.total_nodes, 1))
            inc = np.array([dS, dE, dI, dJ, dR, dD])[None, :] * ratio * delta_time
            self.from_node += inc

        self.state = np.clip(self.state, 0, MAX_POP)
        self.from_node = np.clip(self.from_node, 0, MAX_POP)


class SEIR_Network:
    def __init__(self, num, states, matrix=None, paras=backup_paras):
        self.node_num = num
        self.nodes = {}
        for i in range(num):
            self.nodes[i] = SEIR_Node(states[i], num)
        if matrix is not None:
            row_sum = matrix.sum(axis=1, keepdims=True)
            self.A = matrix / row_sum
        else:
            A_ = np.random.rand(num, num)
            A_ /= A_.sum(axis=1, keepdims=True)
            self.A = A_
        self.paras = paras
        self.delta_time = 1 / 2400

    def morning_commuting(self):
        for i in range(self.node_num):
            SEIJR = self.nodes[i].state[:5]
            for j in range(self.node_num):
                self.nodes[i].to_node[j] = SEIJR * self.A[i][j]
        # 倾巢而出
        for i in range(self.node_num):
            self.nodes[i].state[:5] = 0.0

        # 抵达
        for i in range(self.node_num):
            for j in range(self.node_num):
                self.nodes[j].state[:5] += self.nodes[i].to_node[j]
                self.nodes[j].from_node[i, :5] = self.nodes[i].to_node[j]

    def evening_commuting(self):
        for i in range(self.node_num):
            for j in range(self.node_num):
                if i == j:
                    continue
                self.nodes[j].state -= self.nodes[j].from_node[i]
                self.nodes[i].state += self.nodes[j].from_node[i]
        for i in range(self.node_num):
            self.nodes[i].from_node = np.zeros((self.node_num, 6))
            self.nodes[i].to_node = np.zeros((self.node_num, 5))

    def update_network(self):
        for i in range(self.node_num):
            self.nodes[i].update_seir(self.delta_time, 100, self.paras)


# ================== Env (类似老代码) ==================
class Env:
    def __init__(self, is_states=DEFAULT_IS_STATES):
        self.is_states = is_states
        self.num = 23
        self.network = None
        self.competitions = []
        self.places = {}
        self.matrix = None
        self.origin_matrix = None
        self.current_tick = 0
        self.paras = backup_paras.copy()

    def init_env(self):
        data_json = json.load(open(file_path, "r", encoding="utf-8"))
        self.competitions = process_competitions(data_json)
        self.places = process_places(data_json)
        for c in self.competitions:
            vid = c[0]
            self.places[vid].agenda.append(c)
        for pl in self.places.values():
            pl.agenda.sort(key=lambda x: x[1])

        # 初始化states
        if self.is_states:
            init_states = Input_Intial("data/SEIJRD.csv")
        else:
            # 人口 + initial_infect
            pops = Input_Population("data/tokyo_population.csv")
            init_states = np.zeros((self.num, 6))
            for i in range(self.num):
                S_ = pops[i] - self.paras["initial_infect"] * 4.5 / self.num
                E_ = self.paras["initial_infect"] * 2 / self.num
                I_ = self.paras["initial_infect"] * 1 / self.num
                J_ = self.paras["initial_infect"] * 1.5 / self.num
                init_states[i] = np.array([S_, E_, I_, J_, 0, 0], dtype=float)

        # 通勤矩阵
        try:
            mat = Input_Matrix("data/tokyo_commuting_flows_with_intra.csv")
            row_sum = mat.sum(axis=1, keepdims=True)
            mat = mat / row_sum
            self.origin_matrix = mat.copy()
            self.matrix = mat.copy()
        except:
            self.origin_matrix = np.eye(self.num)
            self.matrix = np.eye(self.num)

        self.network = SEIR_Network(self.num, init_states, self.matrix, self.paras)
        self.current_tick = 0

    def check_competition_start(self):
        for pid, place in self.places.items():
            if not place.agenda:
                continue
            comp = place.agenda[0]
            if comp[1] == self.current_tick:
                place.audience_from = np.zeros((self.num, 6))
                place.agenda.pop(0)
                place.endtime = comp[2]
                capacity = comp[3]

                sums = [np.sum(self.network.nodes[i].state[:4]) for i in range(self.num)]
                sums = np.array(sums)
                tot = np.sum(sums)
                if tot < 1e-9:
                    tot = 1.0
                for i in range(self.num):
                    portion = self.network.nodes[i].state[:5] * (capacity / tot)
                    portion = np.minimum(portion, self.network.nodes[i].state[:5])
                    place.audience_from[i, :5] = portion
                    self.network.nodes[i].state[:5] -= portion
                place.infect(self.paras)

    def check_competition_end(self):
        for pid, place in self.places.items():
            if place.endtime == self.current_tick:
                for i in range(self.num):
                    self.network.nodes[i].state[:5] += place.audience_from[i, :5]
                place.audience_from = []
                place.endtime = None

    def modify_matrix(self, matrix):
        row_sum = matrix.sum(axis=1, keepdims=True)
        matrix_ = matrix / row_sum
        self.network.A = matrix_.copy()
        self.matrix = matrix_.copy()

        ratio = (np.sum(self.matrix) - np.trace(self.matrix)) / (
                    np.sum(self.origin_matrix) - np.trace(self.origin_matrix))
        new_beta = 0.05 + (self.paras["beta"] - 0.05) * ratio
        self.network.paras["beta"] = max(0.01, new_beta)


# ================== 强化学习环境 (每 step=1小时) ==================
class CommuneMatrixEnv(gym.Env):
    """
    1小时更新(=1 step).
    1天=24步 => days*N => episode长度
    """

    def __init__(self, days=20, is_states=False):
        super().__init__()
        self.days = days
        self.is_states = is_states
        self.seir_env = Env(is_states=self.is_states)
        self.seir_env.init_env()
        self.num = self.seir_env.num

        # obs: 23区x6 + day + hour
        obs_dim = self.num * 6 + 2
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        # action: 23x23
        self.action_space = spaces.Box(low=0, high=1, shape=(self.num * self.num,), dtype=np.float32)

        self.current_hour = 0
        self.current_day = 0
        self.episode_steps = 0
        self.max_steps = 24 * self.days

        self.original_matrix = self.seir_env.origin_matrix.copy()
        self.np_random = None

    def seed(self, seed=None):
        self.np_random, seed_ = seeding.np_random(seed)
        return [seed_]

    def reset(self):
        self.seir_env = Env(is_states=self.is_states)
        self.seir_env.init_env()
        self.current_hour = 0
        self.current_day = 0
        self.episode_steps = 0
        self.original_matrix = self.seir_env.origin_matrix.copy()
        return self._get_obs()

    def step(self, action):
        matrix = self._process_action(action)
        self.seir_env.modify_matrix(matrix)

        self.seir_env.check_competition_start()
        if self.current_hour == 8:
            self.seir_env.network.morning_commuting()

        prev_states = np.array([nd.state.copy() for nd in self.seir_env.network.nodes.values()])

        self.seir_env.network.update_network()

        if self.current_hour == 18:
            self.seir_env.network.evening_commuting()

        self.seir_env.check_competition_end()

        curr_states = np.array([nd.state for nd in self.seir_env.network.nodes.values()])
        reward = self._calc_reward(prev_states, curr_states, matrix)

        self.current_hour += 1
        if self.current_hour >= 24:
            self.current_hour = 0
            self.current_day += 1

        self.episode_steps += 1
        done = (self.episode_steps >= self.max_steps)
        obs = self._get_obs()

        info = {}
        prev_infect = np.sum(prev_states[:, 1:4])
        curr_infect = np.sum(curr_states[:, 1:4])
        info['infections'] = max(0, curr_infect - prev_infect)

        prev_death = np.sum(prev_states[:, 5])
        curr_death = np.sum(curr_states[:, 5])
        info['deaths'] = max(0, curr_death - prev_death)

        return obs, reward, done, info

    def _get_obs(self):
        arr = []
        for i in range(self.num):
            arr.append(self.seir_env.network.nodes[i].state)
        arr = np.concatenate(arr, axis=0)
        return np.concatenate([arr, [self.current_day, self.current_hour]]).astype(np.float32)

    def _process_action(self, action):
        mat = action.reshape(self.num, self.num)
        mat = np.maximum(mat, 0)
        row_sum = mat.sum(axis=1, keepdims=True)
        mat = mat / (row_sum + 1e-12)
        for i in range(self.num):
            if mat[i, i] < MIN_SELF_ACTIVITY:
                deficit = MIN_SELF_ACTIVITY - mat[i, i]
                non_diag = 1.0 - mat[i, i]
                if non_diag > 0:
                    scale = (non_diag - deficit) / non_diag
                    for j in range(self.num):
                        if j != i:
                            mat[i, j] *= scale
                    mat[i, i] = MIN_SELF_ACTIVITY
        return mat

    def _calc_reward(self, prev_states, curr_states, mat):
        prev_in = np.sum(prev_states[:, 1:4])
        curr_in = np.sum(curr_states[:, 1:4])
        new_inf = max(0, curr_in - prev_in)

        prev_d = np.sum(prev_states[:, 5])
        curr_d = np.sum(curr_states[:, 5])
        new_d = max(0, curr_d - prev_d)

        ori_c = np.sum(self.original_matrix) - np.trace(self.original_matrix)
        now_c = np.sum(mat) - np.trace(mat)
        ratio = now_c / (ori_c + 1e-12)
        total_pop = np.sum(curr_states[:, :5])

        w1 = 1.0
        w2 = 0.5
        w3 = 3.0
        reward = -w1 * new_inf + w2 * total_pop * ratio - w3 * new_d
        reward /= 1000000
        return reward


# ------------------- 多基准策略 (fixed matrix) -------------------
BASELINE_STRATEGIES = {}


def baseline_original(env):
    """使用env.origin_matrix不变"""
    return env.origin_matrix.copy()


def baseline_diagonal(env):
    """对角线=1，其余=0"""
    mat = np.zeros_like(env.origin_matrix)
    np.fill_diagonal(mat, 1.0)
    return mat


def baseline_neighbor(env):
    """保留邻居(>0即邻居) 并行归一化，对角线>=MIN_SELF_ACTIVITY"""
    mat = (env.origin_matrix > 1e-9).astype(float)
    row_sum = mat.sum(axis=1, keepdims=True)
    mat = mat / (row_sum + 1e-12)
    for i in range(env.num):
        if mat[i, i] < MIN_SELF_ACTIVITY:
            deficit = MIN_SELF_ACTIVITY - mat[i, i]
            non_diag = 1.0 - mat[i, i]
            if non_diag > 0:
                scale = (non_diag - deficit) / non_diag
                for j in range(env.num):
                    if j != i:
                        mat[i, j] *= scale
            mat[i, i] = MIN_SELF_ACTIVITY
    return mat


def baseline_random50(env):
    """对角线保留0.5，其余随机分布"""
    n = env.num
    m = np.random.rand(n, n)
    rs = m.sum(axis=1, keepdims=True)
    m /= (rs + 1e-12)
    for i in range(n):
        diag = 0.5
        diff = diag - m[i, i]
        if diff > 0:
            non_diag = 1.0 - m[i, i]
            scale = (non_diag - diff) / non_diag if non_diag > 0 else 1.0
            for j in range(n):
                if j != i:
                    m[i, j] *= scale
            m[i, i] = diag
    return m


BASELINE_STRATEGIES["original"] = baseline_original
BASELINE_STRATEGIES["diagonal"] = baseline_diagonal
BASELINE_STRATEGIES["neighbor"] = baseline_neighbor
BASELINE_STRATEGIES["random50"] = baseline_random50


# ------------------- 可视化图函数 -------------------

def create_strategy_comparison(rl_results, baseline_results):
    """Create comparison charts for RL vs baseline strategies on key metrics"""
    plt.figure(figsize=(15, 20))
    metrics = ["cumulative_infections", "cumulative_deaths", "average_commute", "cumulative_reward"]
    titles = ["Cumulative Infections", "Cumulative Deaths", "Average Commute Ratio", "Cumulative Reward"]

    for i, (metric, title) in enumerate(zip(metrics, titles)):
        plt.subplot(4, 1, i + 1)
        # Plot RL results
        for algo, exp_id, results in rl_results:
            label = f"{algo}_exp{exp_id}"
            plt.plot(results[f'hourly_{metric}'], label=label)

        # Plot baseline results
        for baseline, results in baseline_results:
            plt.plot(results[f'hourly_{metric}'], label=f"baseline_{baseline}",
                     linestyle='--', linewidth=2)

        plt.title(title)
        plt.legend()
        plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("./visualizations/strategy_comparison.png")
    plt.close()


def create_final_state_comparison(rl_results, baseline_results):
    """Create bar charts comparing final states across different strategies"""
    metrics = ["total_infections", "total_deaths", "final_commute_ratio"]
    titles = ["Total Infections", "Total Deaths", "Average Commute Ratio"]

    # Extract data
    strategies = []
    data = {m: [] for m in metrics}

    for algo, exp_id, results in rl_results:
        strategies.append(f"{algo}_exp{exp_id}")
        data["total_infections"].append(np.sum(results['hourly_infections']))
        data["total_deaths"].append(np.sum(results['hourly_deaths']))
        data["final_commute_ratio"].append(np.mean(results['hourly_commute_ratio']))

    for baseline, results in baseline_results:
        strategies.append(f"baseline_{baseline}")
        data["total_infections"].append(np.sum(results['hourly_infections']))
        data["total_deaths"].append(np.sum(results['hourly_deaths']))
        data["final_commute_ratio"].append(np.mean(results['hourly_commute_ratio']))

    # Create bar chart
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        axes[i].bar(strategies, data[metric])
        axes[i].set_title(title)
        axes[i].set_xticklabels(strategies, rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig("./visualizations/final_state_comparison.png")
    plt.close()


def visualize_matrix_evolution(model_path, algo, exp_id):
    """Visualize the evolution of commuting matrices over time"""
    if algo == "PPO":
        model = PPO.load(model_path)
    else:
        model = A2C.load(model_path)

    env = CommuneMatrixEnv(days=20)
    obs = env.reset()

    fig, axes = plt.subplots(5, 4, figsize=(20, 15))
    axes = axes.flatten()

    orig_matrix = env.original_matrix

    day = 0
    idx_plot = 0
    while day < 20:
        # Predict once a day, then step 24 hours
        action, _ = model.predict(obs, deterministic=True)
        for _ in range(24):
            obs, _, done, _ = env.step(action)
            if done:
                break

        if day % 4 == 0:
            if idx_plot < len(axes) // 2:  # to avoid out of range
                # left heatmap => original
                sns.heatmap(orig_matrix, ax=axes[idx_plot * 2], cmap="YlGnBu",
                            vmin=0, vmax=1, cbar=True)
                axes[idx_plot * 2].set_title("Original Matrix")

                # right heatmap => new matrix
                mat = env.seir_env.matrix
                sns.heatmap(mat, ax=axes[idx_plot * 2 + 1], cmap="YlGnBu",
                            vmin=0, vmax=1, cbar=True)
                axes[idx_plot * 2 + 1].set_title(f"Day {day} Policy Matrix")
                idx_plot += 1

        day += 1

    plt.tight_layout()
    plt.savefig(f"./visualizations/{algo}_exp{exp_id}_matrix_evolution.png")
    plt.close()


def create_algorithm_radar_chart(rl_results, baseline_results, max_infections, max_deaths):
    """Create radar chart comparing different algorithms across multiple metrics"""
    metrics = ["Infection Control", "Death Prevention", "Commute Preservation",
               "Computational Efficiency", "Policy Stability"]

    algorithms = []
    values = []

    for algo, exp_id, results in rl_results:
        algorithms.append(f"{algo}_exp{exp_id}")
        # 取最后一次episode => hourly_infections, hourly_deaths, hourly_commute_ratio
        inf_sum = np.sum(results['hourly_infections'])
        death_sum = np.sum(results['hourly_deaths'])
        commute_avg = np.mean(results['hourly_commute_ratio'])
        # 简单写死
        efficiency = 0.8 if algo == "A2C" else 0.6
        stability = 0.9 - np.std(results['hourly_commute_ratio'])

        vals = [
            1 - inf_sum / max_infections,  # Infection control
            1 - death_sum / max_deaths,  # Death prevention
            commute_avg,  # Commute preservation
            efficiency,  # Computational Efficiency
            stability  # Policy Stability
        ]
        values.append(vals)

    for baseline, results in baseline_results:
        algorithms.append(f"baseline_{baseline}")
        inf_sum = np.sum(results['hourly_infections'])
        death_sum = np.sum(results['hourly_deaths'])
        commute_avg = np.mean(results['hourly_commute_ratio'])
        # baseline -> highest eff
        efficiency = 1.0
        # stability
        st = 1.0 if baseline != "random50" else 0.5

        vals = [
            1 - inf_sum / max_infections,
            1 - death_sum / max_deaths,
            commute_avg,
            efficiency,
            st
        ]
        values.append(vals)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, polar=True)

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    for i, algorithm in enumerate(algorithms):
        data = values[i] + values[i][:1]
        ax.plot(angles, data, linewidth=2, label=algorithm)
        ax.fill(angles, data, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.05))
    plt.savefig("./visualizations/algorithm_radar_chart.png")
    plt.close()


def visualize_district_infections(rl_model, baseline_model, days=20):
    """Visualize infection comparison between RL and baseline strategies by district"""
    env_rl = CommuneMatrixEnv(days=days)
    env_bl = CommuneMatrixEnv(days=days)
    obs_rl = env_rl.reset()
    obs_bl = env_bl.reset()

    rl_district_infections = [[] for _ in range(env_rl.num)]
    bl_district_infections = [[] for _ in range(env_bl.num)]

    done_rl = False
    done_bl = False
    while not (done_rl and done_bl):
        if not done_rl:
            action_rl, _ = rl_model.predict(obs_rl, deterministic=True)
            obs_rl, _, done_rl, _ = env_rl.step(action_rl)
            for i in range(env_rl.num):
                st = env_rl.seir_env.network.nodes[i].state
                rl_district_infections[i].append(np.sum(st[1:4]))

        if not done_bl:
            fix_matrix = baseline_model(env_bl.seir_env)  # call the baseline function
            action_bl = fix_matrix.flatten()
            obs_bl, _, done_bl, _ = env_bl.step(action_bl)
            for i in range(env_bl.num):
                st = env_bl.seir_env.network.nodes[i].state
                bl_district_infections[i].append(np.sum(st[1:4]))

    plt.figure(figsize=(15, 15))
    for i in range(5):  # top5 district
        plt.subplot(5, 1, i + 1)
        plt.plot(rl_district_infections[i], label='RL Policy')
        plt.plot(bl_district_infections[i], label='Baseline Policy', linestyle='--')
        plt.title(f'District {i + 1} Infections')
        plt.legend()
        plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("./visualizations/district_infection_comparison.png")
    plt.close()


# ------------------- 训练 & 评估 & baseline -------------------
class CustomCallback(BaseCallback):
    def __init__(self, exp_id=0, algo="", log_interval=2000, verbose=1):
        super().__init__(verbose)
        self.exp_id = exp_id
        self.algo = algo
        self.log_interval = log_interval
        self.start_time = None
        self.last_save = None

    def _on_training_start(self):
        self.start_time = time.time()
        self.last_save = self.start_time
        print(f"[Exp {self.exp_id}] start {self.algo} training...")

    def _on_step(self):
        if self.n_calls % self.log_interval == 0:
            steps = self.model.num_timesteps
            total_steps = self.model._total_timesteps
            pct = steps / total_steps * 100
            elapsed = time.time() - self.start_time
            remain = elapsed / (steps + 1e-9) * (total_steps - steps)
            print(f"[Exp {self.exp_id}] {self.algo} step {steps}/{total_steps}({pct:.1f}%), "
                  f"elapsed={elapsed / 60:.1f}m, ETA={remain / 60:.1f}m")
            if time.time() - self.last_save > 600:
                ckpt = f"./models/{self.algo}_exp{self.exp_id}_step{steps}.zip"
                self.model.save(ckpt)
                self.last_save = time.time()
                print(f"[Exp {self.exp_id}] checkpoint => {ckpt}")
        return True


def make_env(seed=0, env_kwargs=None):
    def _init():
        env = CommuneMatrixEnv(**(env_kwargs or {}))
        env.seed(seed)
        return env

    return _init


def make_vec_env(n_envs, seeds, env_kwargs=None):
    env_fns = [make_env(s, env_kwargs) for s in seeds]
    if n_envs == 1:
        return DummyVecEnv(env_fns)
    else:
        return SubprocVecEnv(env_fns)


def train_model(algo, exp_id, gpu_id, n_envs=4, total_timesteps=1000000, env_kwargs=None, seed=0):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    set_random_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    vec_env = make_vec_env(n_envs, [seed + i for i in range(n_envs)], env_kwargs)
    cb = CustomCallback(exp_id, algo, log_interval=2000)
    ckpt_cb = CheckpointCallback(save_freq=50000 // max(1, n_envs),
                                 save_path=f"./models/{algo}_exp{exp_id}_ckpts/",
                                 name_prefix="model")
    callbacks = [cb, ckpt_cb]

    if algo == "PPO":
        model = PPO("MlpPolicy", vec_env,
                    device="cuda",
                    verbose=0,
                    tensorboard_log=f"./tensorboard/{algo}_exp{exp_id}",
                    learning_rate=3e-4,
                    n_steps=2048,
                    batch_size=128,
                    n_epochs=10,
                    gamma=0.99,
                    gae_lambda=0.95,
                    clip_range=0.2,
                    ent_coef=0.01,
                    policy_kwargs=dict(net_arch=[dict(pi=[256, 256, 128], vf=[256, 256, 128])])
                    )
    elif algo == "A2C":
        model = A2C("MlpPolicy", vec_env,
                    device="cuda",
                    verbose=0,
                    tensorboard_log=f"./tensorboard/{algo}_exp{exp_id}",
                    learning_rate=7e-4,
                    n_steps=5,
                    gamma=0.99,
                    ent_coef=0.01,
                    vf_coef=0.5,
                    max_grad_norm=0.5,
                    policy_kwargs=dict(net_arch=[dict(pi=[256, 256, 128], vf=[256, 256, 128])])
                    )
    else:
        raise ValueError("Unsupported algo")

    model.learn(total_timesteps=total_timesteps, callback=callbacks, progress_bar=True)
    final_ckpt = f"./models/{algo}_exp{exp_id}_final.zip"
    model.save(final_ckpt)
    print(f"[Exp {exp_id}] training done => {final_ckpt}")
    vec_env.close()
    return model, final_ckpt


def evaluate_model(model_path, algo, exp_id, n_episodes=3, days=20, is_states=False):
    """
    评估RL策略: 记录每小时的 infections/deaths/commute, 并输出 JSON + XLSX + 时序图.
    """
    if algo == "PPO":
        from stable_baselines3 import PPO
        model = PPO.load(model_path)
    elif algo == "A2C":
        from stable_baselines3 import A2C
        model = A2C.load(model_path)
    else:
        raise ValueError("Unsupported algo")

    results = {
        'method': f"{algo}_exp{exp_id}",
        'episodes': []
    }
    for ep in range(n_episodes):
        env = CommuneMatrixEnv(days=days, is_states=is_states)
        obs = env.reset()
        done = False
        ep_data = {
            'hourly_infections': [],
            'hourly_deaths': [],
            'hourly_commute_ratio': [],
            'hourly_reward': [],
        }
        ep_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_reward += reward

            hour_infections = info['infections']
            hour_deaths = info['deaths']
            mat = env.seir_env.matrix
            orig_c = np.sum(env.original_matrix) - np.trace(env.original_matrix)
            now_c = np.sum(mat) - np.trace(mat)
            commute_ratio = now_c / (orig_c + 1e-12)

            ep_data['hourly_infections'].append(hour_infections)
            ep_data['hourly_deaths'].append(hour_deaths)
            ep_data['hourly_commute_ratio'].append(commute_ratio)
            ep_data['hourly_reward'].append(reward)

        ep_data['total_reward'] = ep_reward
        results['episodes'].append(ep_data)

    # 保存 JSON + XLSX
    out_dir = f"./results/{algo}_exp{exp_id}/"
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "evaluation.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    xlsx_path = os.path.join(out_dir, "evaluation.xlsx")
    with pd.ExcelWriter(xlsx_path) as writer:
        for idx, ep_data in enumerate(results['episodes'], start=1):
            df = pd.DataFrame({
                "hourly_infections": ep_data['hourly_infections'],
                "hourly_deaths": ep_data['hourly_deaths'],
                "hourly_commute_ratio": ep_data['hourly_commute_ratio'],
                "hourly_reward": ep_data['hourly_reward']
            })
            df.to_excel(writer, sheet_name=f"episode_{idx}", index=False)

    # 画图: 仅展示最后一条episode时序
    last_ep = results['episodes'][-1]
    length = len(last_ep['hourly_infections'])
    x_idx = np.arange(length)

    plt.figure(figsize=(12, 8))
    plt.subplot(4, 1, 1)
    plt.plot(x_idx, last_ep['hourly_infections'], 'b-', label='hourly_infections')
    plt.legend();
    plt.grid(alpha=0.3)

    plt.subplot(4, 1, 2)
    plt.plot(x_idx, last_ep['hourly_deaths'], 'r-', label='hourly_deaths')
    plt.legend();
    plt.grid(alpha=0.3)

    plt.subplot(4, 1, 3)
    plt.plot(x_idx, last_ep['hourly_commute_ratio'], 'g-', label='hourly_commute_ratio')
    plt.legend();
    plt.grid(alpha=0.3)

    plt.subplot(4, 1, 4)
    plt.plot(x_idx, last_ep['hourly_reward'], 'k-', label='hourly_reward')
    plt.legend();
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "evaluation_timeseries.png"))
    plt.close()

    print(f"[Exp {exp_id}] RL evaluate done => {json_path}")
    return results


def evaluate_baseline(baseline_name, days=20, n_episodes=1, is_states=False):
    """
    评估指定的 baseline 策略, 记录小时级数据 => JSON+XLSX+时序图
    """
    out_dir = f"./results/baseline_{baseline_name}/"
    os.makedirs(out_dir, exist_ok=True)
    results = {
        'method': f"baseline_{baseline_name}",
        'episodes': []
    }
    for ep in range(n_episodes):
        env = CommuneMatrixEnv(days=days, is_states=is_states)
        obs = env.reset()
        done = False
        ep_data = {
            'hourly_infections': [],
            'hourly_deaths': [],
            'hourly_commute_ratio': [],
            'hourly_reward': []
        }
        ep_reward = 0
        base_func = BASELINE_STRATEGIES[baseline_name]
        fix_mat = base_func(env.seir_env)

        while not done:
            action = fix_mat.flatten()
            obs, rew, done, info = env.step(action)
            ep_reward += rew

            hour_infections = info['infections']
            hour_deaths = info['deaths']
            mat = env.seir_env.matrix
            orig_c = np.sum(env.original_matrix) - np.trace(env.original_matrix)
            now_c = np.sum(mat) - np.trace(mat)
            commute_ratio = now_c / (orig_c + 1e-12)

            ep_data['hourly_infections'].append(hour_infections)
            ep_data['hourly_deaths'].append(hour_deaths)
            ep_data['hourly_commute_ratio'].append(commute_ratio)
            ep_data['hourly_reward'].append(rew)

        ep_data['total_reward'] = ep_reward
        results['episodes'].append(ep_data)

    # 保存 JSON+XLSX
    json_path = os.path.join(out_dir, "baseline_evaluation.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    xlsx_path = os.path.join(out_dir, "baseline_evaluation.xlsx")
    with pd.ExcelWriter(xlsx_path) as writer:
        for idx, ep_data in enumerate(results['episodes'], start=1):
            df = pd.DataFrame({
                "hourly_infections": ep_data['hourly_infections'],
                "hourly_deaths": ep_data['hourly_deaths'],
                "hourly_commute_ratio": ep_data['hourly_commute_ratio'],
                "hourly_reward": ep_data['hourly_reward']
            })
            df.to_excel(writer, sheet_name=f"episode_{idx}", index=False)

    # 画图
    last_ep = results['episodes'][-1]
    length = len(last_ep['hourly_infections'])
    x_idx = np.arange(length)
    plt.figure(figsize=(12, 8))
    plt.subplot(4, 1, 1)
    plt.plot(x_idx, last_ep['hourly_infections'], 'b-', label='hourly_infections')
    plt.legend();
    plt.grid(alpha=0.3)

    plt.subplot(4, 1, 2)
    plt.plot(x_idx, last_ep['hourly_deaths'], 'r-', label='hourly_deaths')
    plt.legend();
    plt.grid(alpha=0.3)

    plt.subplot(4, 1, 3)
    plt.plot(x_idx, last_ep['hourly_commute_ratio'], 'g-', label='hourly_commute_ratio')
    plt.legend();
    plt.grid(alpha=0.3)

    plt.subplot(4, 1, 4)
    plt.plot(x_idx, last_ep['hourly_reward'], 'k-', label='hourly_reward')
    plt.legend();
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "baseline_timeseries.png"))
    plt.close()
    print(f"Baseline {baseline_name} evaluate done => {json_path}")


# ================== 并行训练 & 并行评估 ==================
def parallel_train_models(configs, gpu_list):
    procs = []
    for cfg in configs:
        exp_id = cfg['exp_id']
        algo = cfg['algo']
        gpu = gpu_list[exp_id % len(gpu_list)]
        n_envs = cfg.get('n_envs', 4)
        ts = cfg.get('total_timesteps', 10000000)
        days = cfg.get('days', 20)
        is_st = cfg.get('is_states', False)
        seed = cfg.get('seed', 0)
        p = multiprocessing.Process(
            target=train_model,
            args=(algo, exp_id, gpu, n_envs, ts, {'days': days, 'is_states': is_st}, seed)
        )
        p.start()
        procs.append(p)
        time.sleep(2)
    for pp in procs:
        pp.join()
    print("All training processes done")


def parallel_evaluate_models(configs):
    with Pool(processes=min(multiprocessing.cpu_count(), len(configs))) as pool:
        tasks = []
        for cfg in configs:
            exp_id = cfg['exp_id']
            algo = cfg['algo']
            days = cfg.get('days', 20)
            is_st = cfg.get('is_states', False)
            modelp = f"./models/{algo}_exp{exp_id}_final.zip"
            if not os.path.exists(modelp):
                print(f"Model not found: {modelp}")
                continue
            tasks.append((modelp, algo, exp_id, 3, days, is_st))
        pool.starmap(evaluate_model, tasks)


def parallel_evaluate_baselines():
    # 简化串行
    for bname in BASELINE_STRATEGIES.keys():
        evaluate_baseline(bname, days=20, n_episodes=1, is_states=False)


# ================ 主函数 + 可视化综合调用 ================
def main():
    parser = argparse.ArgumentParser(description="SEIJRD RL with advanced visualization & baseline evaluation")
    parser.add_argument("--mode", type=str, choices=["train", "evaluate", "baseline", "visualize", "all"],
                        default="all",
                        help="Mode to run (train, evaluate, baseline, visualize, or all)")
    parser.add_argument("--algos", nargs="+", default=["PPO", "A2C"])
    parser.add_argument("--exp-start", type=int, default=1)
    parser.add_argument("--exp-count", type=int, default=8)
    parser.add_argument("--timesteps", type=int, default=10000000)
    parser.add_argument("--days", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--is-states", action="store_true",
                        help="Whether to load initial states from CSV (SEIJRD.csv)")
    args = parser.parse_args()

    configs = []
    for algo in args.algos:
        for eid in range(args.exp_start, args.exp_start + args.exp_count):
            configs.append({
                'exp_id': eid,
                'algo': algo,
                'total_timesteps': args.timesteps,
                'days': args.days,
                'is_states': args.is_states,
                'seed': args.seed + eid,
                'n_envs': 8
            })

    if args.mode == "all":
        # Run all modes sequentially
        print("Running complete pipeline: train → evaluate → baseline → visualize")

        # Training
        print("\n===== STEP 1: TRAINING MODELS =====")
        gpu_list = list(range(NUM_GPUS))
        parallel_train_models(configs, gpu_list)

        # Evaluation
        print("\n===== STEP 2: EVALUATING MODELS =====")
        parallel_evaluate_models(configs)

        # Baseline
        print("\n===== STEP 3: EVALUATING BASELINES =====")
        parallel_evaluate_baselines()

        # Visualization
        print("\n===== STEP 4: GENERATING VISUALIZATIONS =====")
        run_visualization(args)

        print("\nComplete pipeline finished! All results saved to ./results/ and ./visualizations/")

    elif args.mode == "train":
        gpu_list = list(range(NUM_GPUS))
        parallel_train_models(configs, gpu_list)

    elif args.mode == "evaluate":
        parallel_evaluate_models(configs)

    elif args.mode == "baseline":
        parallel_evaluate_baselines()

    elif args.mode == "visualize":
        run_visualization(args)

    else:
        parser.print_help()


def run_visualization(args):
    """Extracted visualization logic to a separate function for reuse"""
    print("Generating comparative visualizations...")

    # 读取RL结果
    rl_results = []
    for algo in args.algos:
        for eid in range(args.exp_start, args.exp_start + args.exp_count):
            result_path = f"./results/{algo}_exp{eid}/evaluation.json"
            if os.path.exists(result_path):
                with open(result_path, "r") as f:
                    data = json.load(f)
                    # 取最后一次episode
                    if data['episodes']:
                        last_ep = data['episodes'][-1]
                        # 额外构建一些衍生指标, 例如 cumulative_infections, ...
                        # 这里为了匹配 create_strategy_comparison 需要:
                        #   hourly_cumulative_infections / hourly_cumulative_deaths / ...
                        #   average_commute / ...
                        # 简单处理: 累加
                        c_infs = np.cumsum(last_ep['hourly_infections'])
                        c_deaths = np.cumsum(last_ep['hourly_deaths'])
                        avg_commute = []
                        cum_rew = []
                        sum_rew = 0
                        for r in last_ep['hourly_reward']:
                            sum_rew += r
                            cum_rew.append(sum_rew)
                        # 计算平均commute
                        tmp_sum = 0
                        tmp_arr = []
                        for c in last_ep['hourly_commute_ratio']:
                            tmp_sum += c
                            tmp_arr.append(tmp_sum / (len(tmp_arr) + 1))

                        new_data = {
                            'hourly_cumulative_infections': c_infs,
                            'hourly_cumulative_deaths': c_deaths,
                            'hourly_average_commute': tmp_arr,
                            'hourly_cumulative_reward': cum_rew,
                            'hourly_infections': last_ep['hourly_infections'],
                            'hourly_deaths': last_ep['hourly_deaths'],
                            'hourly_commute_ratio': last_ep['hourly_commute_ratio'],
                            'hourly_reward': last_ep['hourly_reward'],
                        }
                        rl_results.append((algo, eid, new_data))

    # 读取 baseline 结果
    baseline_results = []
    for bname in BASELINE_STRATEGIES.keys():
        pathb = f"./results/baseline_{bname}/baseline_evaluation.json"
        if os.path.exists(pathb):
            with open(pathb, 'r') as f:
                data = json.load(f)
                if data['episodes']:
                    last_ep = data['episodes'][-1]
                    c_infs = np.cumsum(last_ep['hourly_infections'])
                    c_deaths = np.cumsum(last_ep['hourly_deaths'])
                    avg_commute = []
                    cum_rew = []
                    sum_r = 0
                    for rr in last_ep['hourly_reward']:
                        sum_r += rr
                        cum_rew.append(sum_r)
                    tmp_sum = 0
                    tmp_arr = []
                    for c in last_ep['hourly_commute_ratio']:
                        tmp_sum += c
                        tmp_arr.append(tmp_sum / (len(tmp_arr) + 1))

                    newb = {
                        'hourly_cumulative_infections': c_infs,
                        'hourly_cumulative_deaths': c_deaths,
                        'hourly_average_commute': tmp_arr,
                        'hourly_cumulative_reward': cum_rew,
                        'hourly_infections': last_ep['hourly_infections'],
                        'hourly_deaths': last_ep['hourly_deaths'],
                        'hourly_commute_ratio': last_ep['hourly_commute_ratio'],
                        'hourly_reward': last_ep['hourly_reward'],
                    }
                    baseline_results.append((bname, newb))

    # 计算 max_infections, max_deaths 用于雷达图
    all_infs = []
    all_deaths = []
    for a, e, res in rl_results:
        all_infs.append(np.sum(res['hourly_infections']))
        all_deaths.append(np.sum(res['hourly_deaths']))
    for b, res in baseline_results:
        all_infs.append(np.sum(res['hourly_infections']))
        all_deaths.append(np.sum(res['hourly_deaths']))

    if not all_infs or not all_deaths:
        print("No results found to visualize.")
        return

    max_infs = max(all_infs)
    max_deaths = max(all_deaths)

    # 生成可视化
    if rl_results and baseline_results:
        create_strategy_comparison(rl_results, baseline_results)
        create_final_state_comparison(rl_results, baseline_results)
        create_algorithm_radar_chart(rl_results, baseline_results, max_infs, max_deaths)

        # matrix_evolution -> 任意一个 RL
        if rl_results:
            algo, eid, _ = rl_results[0]
            modp = f"./models/{algo}_exp{eid}_final.zip"
            if os.path.exists(modp):
                visualize_matrix_evolution(modp, algo, eid)

        # district_infections -> 用第一个 RL + 第一个 baseline
        if rl_results and baseline_results:
            from stable_baselines3 import PPO, A2C
            algo, eid, _ = rl_results[0]
            modp = f"./models/{algo}_exp{eid}_final.zip"
            bname, bres = baseline_results[0]
            if os.path.exists(modp):
                if algo == "PPO":
                    rl_m = PPO.load(modp)
                else:
                    rl_m = A2C.load(modp)
                baseline_fn = BASELINE_STRATEGIES[bname]
                visualize_district_infections(rl_m, baseline_fn)

    print("Visualization complete. Results saved to ./visualizations/")


if __name__ == "__main__":
    main()
