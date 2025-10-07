#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
main_RL_task2_910.py

- Task2: 使用“717版”的双β动力学(beta_1 作用 S×E，beta_2 作用 S×(I+J))
- 与 Task3 对齐的评估与可视化落盘格式：逐时指标、区县维度、策略矩阵序列
- 支持 baseline 评估；支持“一键全流程”（train → evaluate → baseline → visualize）

结果目录规范（advanced_visuals 会按此读取）：
  ./results/{ALGO}_exp{ID}/evaluation.json                # RL
  ./results/baseline_{NAME}/baseline_evaluation.json      # baseline
"""

import os
import argparse
import json
import time
import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
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

# ---------- 目录 ----------
os.makedirs("./models", exist_ok=True)
os.makedirs("./logs", exist_ok=True)
os.makedirs("./results", exist_ok=True)
os.makedirs("./visualizations", exist_ok=True)
os.makedirs("./tensorboard", exist_ok=True)
os.makedirs("./configs", exist_ok=True)
os.makedirs("./interactive", exist_ok=True)

# ---------- 全局 ----------
NUM_GPUS = 8
MAX_POP = 1e12
FILE_PATH = "data/end_copy.json"

# ---------- “双β”动力学参数（717思路） ----------
backup_paras = {
    "beta_1": 0.0614,   # S×E
    "beta_2": 0.0696,   # S×(I+J)
    "gamma_i": 0.0496,
    "gamma_j": 0.0376,
    "sigma_i": 0.01378,
    "sigma_j": 0.03953,
    "mu_i": 2e-5,
    "mu_j": 0.00027,
    "rho": 8.62e-05,     # 场馆内部接触感染基数
    "initial_infect": 500
}

MIN_SELF_ACTIVITY = 0.3  # 通勤矩阵对角线下限（保留本区活动）
DEFAULT_IS_STATES = False


# ================== 数据读取 ==================
def Input_Population(fp="data/tokyo_population.csv"):
    df = pd.read_csv(fp, header=None, names=["code", "ward_jp", "ward_en", "dummy", "population"])
    df["population"] = df["population"].astype(int)
    return df["population"].tolist()

def Input_Matrix(fp="data/tokyo_commuting_flows_with_intra.csv"):
    df = pd.read_csv(fp, index_col=0).apply(pd.to_numeric, errors='coerce')
    return df.values

def Input_Initial(fp="data/SEIJRD.csv"):
    df = pd.read_csv(fp, index_col=None).astype(float)
    return df.values

def process_competitions(data_file):
    slot_mapping = {0:(8,11), 1:(13,17), 2:(19,22)}
    year = 2021
    earliest = None
    for venue in data_file:
        for t in venue["time"]:
            d = datetime.date(year, t["month"], t["day"])
            earliest = d if (earliest is None or d < earliest) else earliest

    comps = []
    for venue in data_file:
        vid = venue["venue_id"]
        cap = int(venue["capacity"].replace(',', ''))
        for t in venue["time"]:
            d = datetime.date(year, t["month"], t["day"])
            diff = (d - earliest).days
            sh, eh = slot_mapping[t["slot"]]
            st, ed = diff*24+sh, diff*24+eh
            comps.append((vid, st, ed, cap, t["slot"]))
    return comps

def process_places(data_file):
    places = {}
    for d in data_file:
        vid = d["venue_id"]
        places[vid] = Place({
            "venue_id": vid,
            "venue_name": d["venue_name"],
            "capacity": int(d["capacity"].replace(',', ''))
        })
    return places


# ================== 环境内部对象 ==================
class Place:
    def __init__(self, data: Dict):
        self.id = data["venue_id"]
        self.name = data["venue_name"]
        self.capacity = data["capacity"]
        self.agenda = []
        self.endtime = None
        self.audience_from = []

    def infect(self, paras):
        """场馆内：E+I+J 共同决定感染概率；按各区 S 占比分配新增 E。"""
        if len(self.audience_from) == 0:
            return
        arr = np.sum(self.audience_from, axis=0)
        infectious = arr[1] + arr[2] + arr[3]
        if infectious <= 0:
            return
        prob = 1 - (1 - paras["rho"])**infectious
        prob = float(np.clip(prob, 0, 1))
        S_tot = arr[0]
        if S_tot < 1e-6:
            return
        new_E = S_tot * prob
        frac_S = self.audience_from[:, 0] / (S_tot + 1e-12)
        inc = frac_S * new_E
        self.audience_from[:, 0] -= inc
        self.audience_from[:, 1] += inc


class SEIR_Node:
    """双β动力学节点：beta_1*S*E/N + beta_2*S*(I+J)/N"""
    def __init__(self, state, total_nodes):
        self.total_nodes = total_nodes
        self.state = state.astype(np.float64)
        self.from_node = np.zeros((total_nodes, 6))
        self.to_node = np.zeros((total_nodes, 5))

    def update_seir(self, delta_time=1/2400, times=100, paras=backup_paras):
        for _ in range(times):
            S, E, I, J, R, D = self.state
            N = S + E + I + J + R
            if N <= 1e-12:
                continue

            b1, b2 = paras["beta_1"], paras["beta_2"]
            si, sj = paras["sigma_i"], paras["sigma_j"]
            gi, gj = paras["gamma_i"], paras["gamma_j"]
            mui, muj = paras["mu_i"], paras["mu_j"]

            inf_SE = b1 * S * E / N
            inf_SIJ = b2 * S * (I + J) / N

            dS = -(inf_SE + inf_SIJ)
            dE = (inf_SE + inf_SIJ) - si * E - sj * E
            dI = si * E - gi * I - mui * I
            dJ = sj * E - gj * J - muj * J
            dR = gi * I + gj * J
            dD = mui * I + muj * J

            self.state += np.array([dS, dE, dI, dJ, dR, dD]) * delta_time

        self.state = np.clip(self.state, 0, MAX_POP)


class SEIR_Network:
    def __init__(self, num, states, matrix=None, paras=backup_paras):
        self.node_num = num
        self.nodes = {i: SEIR_Node(states[i], num) for i in range(num)}
        if matrix is not None:
            rs = matrix.sum(axis=1, keepdims=True)
            self.A = matrix / (rs + 1e-12)
        else:
            A = np.random.rand(num, num)
            self.A = A / A.sum(axis=1, keepdims=True)
        self.paras = paras
        self.delta_time = 1/2400

    def morning_commuting(self):
        # 发出
        for i in range(self.node_num):
            SEIJR = self.nodes[i].state[:5]
            for j in range(self.node_num):
                self.nodes[i].to_node[j] = SEIJR * self.A[i][j]
        # 清空本地（离开）
        for i in range(self.node_num):
            self.nodes[i].state[:5] = 0.0
        # 抵达并记录来源
        for i in range(self.node_num):
            for j in range(self.node_num):
                self.nodes[i].state[:5] += self.nodes[j].to_node[i]
                self.nodes[i].from_node[j, :5] = self.nodes[j].to_node[i]

    def evening_commuting(self):
        # 回流
        for i in range(self.node_num):
            for j in range(self.node_num):
                if i == j:
                    continue
                self.nodes[j].state -= self.nodes[j].from_node[i]
                self.nodes[i].state += self.nodes[j].from_node[i]
        # 清空缓存
        for i in range(self.node_num):
            self.nodes[i].from_node = np.zeros((self.node_num, 6))
            self.nodes[i].to_node = np.zeros((self.node_num, 5))

    def update_network(self):
        for i in range(self.node_num):
            self.nodes[i].update_seir(self.delta_time, 100, self.paras)


class Env:
    """底层 SEIJRD 环境（含场馆事件与通勤矩阵、双β联动）"""
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
        data_json = json.load(open(FILE_PATH, "r", encoding="utf-8"))
        self.competitions = process_competitions(data_json)
        self.places = process_places(data_json)
        for c in self.competitions:
            self.places[c[0]].agenda.append(c)
        for pl in self.places.values():
            pl.agenda.sort(key=lambda x: x[1])

        # 初始 SEIJRD
        if self.is_states:
            init_states = Input_Initial("data/SEIJRD.csv")
        else:
            pops = Input_Population("data/tokyo_population.csv")
            init_states = np.zeros((self.num, 6))
            for i in range(self.num):
                S_ = pops[i] - self.paras["initial_infect"]*4.5/self.num
                E_ = self.paras["initial_infect"]*2/self.num
                I_ = self.paras["initial_infect"]*1/self.num
                J_ = self.paras["initial_infect"]*1.5/self.num
                init_states[i] = np.array([S_, E_, I_, J_, 0, 0], dtype=float)

        # 通勤矩阵
        try:
            mat = Input_Matrix("data/tokyo_commuting_flows_with_intra.csv")
            rs = mat.sum(axis=1, keepdims=True)
            mat = mat / (rs + 1e-12)
            self.origin_matrix = mat.copy()
            self.matrix = mat.copy()
        except:
            self.origin_matrix = np.eye(self.num)
            self.matrix = np.eye(self.num)

        self.network = SEIR_Network(self.num, init_states, self.matrix, self.paras)
        self.current_tick = 0

    def check_competition_start(self):
        for pid, pl in self.places.items():
            if not pl.agenda:
                continue
            comp = pl.agenda[0]
            if comp[1] == self.current_tick:
                pl.audience_from = np.zeros((self.num, 6))
                pl.agenda.pop(0)
                pl.endtime = comp[2]
                capacity = comp[3]
                # 按当前各区 SEIJR 规模分配入场
                N = np.array([np.sum(self.network.nodes[i].state[:5]) for i in range(self.num)])
                tot = np.sum(N) if np.sum(N) > 0 else 1.0
                for i in range(self.num):
                    portion = self.network.nodes[i].state[:5] * (capacity / tot)
                    portion = np.minimum(portion, self.network.nodes[i].state[:5])
                    pl.audience_from[i, :5] = portion
                    self.network.nodes[i].state[:5] -= portion
                # 场馆内感染
                pl.infect(self.paras)

    def check_competition_end(self):
        for pid, pl in self.places.items():
            if pl.endtime == self.current_tick:
                for i in range(self.num):
                    self.network.nodes[i].state[:5] += pl.audience_from[i, :5]
                pl.audience_from = []
                pl.endtime = None

    def modify_matrix(self, matrix):
        """应用 RL 动作生成的新矩阵，并联动调节 beta_1/beta_2。"""
        rs = matrix.sum(axis=1, keepdims=True)
        matrix_ = matrix / (rs + 1e-12)
        self.network.A = matrix_.copy()
        self.matrix = matrix_.copy()

        # 与原始矩阵的通勤强度比 -> 调整 β
        orig = np.sum(self.origin_matrix) - np.trace(self.origin_matrix)
        curr = np.sum(self.matrix) - np.trace(self.matrix)
        ratio = curr / (orig + 1e-12)

        # 给定下限，线性插值
        b1_base, b2_base = backup_paras["beta_1"], backup_paras["beta_2"]
        new_b1 = max(0.01, 0.02 + (b1_base - 0.02) * ratio)
        new_b2 = max(0.01, 0.03 + (b2_base - 0.03) * ratio)
        self.network.paras["beta_1"] = new_b1
        self.network.paras["beta_2"] = new_b2


# ================== RL 环境（每小时一步） ==================
class CommuneMatrixEnv(gym.Env):
    def __init__(self, days=20, is_states=False):
        super().__init__()
        self.days = days
        self.is_states = is_states
        self.seir_env = Env(is_states=self.is_states)
        self.seir_env.init_env()
        self.num = self.seir_env.num

        obs_dim = self.num * 6 + 2
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=0, high=1, shape=(self.num * self.num,), dtype=np.float32)

        self.current_hour = 0
        self.current_day = 0
        self.episode_steps = 0
        self.max_steps = 24 * self.days

        self.original_matrix = self.seir_env.origin_matrix.copy()
        self.np_random = None

    # 兼容 gym 接口
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

    def _get_obs(self):
        arr = np.concatenate([self.seir_env.network.nodes[i].state for i in range(self.num)], axis=0)
        return np.concatenate([arr, [self.current_day, self.current_hour]]).astype(np.float32)

    def _process_action(self, action):
        mat = action.reshape(self.num, self.num)
        mat = np.maximum(mat, 0.0)
        rs = mat.sum(axis=1, keepdims=True)
        mat = mat / (rs + 1e-12)
        # 对角线最小活动度
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

        # 多目标线性权重（与 Task3 一致思想）
        w_inf, w_d, w_c = -1.0, -3.0, 0.5
        reward = w_inf * new_inf + w_d * new_d + w_c * total_pop * ratio
        # 归一尺度，避免数值过大
        return reward / 1e6

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

        # 时间推进
        self.current_hour += 1
        if self.current_hour >= 24:
            self.current_hour = 0
            self.current_day += 1
        self.episode_steps += 1
        done = (self.episode_steps >= self.max_steps)

        # info
        prev_in = np.sum(prev_states[:, 1:4])
        curr_in = np.sum(curr_states[:, 1:4])
        prev_d = np.sum(prev_states[:, 5])
        curr_d = np.sum(curr_states[:, 5])
        info = {
            "infections": float(max(0, curr_in - prev_in)),
            "deaths": float(max(0, curr_d - prev_d)),
            "matrix": matrix
        }

        return self._get_obs(), reward, done, info


# ================== Baselines（与 Task3 对齐） ==================
BASELINE_STRATEGIES = {}

def baseline_original(env):
    return env.origin_matrix.copy()

def baseline_diagonal(env):
    m = np.zeros_like(env.origin_matrix)
    np.fill_diagonal(m, 1.0)
    return m

def baseline_neighbor(env):
    m = (env.origin_matrix > 1e-12).astype(float)
    rs = m.sum(axis=1, keepdims=True)
    m = m / (rs + 1e-12)
    for i in range(env.num):
        if m[i, i] < MIN_SELF_ACTIVITY:
            deficit = MIN_SELF_ACTIVITY - m[i, i]
            non_diag = 1.0 - m[i, i]
            if non_diag > 0:
                scale = (non_diag - deficit) / non_diag
                for j in range(env.num):
                    if j != i:
                        m[i, j] *= scale
            m[i, i] = MIN_SELF_ACTIVITY
    return m

def baseline_random50(env):
    n = env.num
    m = np.random.rand(n, n)
    rs = m.sum(axis=1, keepdims=True)
    m = m / (rs + 1e-12)
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


# ================== 训练 / 评估 ==================
class ProgressCB(BaseCallback):
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
        print(f"[Exp {self.exp_id}] {self.algo} training start.")

    def _on_step(self):
        if self.n_calls % self.log_interval == 0:
            steps = self.model.num_timesteps
            total_steps = self.model._total_timesteps
            pct = steps / max(1, total_steps) * 100
            elapsed = time.time() - self.start_time
            remain = elapsed / max(1, steps) * max(0, total_steps - steps)
            print(f"[Exp {self.exp_id}] {self.algo} {steps}/{total_steps} ({pct:.1f}%), "
                  f"elapsed={elapsed/60:.1f}m, ETA={remain/60:.1f}m")
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

def make_vec_env(n_envs, seeds=None, env_kwargs=None):
    if seeds is None:
        seeds = list(range(n_envs))
    fns = [make_env(seed=seeds[i], env_kwargs=env_kwargs) for i in range(n_envs)]
    return DummyVecEnv(fns) if n_envs == 1 else SubprocVecEnv(fns)


def train_model(algo, exp_id, gpu_id, n_envs=4, total_timesteps=1_000_000, env_kwargs=None, seed=0):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    set_random_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    vec_env = make_vec_env(n_envs=n_envs, seeds=[seed + i for i in range(n_envs)], env_kwargs=env_kwargs)
    cb = ProgressCB(exp_id=exp_id, algo=algo, log_interval=2000)
    ckpt_cb = CheckpointCallback(save_freq=max(1, 50_000 // n_envs),
                                 save_path=f"./models/{algo}_exp{exp_id}_ckpts/",
                                 name_prefix="model")

    if algo == "PPO":
        model = PPO("MlpPolicy", vec_env, device="cuda", verbose=0,
                    tensorboard_log=f"./tensorboard/{algo}_exp{exp_id}",
                    learning_rate=3e-4, n_steps=2048, batch_size=128, n_epochs=10,
                    gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01)
    elif algo == "A2C":
        model = A2C("MlpPolicy", vec_env, device="cuda", verbose=0,
                    tensorboard_log=f"./tensorboard/{algo}_exp{exp_id}",
                    learning_rate=7e-4, n_steps=5, gamma=0.99,
                    ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5)
    else:
        raise ValueError("Unsupported algo")

    model.learn(total_timesteps=total_timesteps, callback=[cb, ckpt_cb], progress_bar=True)
    final_p = f"./models/{algo}_exp{exp_id}_final.zip"
    model.save(final_p)
    vec_env.close()
    print(f"[Exp {exp_id}] training done => {final_p}")
    return final_p


def evaluate_model(model_path, algo, exp_id, n_episodes=3, days=20):
    """与 Task3 对齐的落盘结构：逐时、区县、策略矩阵（日粒度）"""
    model = PPO.load(model_path) if algo == "PPO" else A2C.load(model_path)
    results = {"method": f"{algo}_exp{exp_id}", "episodes": []}

    for ep in range(n_episodes):
        env = CommuneMatrixEnv(days=days)
        obs = env.reset()
        done = False
        ep_data = {
            "hourly_infections": [], "hourly_deaths": [], "hourly_commute_ratio": [],
            "hourly_reward": [], "district_infections": [], "policy_matrices": []
        }
        dist_infs = [[] for _ in range(env.num)]
        step = 0
        total_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward

            # 逐时
            mat = env.seir_env.matrix
            ori_c = np.sum(env.original_matrix) - np.trace(env.original_matrix)
            now_c = np.sum(mat) - np.trace(mat)
            c_ratio = now_c / (ori_c + 1e-12)

            ep_data["hourly_reward"].append(float(reward))
            ep_data["hourly_infections"].append(float(info["infections"]))
            ep_data["hourly_deaths"].append(float(info["deaths"]))
            ep_data["hourly_commute_ratio"].append(float(c_ratio))

            # 区县感染堆栈（E+I+J）
            for d in range(env.num):
                eij = env.seir_env.network.nodes[d].state[1:4].sum()
                dist_infs[d].append(float(eij))

            # 每天记录一次策略矩阵
            if step % 24 == 0:
                ep_data["policy_matrices"].append(mat.copy().tolist())

            step += 1

        ep_data["total_reward"] = float(total_reward)
        ep_data["district_infections"] = dist_infs
        results["episodes"].append(ep_data)

    out_dir = f"./results/{algo}_exp{exp_id}/"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "evaluation.json"), "w") as f:
        json.dump(results, f, indent=2)

    # 简单时序图（最后一条 episode）
    last = results["episodes"][-1]
    X = np.arange(len(last["hourly_infections"]))
    plt.figure(figsize=(12, 8))
    plt.subplot(4,1,1); plt.plot(X, last["hourly_infections"]); plt.title("hourly_infections"); plt.grid(alpha=.3)
    plt.subplot(4,1,2); plt.plot(X, last["hourly_deaths"]); plt.title("hourly_deaths"); plt.grid(alpha=.3)
    plt.subplot(4,1,3); plt.plot(X, last["hourly_commute_ratio"]); plt.title("hourly_commute_ratio"); plt.grid(alpha=.3)
    plt.subplot(4,1,4); plt.plot(X, last["hourly_reward"]); plt.title("hourly_reward"); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "evaluation_timeseries.png")); plt.close()

    print(f"[Exp {exp_id}] RL evaluate done => {out_dir}/evaluation.json")
    return results


def evaluate_baseline(baseline_name, days=20, n_episodes=1):
    """baseline 评估与落盘（与 Task3 对齐）"""
    out_dir = f"./results/baseline_{baseline_name}/"
    os.makedirs(out_dir, exist_ok=True)
    results = {"method": f"baseline_{baseline_name}", "episodes": []}

    for ep in range(n_episodes):
        env = CommuneMatrixEnv(days=days)
        obs = env.reset()
        done = False
        ep_data = {"hourly_infections": [], "hourly_deaths": [], "hourly_commute_ratio": [], "hourly_reward": []}
        total_reward = 0.0
        fix_mat = BASELINE_STRATEGIES[baseline_name](env.seir_env)

        while not done:
            action = fix_mat.flatten()
            obs, rew, done, info = env.step(action)
            total_reward += rew

            mat = env.seir_env.matrix
            ori_c = np.sum(env.original_matrix) - np.trace(env.original_matrix)
            now_c = np.sum(mat) - np.trace(mat)
            c_ratio = now_c / (ori_c + 1e-12)

            ep_data["hourly_infections"].append(float(info["infections"]))
            ep_data["hourly_deaths"].append(float(info["deaths"]))
            ep_data["hourly_commute_ratio"].append(float(c_ratio))
            ep_data["hourly_reward"].append(float(rew))

        ep_data["total_reward"] = float(total_reward)
        results["episodes"].append(ep_data)

    with open(os.path.join(out_dir, "baseline_evaluation.json"), "w") as f:
        json.dump(results, f, indent=2)

    # 简单时序
    last = results["episodes"][-1]; X = np.arange(len(last["hourly_infections"]))
    plt.figure(figsize=(12,8))
    plt.subplot(4,1,1); plt.plot(X,last["hourly_infections"]); plt.title("hourly_infections"); plt.grid(alpha=.3)
    plt.subplot(4,1,2); plt.plot(X,last["hourly_deaths"]); plt.title("hourly_deaths"); plt.grid(alpha=.3)
    plt.subplot(4,1,3); plt.plot(X,last["hourly_commute_ratio"]); plt.title("hourly_commute_ratio"); plt.grid(alpha=.3)
    plt.subplot(4,1,4); plt.plot(X,last["hourly_reward"]); plt.title("hourly_reward"); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "baseline_timeseries.png")); plt.close()

    print(f"Baseline {baseline_name} evaluate done => {out_dir}/baseline_evaluation.json")


# ================== 并行封装 ==================
def parallel_train_models(configs, gpu_list):
    procs = []
    for cfg in configs:
        exp_id, algo = cfg["exp_id"], cfg["algo"]
        gpu = gpu_list[exp_id % len(gpu_list)]
        p = multiprocessing.Process(
            target=train_model,
            args=(algo, exp_id, gpu, cfg.get("n_envs", 4), cfg.get("timesteps", 1_000_000),
                  {"days": cfg.get("days", 20)}, cfg.get("seed", exp_id))
        )
        p.start(); procs.append(p); time.sleep(2)
    for p in procs: p.join()
    print("All trainings done.")

def parallel_evaluate_models(configs):
    with Pool(processes=min(multiprocessing.cpu_count(), len(configs))) as pool:
        tasks = []
        for cfg in configs:
            exp_id, algo = cfg["exp_id"], cfg["algo"]
            model_path = f"./models/{algo}_exp{exp_id}_final.zip"
            if os.path.exists(model_path):
                tasks.append((model_path, algo, exp_id, 3, cfg.get("days", 20)))
            else:
                print(f"Model not found: {model_path}")
        pool.starmap(evaluate_model, tasks)

def evaluate_all_baselines():
    for b in BASELINE_STRATEGIES.keys():
        evaluate_baseline(b, days=20, n_episodes=1)


# ================== 可视化入口（统一用 advanced_visuals） ==================
def run_visualization():
    # 统一调用改良版 advanced_visuals（读取 ./results/**/evaluation.json）
    from advanced_visuals import run_all_visualization_improvements
    run_all_visualization_improvements("./results")


# ================== main ==================
def main():
    ap = argparse.ArgumentParser(description="Task2 - Dual-beta RL with full evaluation & visualization")
    ap.add_argument("--mode", choices=["train", "evaluate", "baseline", "visualize", "all"], default="all")
    ap.add_argument("--algos", nargs="+", default=["PPO", "A2C"])
    ap.add_argument("--exp-start", type=int, default=1)
    ap.add_argument("--exp-count", type=int, default=8)
    ap.add_argument("--timesteps", type=int, default=1_000_000)
    ap.add_argument("--days", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    exp_ids = list(range(args.exp_start, args.exp_start + args.exp_count))
    configs = [{"exp_id": eid, "algo": algo, "timesteps": args.timesteps, "days": args.days,
                "seed": args.seed + eid, "n_envs": 16}
               for algo in args.algos for eid in exp_ids]

    if args.mode in ("train", "all"):
        parallel_train_models(configs, list(range(NUM_GPUS)))

    if args.mode in ("evaluate", "all"):
        parallel_evaluate_models(configs)

    if args.mode in ("baseline", "all"):
        evaluate_all_baselines()

    if args.mode in ("visualize", "all"):
        run_visualization()

if __name__ == "__main__":
    main()
