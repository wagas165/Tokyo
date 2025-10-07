#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Task2 main - main_RL_task2_910.py

- 双β动力学(717版思路):
    dS = -β1*S*E/N - β2*S*(I+J)/N
    dE = +β1*S*E/N + β2*S*(I+J)/N - σi*E - σj*E
    dI = +σi*E - γi*I - μi*I
    dJ = +σj*E - γj*J - μj*J
    dR = +γi*I + γj*J
    dD = +μi*I + μj*J

- β2 与通勤非对角强度联动；β1保持基线（更贴合“近距离接触/家庭内传播”不随通勤强度线性变化）。
- 每 step=1小时；完整记录每小时 infection/death/commute/reward。
- 训练/评估/基线/可视化一键式执行。
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

from typing import Dict, List, Tuple
import multiprocessing
from multiprocessing import Pool

import torch
import gym
from gym import spaces
from gym.utils import seeding

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed

# ---------- dirs ----------
for d in ["./models", "./logs", "./results", "./visualizations", "./tensorboard", "./interactive"]:
    os.makedirs(d, exist_ok=True)

# ---------- hardware/global ----------
NUM_GPUS = int(os.environ.get("NUM_GPUS", 8))
MAX_POP = 1e12
MIN_SELF_ACTIVITY = 0.3
FILE_JSON = "data/end_copy.json"

# ---------- Double-β parameters (717风格&对齐范围) ----------
BASE_PARAS = {
    "beta_1": 0.0614,
    "beta_2": 0.0696,   # 会随通勤强度缩放
    "sigma_i": 0.01378,
    "sigma_j": 0.03953,
    "gamma_i": 0.0496,
    "gamma_j": 0.0376,
    "mu_i": 2.0e-5,
    "mu_j": 2.7e-4,
    "rho": 8.62e-5,     # 赛事传播基数
    "initial_infect": 500
}

# ---------- CSV readers ----------
def Input_Population(fp="data/tokyo_population.csv"):
    df = pd.read_csv(fp, header=None, names=["code","ward_jp","ward_en","dummy","population"])
    df["population"] = df["population"].astype(int)
    return df["population"].tolist()

def Input_Matrix(fp="data/tokyo_commuting_flows_with_intra.csv"):
    df = pd.read_csv(fp, index_col=0)
    df = df.apply(pd.to_numeric, errors="coerce")
    return df.values

def Input_Initial(fp="data/SEIJRD.csv"):
    df = pd.read_csv(fp, index_col=None).astype(float)
    return df.values

# ---------- competitions / places ----------
def process_competitions(data_file):
    slot_mapping = {0:(8,11), 1:(13,17), 2:(19,22)}
    year = 2021
    earliest = None
    for v in data_file:
        for t in v["time"]:
            d = datetime.date(year, t["month"], t["day"])
            if earliest is None or d < earliest:
                earliest = d
    comps=[]
    for v in data_file:
        vid = v["venue_id"]
        cap = int(v["capacity"].replace(',',''))
        for t in v["time"]:
            d = datetime.date(year, t["month"], t["day"])
            diff = (d - earliest).days
            sh, eh = slot_mapping[t["slot"]]
            comps.append((vid, diff*24+sh, diff*24+eh, cap, t["slot"]))
    return comps

class Place:
    def __init__(self, data: Dict):
        self.id = data["venue_id"]
        self.name = data["venue_name"]
        self.capacity = data["capacity"]
        self.agenda = []
        self.endtime = None
        self.audience_from = []

    def infect(self, paras: Dict):
        """赛事内传播：E+I+J 皆可传染；按易感者比例分配新增E。"""
        if len(self.audience_from) == 0:
            return
        arr = np.sum(self.audience_from, axis=0)  # [S,E,I,J,R,D]
        infectious = arr[1] + arr[2] + arr[3]
        if infectious <= 0:
            return
        prob = 1 - (1 - paras["rho"])**infectious
        prob = np.clip(prob, 0, 1)
        S_tot = arr[0]
        if S_tot < 1e-8:
            return
        new_inf = S_tot * prob
        frac = self.audience_from[:,0] / (S_tot + 1e-12)
        inc = frac * new_inf
        self.audience_from[:,0] -= inc
        self.audience_from[:,1] += inc

def process_places(data_file):
    places={}
    for d in data_file:
        info = {
            "venue_id": d["venue_id"],
            "venue_name": d["venue_name"],
            "capacity": int(d["capacity"].replace(',',''))
        }
        places[info["venue_id"]] = Place(info)
    return places

# ---------- SEIJRD with double-β ----------
class Node:
    def __init__(self, state, total_nodes):
        self.state = state.astype(np.float64)     # [S,E,I,J,R,D]
        self.total_nodes = total_nodes
        self.from_node = np.zeros((total_nodes, 6))
        self.to_node = np.zeros((total_nodes, 5))

    def update(self, delta_time=1/2400, times=100, paras: Dict=BASE_PARAS):
        S,E,I,J,R,D = self.state
        N = S+E+I+J+R
        if N > 0:
            dS = - paras["beta_1"]*S*E/N - paras["beta_2"]*S*(I+J)/N
            dE = + paras["beta_1"]*S*E/N + paras["beta_2"]*S*(I+J)/N - paras["sigma_i"]*E - paras["sigma_j"]*E
            dI = + paras["sigma_i"]*E - paras["gamma_i"]*I - paras["mu_i"]*I
            dJ = + paras["sigma_j"]*E - paras["gamma_j"]*J - paras["mu_j"]*J
            dR = + paras["gamma_i"]*I + paras["gamma_j"]*J
            dD = + paras["mu_i"]*I + paras["mu_j"]*J
        else:
            dS=dE=dI=dJ=dR=dD=0.0

        inc = np.array([dS,dE,dI,dJ,dR,dD])
        for _ in range(times):
            self.state += inc * delta_time
            # 把增量按“来源”份额回灌（细节延续旧实现）
            Nvalue = np.sum(self.from_node, axis=1)
            ratio = (Nvalue / (N + 1e-12))[:,None]
            self.from_node += inc[None,:] * ratio * delta_time

        self.state = np.clip(self.state, 0, MAX_POP)
        self.from_node = np.clip(self.from_node, 0, MAX_POP)

class Network:
    def __init__(self, num, states, matrix=None, paras: Dict=BASE_PARAS):
        self.node_num = num
        self.nodes = {i: Node(states[i], num) for i in range(num)}
        if matrix is not None:
            row = matrix.sum(axis=1, keepdims=True)
            self.A = matrix / (row + 1e-12)
        else:
            A = np.random.rand(num, num); A /= A.sum(axis=1, keepdims=True)
            self.A = A
        self.paras = paras
        self.delta_time = 1/2400

    def morning_commuting(self):
        for i in range(self.node_num):
            SEIJR = self.nodes[i].state[:5]
            for j in range(self.node_num):
                self.nodes[i].to_node[j] = SEIJR * self.A[i,j]
        for i in range(self.node_num):
            self.nodes[i].state[:5] = 0.0
        for i in range(self.node_num):
            for j in range(self.node_num):
                self.nodes[j].state[:5] += self.nodes[i].to_node[j]
                self.nodes[j].from_node[i,:5] = self.nodes[i].to_node[j]

    def evening_commuting(self):
        for i in range(self.node_num):
            for j in range(self.node_num):
                if i == j:
                    continue
                self.nodes[j].state -= self.nodes[j].from_node[i]
                self.nodes[i].state += self.nodes[j].from_node[i]
        for i in range(self.node_num):
            self.nodes[i].from_node = np.zeros((self.node_num,6))
            self.nodes[i].to_node   = np.zeros((self.node_num,5))

    def update_network(self):
        for i in range(self.node_num):
            self.nodes[i].update(self.delta_time, 100, self.paras)

# ---------- Env (加载数据、赛事、矩阵与β联动) ----------
class Env:
    def __init__(self, is_states=False):
        self.is_states = is_states
        self.num = 23
        self.paras = BASE_PARAS.copy()
        self.network: Network = None
        self.places = {}
        self.competitions = []
        self.matrix = None
        self.origin_matrix = None
        self.current_tick = 0

    def init_env(self):
        data_json = json.load(open(FILE_JSON, "r", encoding="utf-8"))
        self.competitions = process_competitions(data_json)
        self.places = process_places(data_json)
        for c in self.competitions:
            self.places[c[0]].agenda.append(c)
        for p in self.places.values():
            p.agenda.sort(key=lambda x: x[1])

        if self.is_states:
            init_states = Input_Initial()
        else:
            pops = Input_Population()
            init_states = np.zeros((self.num,6), dtype=float)
            for i in range(self.num):
                E0 = self.paras["initial_infect"]*2/self.num
                I0 = self.paras["initial_infect"]*1/self.num
                J0 = self.paras["initial_infect"]*1.5/self.num
                S0 = pops[i] - (E0+I0+J0)
                init_states[i] = np.array([S0,E0,I0,J0,0,0], dtype=float)

        # commuting matrix
        try:
            M = Input_Matrix()
            M /= (M.sum(axis=1, keepdims=True) + 1e-12)
            self.origin_matrix = M.copy()
            self.matrix = M.copy()
        except:
            self.origin_matrix = np.eye(self.num)
            self.matrix = np.eye(self.num)

        self.network = Network(self.num, init_states, self.matrix, self.paras)
        self.current_tick = 0

    def modify_matrix(self, matrix):
        row = matrix.sum(axis=1, keepdims=True)
        matrix_ = matrix / (row + 1e-12)
        self.network.A = matrix_.copy()
        self.matrix = matrix_.copy()

        # β联动：只放大/缩小 β2（通勤代表跨区混合强度）
        ratio = (np.sum(self.matrix) - np.trace(self.matrix)) / (np.sum(self.origin_matrix) - np.trace(self.origin_matrix) + 1e-12)
        self.network.paras["beta_2"] = max(0.01, BASE_PARAS["beta_2"] * ratio)

    def check_competition_start(self):
        for pid, pl in self.places.items():
            if not pl.agenda:
                continue
            comp = pl.agenda[0]
            if comp[1] == self.current_tick:
                pl.audience_from = np.zeros((self.num,6))
                pl.agenda.pop(0)
                pl.endtime = comp[2]
                capacity = comp[3]

                sums = np.array([np.sum(self.network.nodes[i].state[:5]) for i in range(self.num)])
                tot = max(1.0, np.sum(sums))
                for i in range(self.num):
                    portion = self.network.nodes[i].state[:5]*(capacity/tot)
                    portion = np.minimum(portion, self.network.nodes[i].state[:5])
                    pl.audience_from[i,:5] = portion
                    self.network.nodes[i].state[:5] -= portion

                pl.infect(self.network.paras)

    def check_competition_end(self):
        for pid, pl in self.places.items():
            if pl.endtime == self.current_tick:
                for i in range(self.num):
                    self.network.nodes[i].state[:5] += pl.audience_from[i,:5]
                pl.audience_from = []
                pl.endtime = None

# ---------- RL Env ----------
class CommuneMatrixEnv(gym.Env):
    """step=1小时；动作是23x23通勤矩阵。"""
    def __init__(self, days=20, is_states=False):
        super().__init__()
        self.days = days
        self.is_states = is_states
        self.se = Env(is_states=self.is_states)
        self.se.init_env()
        self.n = self.se.num

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.n*6+2,), dtype=np.float32)
        self.action_space = spaces.Box(low=0, high=1, shape=(self.n*self.n,), dtype=np.float32)

        self.current_hour = 0
        self.current_day = 0
        self.episode_steps = 0
        self.max_steps = 24*self.days

        self.original_matrix = self.se.origin_matrix.copy()

    def seed(self, seed=None):
        self.np_random, s = seeding.np_random(seed)
        return [s]

    def reset(self):
        self.se = Env(is_states=self.is_states)
        self.se.init_env()
        self.current_hour = 0
        self.current_day = 0
        self.episode_steps = 0
        self.original_matrix = self.se.origin_matrix.copy()
        return self._obs()

    def step(self, action):
        matrix = self._process_action(action)
        self.se.modify_matrix(matrix)

        self.se.check_competition_start()
        if self.current_hour == 8:
            self.se.network.morning_commuting()

        prev = np.array([nd.state.copy() for nd in self.se.network.nodes.values()])
        self.se.network.update_network()
        if self.current_hour == 18:
            self.se.network.evening_commuting()
        self.se.check_competition_end()

        now = np.array([nd.state for nd in self.se.network.nodes.values()])
        reward = self._reward(prev, now, matrix)

        # info
        prev_inf = np.sum(prev[:,1:4]); now_inf = np.sum(now[:,1:4])
        prev_d = np.sum(prev[:,5]);     now_d  = np.sum(now[:,5])
        ori_c = np.sum(self.original_matrix) - np.trace(self.original_matrix)
        cur_c = np.sum(matrix) - np.trace(matrix)
        commute_ratio = cur_c / (ori_c + 1e-12)
        info = {
            "infections": float(max(0, now_inf-prev_inf)),
            "deaths": float(max(0, now_d-prev_d)),
            "commute_ratio": float(commute_ratio)
        }

        # time update
        self.current_hour += 1
        if self.current_hour >= 24:
            self.current_hour = 0
            self.current_day += 1
        self.episode_steps += 1
        done = (self.episode_steps >= self.max_steps)

        return self._obs(), reward, done, info

    def _obs(self):
        arr = np.concatenate([self.se.network.nodes[i].state for i in range(self.n)], axis=0)
        return np.concatenate([arr, [self.current_day, self.current_hour]]).astype(np.float32)

    def _process_action(self, action):
        m = action.reshape(self.n, self.n)
        m = np.maximum(m, 0)
        rs = m.sum(axis=1, keepdims=True)
        m = m / (rs + 1e-12)
        for i in range(self.n):
            if m[i,i] < MIN_SELF_ACTIVITY:
                deficit = MIN_SELF_ACTIVITY - m[i,i]
                non_diag = 1.0 - m[i,i]
                if non_diag > 0:
                    scale = (non_diag - deficit) / (non_diag + 1e-12)
                    for j in range(self.n):
                        if j != i:
                            m[i,j] *= scale
                m[i,i] = MIN_SELF_ACTIVITY
        return m

    def _reward(self, prev, now, mat):
        prev_inf = np.sum(prev[:,1:4]);   now_inf = np.sum(now[:,1:4])
        prev_d   = np.sum(prev[:,5]);     now_d   = np.sum(now[:,5])
        new_inf = max(0, now_inf - prev_inf)
        new_d   = max(0, now_d - prev_d)

        ori_c = np.sum(self.original_matrix) - np.trace(self.original_matrix)
        cur_c = np.sum(mat) - np.trace(mat)
        ratio = cur_c / (ori_c + 1e-12)
        total_pop = np.sum(now[:,:5])

        # 权重可按需调整
        w_inf, w_d, w_c = -1.0, -3.0, +0.5
        rew = w_inf*new_inf + w_d*new_d + w_c*total_pop*ratio
        # 缩放使值幅适中（可用 TensorBoard 观察）
        return float(rew / 1e6)

# ---------- Baselines ----------
def baseline_original(env: Env):
    return env.origin_matrix.copy()

def baseline_diagonal(env: Env):
    M = np.zeros_like(env.origin_matrix)
    np.fill_diagonal(M, 1.0)
    return M

def baseline_neighbor(env: Env):
    M = (env.origin_matrix > 1e-9).astype(float)
    rs = M.sum(axis=1, keepdims=True)
    M = M/(rs + 1e-12)
    for i in range(env.num):
        if M[i,i] < MIN_SELF_ACTIVITY:
            deficit = MIN_SELF_ACTIVITY - M[i,i]
            non_diag = 1.0 - M[i,i]
            if non_diag > 0:
                scale = (non_diag - deficit) / (non_diag + 1e-12)
                for j in range(env.num):
                    if j != i:
                        M[i,j] *= scale
            M[i,i] = MIN_SELF_ACTIVITY
    return M

def baseline_random50(env: Env):
    n = env.num
    M = np.random.rand(n,n)
    M /= (M.sum(axis=1, keepdims=True) + 1e-12)
    for i in range(n):
        diag = 0.5
        diff = diag - M[i,i]
        non_diag = 1.0 - M[i,i]
        if diff > 0 and non_diag > 0:
            scale = (non_diag - diff) / (non_diag + 1e-12)
            for j in range(n):
                if j != i:
                    M[i,j] *= scale
            M[i,i] = diag
    return M

BASELINES = {
    "original": baseline_original,
    "diagonal": baseline_diagonal,
    "neighbor": baseline_neighbor,
    "random50": baseline_random50
}

# ---------- Train / Evaluate ----------
class LogCallback(BaseCallback):
    def __init__(self, name="", log_interval=2000, verbose=1):
        super().__init__(verbose)
        self.name = name
        self.log_interval = log_interval
        self.st = None
        self.last = None
    def _on_training_start(self):
        self.st = time.time()
        self.last = self.st
        print(f"[{self.name}] start training...")
    def _on_step(self):
        if self.n_calls % self.log_interval == 0:
            steps = self.model.num_timesteps
            total = self.model._total_timesteps
            pct = steps/total*100
            el = time.time()-self.st
            eta = el/(steps+1e-9)*(total-steps)
            print(f"[{self.name}] step {steps}/{total} ({pct:.1f}%) elapsed {el/60:.1f}m ETA {eta/60:.1f}m")
        return True

def _make_env(days=20, is_states=False, seed=0, n_envs=1):
    def thunk(s):
        def _fn():
            env = CommuneMatrixEnv(days=days, is_states=is_states)
            env.seed(s)
            return env
        return _fn
    if n_envs == 1:
        return DummyVecEnv([thunk(seed)])
    else:
        return SubprocVecEnv([thunk(seed+i) for i in range(n_envs)])

def train(exp_cfg: Dict):
    algo = exp_cfg['algo']
    exp_id = exp_cfg['exp_id']
    gpu = exp_cfg.get('gpu_id', 0)
    days = exp_cfg.get('days', 20)
    tsteps = exp_cfg.get('total_timesteps', 300_000)
    is_states = exp_cfg.get('is_states', False)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    seed = 10000 + exp_id
    set_random_seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    vec_env = _make_env(days=days, is_states=is_states, seed=seed, n_envs=1)
    name = f"{algo}_exp{exp_id}"
    cb = LogCallback(name=name, log_interval=2000)

    if algo == "PPO":
        model = PPO("MlpPolicy", vec_env, device="cuda", verbose=0,
                    tensorboard_log=f"./tensorboard/{name}",
                    learning_rate=3e-4, batch_size=128,
                    n_steps=2048, n_epochs=10, gae_lambda=0.95,
                    gamma=0.99, clip_range=0.2, ent_coef=0.01,
                    policy_kwargs=dict(net_arch=[dict(pi=[256,256,128], vf=[256,256,128])]))
    else:
        model = A2C("MlpPolicy", vec_env, device="cuda", verbose=0,
                    tensorboard_log=f"./tensorboard/{name}",
                    learning_rate=7e-4, n_steps=5, gamma=0.99,
                    ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5,
                    policy_kwargs=dict(net_arch=[dict(pi=[256,256,128], vf=[256,256,128])]))
    model.learn(total_timesteps=tsteps, callback=cb, progress_bar=True)
    out = f"./models/{name}_final.zip"
    model.save(out); vec_env.close()
    print(f"[{name}] saved => {out}")
    return out

def evaluate_model(model_path: str, algo: str, exp_id: int, days=20, is_states=False, n_episodes=1):
    # load model
    model = (PPO if algo=="PPO" else A2C).load(model_path)
    name = f"{algo}_exp{exp_id}"
    results = {"method": name, "episodes": []}

    for ep in range(n_episodes):
        env = CommuneMatrixEnv(days=days, is_states=is_states)
        obs = env.reset()
        done = False
        ep_data = {
            "hourly_infections": [], "hourly_deaths": [],
            "hourly_commute_ratio": [], "hourly_reward": [],
            "policy_matrices": [], "district_infections": []
        }
        dist_series = [[] for _ in range(env.n)]
        step = 0; ep_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_reward += reward
            ep_data["hourly_infections"].append(float(info["infections"]))
            ep_data["hourly_deaths"].append(float(info["deaths"]))
            ep_data["hourly_commute_ratio"].append(float(info["commute_ratio"]))
            ep_data["hourly_reward"].append(float(reward))
            # district E+I+J
            for d in range(env.n):
                dist_series[d].append(float(env.se.network.nodes[d].state[1:4].sum()))
            if step % 24 == 0:
                ep_data["policy_matrices"].append(env.se.matrix.copy().tolist())
            step += 1
        ep_data["district_infections"] = dist_series
        ep_data["total_reward"] = float(ep_reward)
        results["episodes"].append(ep_data)

    out_dir = f"./results/{name}/"; os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir,"evaluation.json"), "w") as f:
        json.dump(results, f, indent=2)
    # 简易时序图
    last = results["episodes"][-1]; T = len(last["hourly_infections"])
    x = np.arange(T)
    plt.figure(figsize=(12,8))
    plt.subplot(4,1,1); plt.plot(x,last["hourly_infections"]); plt.title("hourly_infections"); plt.grid(alpha=.3)
    plt.subplot(4,1,2); plt.plot(x,last["hourly_deaths"]); plt.title("hourly_deaths"); plt.grid(alpha=.3)
    plt.subplot(4,1,3); plt.plot(x,last["hourly_commute_ratio"]); plt.title("hourly_commute_ratio"); plt.grid(alpha=.3)
    plt.subplot(4,1,4); plt.plot(x,last["hourly_reward"]); plt.title("hourly_reward"); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir,"evaluation_timeseries.png")); plt.close()
    return results

def evaluate_baseline(basename: str, days=20, n_episodes=1, is_states=False):
    out_dir = f"./results/baseline_{basename}/"; os.makedirs(out_dir, exist_ok=True)
    res = {"method": f"baseline_{basename}", "episodes": []}
    for ep in range(n_episodes):
        env = CommuneMatrixEnv(days=days, is_states=is_states)
        obs = env.reset(); done = False
        ep_data = {"hourly_infections": [], "hourly_deaths": [], "hourly_commute_ratio": [], "hourly_reward": []}
        ep_reward = 0.0
        M = BASELINES[basename](env.se)
        while not done:
            action = M.flatten()
            obs, reward, done, info = env.step(action)
            ep_reward += reward
            ep_data["hourly_infections"].append(float(info["infections"]))
            ep_data["hourly_deaths"].append(float(info["deaths"]))
            ep_data["hourly_commute_ratio"].append(float(info["commute_ratio"]))
            ep_data["hourly_reward"].append(float(reward))
        ep_data["total_reward"] = float(ep_reward)
        res["episodes"].append(ep_data)
    with open(os.path.join(out_dir, "baseline_evaluation.json"), "w") as f:
        json.dump(res, f, indent=2)
    return res

# ---------- per-experiment strategy comparison (含 cumulative reward) ----------
def _build_derived(ep):
    arr = {}
    arr["hourly_cumulative_infections"] = np.cumsum(ep["hourly_infections"]).tolist()
    arr["hourly_cumulative_deaths"]     = np.cumsum(ep["hourly_deaths"]).tolist()
    # 平均通勤
    avg = []; s=0.0
    for i,v in enumerate(ep["hourly_commute_ratio"]):
        s += v; avg.append(s/(i+1))
    arr["hourly_average_commute"] = avg
    # 累计奖励
    s=0.0; cr=[]
    for v in ep["hourly_reward"]:
        s += v; cr.append(s)
    arr["hourly_cumulative_reward"] = cr
    return arr

def per_experiment_strategy_comparison(algo, exp_id):
    name = f"{algo}_exp{exp_id}"
    rl_path = f"./results/{name}/evaluation.json"
    if not os.path.exists(rl_path):
        return
    with open(rl_path,"r") as f:
        rl = json.load(f)["episodes"][-1]
    rl_d = _build_derived(rl)

    # baselines
    b_dict = {}
    for b in BASELINES.keys():
        p = f"./results/baseline_{b}/baseline_evaluation.json"
        if not os.path.exists(p):
            evaluate_baseline(b, days=20, n_episodes=1, is_states=False)
        with open(p,"r") as f:
            be = json.load(f)["episodes"][-1]
        b_dict[b] = _build_derived(be)

    # 画图：四行 cumulative/avg
    out_dir = f"./results/{name}/"
    plt.figure(figsize=(16,18))
    ms = [("hourly_cumulative_infections","Cumulative Infections"),
          ("hourly_cumulative_deaths","Cumulative Deaths"),
          ("hourly_average_commute","Average Commute Ratio"),
          ("hourly_cumulative_reward","Cumulative Reward")]
    for i,(k,title) in enumerate(ms):
        plt.subplot(4,1,i+1)
        plt.plot(rl_d[k], label=name, linewidth=2)
        for bn,bd in b_dict.items():
            plt.plot(bd[k], label=f"baseline_{bn}", linestyle="--", alpha=0.9)
        plt.title(title); plt.grid(alpha=.3); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(out_dir,"strategy_comparison.png")); plt.close()

# ---------- parallel helpers ----------
def _run_single(cfg):
    model_path = train(cfg)
    if model_path:
        evaluate_model(model_path, cfg["algo"], cfg["exp_id"], days=cfg.get("days",20), is_states=cfg.get("is_states",False))
        per_experiment_strategy_comparison(cfg["algo"], cfg["exp_id"])

def parallel_train_eval(configs: List[Dict]):
    ps=[]
    for i,cfg in enumerate(configs):
        p = multiprocessing.Process(target=_run_single, args=(cfg,))
        p.start(); ps.append(p); time.sleep(2)
    for p in ps: p.join()

def ensure_all_baselines(days=20, is_states=False):
    for b in BASELINES.keys():
        p = f"./results/baseline_{b}/baseline_evaluation.json"
        if not os.path.exists(p):
            evaluate_baseline(b, days=days, n_episodes=1, is_states=is_states)

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser("Task2 - main_RL_task2_910.py")
    ap.add_argument("--mode", choices=["train","evaluate","baseline","visualize","all"], default="all")
    ap.add_argument("--algos", nargs="+", default=["PPO","A2C"])
    ap.add_argument("--exp-start", type=int, default=1)
    ap.add_argument("--exp-count", type=int, default=8)
    ap.add_argument("--timesteps", type=int, default=300000)
    ap.add_argument("--days", type=int, default=20)
    ap.add_argument("--is-states", action="store_true")
    args = ap.parse_args()

    exp_ids = list(range(args.exp_start, args.exp_start+args.exp_count))
    cfgs=[]
    gpu=0
    for algo in args.algos:
        for eid in exp_ids:
            cfgs.append({
                "algo": algo, "exp_id": eid, "gpu_id": gpu,
                "days": args.days, "is_states": args.is_states,
                "total_timesteps": args.timesteps
            })
            gpu = (gpu+1) % max(1, NUM_GPUS)

    if args.mode in ["train","all"]:
        parallel_train_eval(cfgs)
    if args.mode in ["baseline","all"]:
        ensure_all_baselines(days=args.days, is_states=args.is_states)
    if args.mode in ["evaluate",]:
        # 只评估已训练模型
        for c in cfgs:
            p = f"./models/{c['algo']}_exp{c['exp_id']}_final.zip"
            if os.path.exists(p):
                evaluate_model(p, c["algo"], c["exp_id"], days=c["days"], is_states=c["is_states"])
                per_experiment_strategy_comparison(c["algo"], c["exp_id"])
    if args.mode in ["visualize","all"]:
        from visualize_task2_912 import run_all_visualizations
        run_all_visualizations("./results")

if __name__ == "__main__":
    main()
