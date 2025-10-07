#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
main_A2C_914.py

升级要点：
1) 动力学统一为 717 版“双 β”SEIJRD：dS,dE 同时受 E 与 (I+J) 传染项影响；
   并与通勤强度联动：在 Env.modify_matrix 中按 off-diagonal 比例共同缩放 beta_1、beta_2。
2) 训练/评估/基线/综合可视化一键跑，与 910 版 pipeline 与 advanced_visuals.py 对齐。
3) 评估期完整记录小时级四条时序（感染/死亡/通勤/奖励）、district_infections、每日策略矩阵；
   全部转为 Python 原生类型写入 JSON，避免 ndarray 序列化报错。
4) 默认一次性 16 个实验（--exp-count 默认=16），支持 PPO/A2C（默认两个都跑）。
5) 环境步长采用“**A2 第二种方案**”：**以小时为 step**（一集=24*days 步），每小时可更新通勤矩阵。
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

# --------------------- 路径与目录 ---------------------
os.makedirs("./models", exist_ok=True)
os.makedirs("./logs", exist_ok=True)
os.makedirs("./results", exist_ok=True)
os.makedirs("./visualizations", exist_ok=True)
os.makedirs("./tensorboard", exist_ok=True)

# --------------------- 硬件参数 ---------------------
NUM_GPUS = 8
MAX_POP = 1e12

# --------------------- 717 版 双β + 典型参数 ---------------------
# 取自你“717 版”的实现并适度清洗为本脚本可直接用的字典结构
# （beta_1 驱动 S-E 接触自 E；beta_2 驱动 S 与 (I+J) 接触） 参照 717 版实现。  # noqa
backup_paras = {
    "beta_1": 0.0614,
    "beta_2": 0.0696,
    "gamma_i": 0.0496,
    "gamma_j": 0.0376,
    "sigma_i": 0.01378,
    "sigma_j": 0.03953,
    "mu_i": 2.0e-5,
    "mu_j": 2.7e-4,
    "rho": 8.62e-5,          # 场馆接触传播基率（用于 Place.infect）
    "initial_infect": 500,   # 初始感染
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

# ================== 赛事 / 场馆（与 910/58 系列对齐） ==================
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
            comps.append((vid, st_tick, ed_tick, cap, slot))
    return comps

class Place:
    def __init__(self, data: Dict):
        self.id = data["venue_id"]
        self.name = data["venue_name"]
        self.capacity = data["capacity"]
        self.agenda = []       # list of (vid, start_tick, end_tick, cap, slot)
        self.endtime = None
        self.audience_from = []  # shape = (node_num, 6)

    def infect(self, paras):
        """场馆内传播：把所有到场人群按行求和后，用 E/I/J 作为传染源，按 rho 计算新感染。"""
        if len(self.audience_from) == 0:
            return
        arr = np.sum(self.audience_from, axis=0)  # [S,E,I,J,R,D]
        infectious_num = arr[1] + arr[2] + arr[3]
        if infectious_num <= 0:
            return
        prob = 1 - (1 - paras["rho"]) ** infectious_num
        prob = np.clip(prob, 0, 1)

        S_tot = arr[0]
        if S_tot < 1e-9:
            return
        new_inf = S_tot * prob

        frac_sus = self.audience_from[:, 0] / (S_tot + 1e-12)
        new_inf_each = frac_sus * new_inf
        self.audience_from[:, 0] -= new_inf_each
        self.audience_from[:, 1] += new_inf_each

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

# ================== SEIR 节点/网络：双β ODE（717 版） ==================
class SEIR_Node:
    def __init__(self, state, total_nodes):
        self.state = state.astype(float)         # [S,E,I,J,R,D]
        self.total_nodes = total_nodes
        self.from_node = np.zeros((total_nodes, 6))
        self.to_node = np.zeros((total_nodes, 5))

    def update_seir(self, delta_time=1/2400, times=100, paras=backup_paras):
        """
        双 β 动力学（717）:
          dS = -β1 * S * E / N  - β2 * S * (I+J) / N
          dE = +β1 * S * E / N  + β2 * S * (I+J) / N - σi*E - σj*E
          dI =  σi*E - γi*I - μi*I
          dJ =  σj*E - γj*J - μj*J
          dR =  γi*I + γj*J
          dD =  μi*I + μj*J
        """
        S, E, I, J, R, D_ = self.state
        N = S + E + I + J + R
        if N > 0:
            dS = -(paras["beta_1"] * S * E / N) - (paras["beta_2"] * S * (I + J) / N)
            dE = (paras["beta_1"] * S * E / N) + (paras["beta_2"] * S * (I + J) / N) \
                 - paras["sigma_i"] * E - paras["sigma_j"] * E
            dI = paras["sigma_i"] * E - paras["gamma_i"] * I - paras["mu_i"] * I
            dJ = paras["sigma_j"] * E - paras["gamma_j"] * J - paras["mu_j"] * J
            dR = paras["gamma_i"] * I + paras["gamma_j"] * J
            dD = paras["mu_i"] * I + paras["mu_j"] * J
        else:
            dS = dE = dI = dJ = dR = dD = 0.0

        delta = np.array([dS, dE, dI, dJ, dR, dD]) * delta_time
        for _ in range(times):
            self.state += delta
            # 同 910/58 系列：把本节点的“增长量”按 from_node 占本节点份额分配回程
            Nvalue = np.sum(self.from_node, axis=1)    # (total_nodes,)
            ratio = (Nvalue / max(N, 1e-12))[:, None]  # (total_nodes,1)
            self.from_node += delta[None, :] * ratio

        self.state = np.clip(self.state, 0, MAX_POP)
        self.from_node = np.clip(self.from_node, 0, MAX_POP)

class SEIR_Network:
    def __init__(self, num, states, matrix=None, paras=backup_paras):
        self.node_num = num
        self.nodes = {i: SEIR_Node(states[i], num) for i in range(num)}
        if matrix is not None:
            row_sum = matrix.sum(axis=1, keepdims=True)
            self.A = matrix / (row_sum + 1e-12)
        else:
            A_ = np.random.rand(num, num)
            A_ /= A_.sum(axis=1, keepdims=True)
            self.A = A_
        self.paras = paras
        self.delta_time = 1 / 2400

    def morning_commuting(self):
        # 按行外积分发 to_node，然后“倾巢而出”
        for i in range(self.node_num):
            SEIJR = self.nodes[i].state[:5]
            self.nodes[i].to_node = np.outer(self.A[i], SEIJR)
        for i in range(self.node_num):
            self.nodes[i].state[:5] = 0.0
        # 汇聚到目的地、同时登记 from_node
        for i in range(self.node_num):
            for j in range(self.node_num):
                self.nodes[i].state[:5] += self.nodes[j].to_node[i]
                self.nodes[j].from_node[i, :5] = self.nodes[i].to_node[j]

    def evening_commuting(self):
        # 回家
        for i in range(self.node_num):
            for j in range(self.node_num):
                if i == j:
                    continue
                self.nodes[j].state -= self.nodes[j].from_node[i]
                self.nodes[i].state += self.nodes[j].from_node[i]
        # 清空流动记录
        for i in range(self.node_num):
            self.nodes[i].from_node = np.zeros((self.node_num, 6))
            self.nodes[i].to_node = np.zeros((self.node_num, 5))

    def update_network(self):
        for i in range(self.node_num):
            self.nodes[i].update_seir(self.delta_time, 100, self.paras)

# ================== 仿真环境（小时步长，A2 方案②） ==================
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
        self.paras = backup_paras.copy()  # 将在 modify_matrix 中动态更新 beta_1/2

        # 记录 base beta_1/2 以便缩放
        self._base_beta_1 = self.paras["beta_1"]
        self._base_beta_2 = self.paras["beta_2"]

    def init_env(self):
        data_json = json.load(open(file_path, "r", encoding="utf-8"))
        self.competitions = process_competitions(data_json)
        self.places = process_places(data_json)
        for c in self.competitions:
            vid = c[0]
            self.places[vid].agenda.append(c)
        for pl in self.places.values():
            pl.agenda.sort(key=lambda x: x[1])

        # 初始化 SEIJRD 状态
        if self.is_states and os.path.exists("data/SEIJRD.csv"):
            init_states = Input_Intial("data/SEIJRD.csv")
        else:
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
            mat = mat / (row_sum + 1e-12)
            self.origin_matrix = mat.copy()
            self.matrix = mat.copy()
        except Exception:
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
                slot_id = comp[4]

                # 简单均匀按各区 SEIJR 总量抽人到场
                sums = np.array([np.sum(self.network.nodes[i].state[:5]) for i in range(self.num)])
                tot = np.sum(sums)
                tot = tot if tot > 1e-9 else 1.0
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
        """行归一化 + 与通勤强度联动缩放 (beta_1, beta_2)。"""
        row_sum = matrix.sum(axis=1, keepdims=True)
        matrix_ = matrix / (row_sum + 1e-12)
        self.network.A = matrix_.copy()
        self.matrix = matrix_.copy()

        # 以 off-diagonal 总和比值作为通勤强度
        base_off = np.sum(self.origin_matrix) - np.trace(self.origin_matrix)
        now_off  = np.sum(self.matrix)        - np.trace(self.matrix)
        ratio = now_off / (base_off + 1e-12)     # ratio<1 => 更少跨区流动

        # 对 beta_1/2 共缩放（设置 0.05 的“流动下限”做线性映射，与你旧代码风格一致）
        new_beta_1 = 0.05 + (self._base_beta_1 - 0.05) * ratio
        new_beta_2 = 0.05 + (self._base_beta_2 - 0.05) * ratio
        self.network.paras["beta_1"] = max(0.01, new_beta_1)
        self.network.paras["beta_2"] = max(0.01, new_beta_2)

# ================== RL 环境（小时步长；每步可改矩阵） ==================
class CommuneMatrixEnv(gym.Env):
    """ 每个 step = 1 小时；一集时长 = 24*days。 """

    def __init__(self, days=20, is_states=False):
        super().__init__()
        self.days = days
        self.is_states = is_states

        self.seir_env = Env(is_states=self.is_states)
        self.seir_env.init_env()
        self.num = self.seir_env.num

        obs_dim = self.num * 6 + 2  # 所有区 SEIJRD + day + hour
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
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

        prev_infect = np.sum(prev_states[:, 1:4])
        curr_infect = np.sum(curr_states[:, 1:4])
        infs = max(0, curr_infect - prev_infect)

        prev_d = np.sum(prev_states[:, 5])
        curr_d = np.sum(curr_states[:, 5])
        deths = max(0, curr_d - prev_d)

        info = {'infections': float(infs), 'deaths': float(deths)}
        return obs, reward, done, info

    def _get_obs(self):
        arr = np.concatenate([self.seir_env.network.nodes[i].state for i in range(self.num)], axis=0)
        return np.concatenate([arr, [self.current_day, self.current_hour]]).astype(np.float32)

    def _process_action(self, action):
        mat = action.reshape(self.num, self.num)
        mat = np.maximum(mat, 0)
        row_sum = mat.sum(axis=1, keepdims=True)
        mat = mat / (row_sum + 1e-12)
        # 每区本地最小活动比例
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
        # 简洁线性加权：奖励鼓励通勤（经济）且惩罚新增感染与死亡
        w_inf, w_dth, w_cmt = -1.0, -3.0, 0.5
        reward = w_inf * new_inf + w_dth * new_d + w_cmt * total_pop * ratio
        # 数值尺度压缩，稳定学习
        return float(reward / 1e6)

# ================== 基线策略 ==================
def baseline_original(env):  # 不动
    return env.origin_matrix.copy()

def baseline_diagonal(env):  # 完全居家
    mat = np.zeros_like(env.origin_matrix)
    np.fill_diagonal(mat, 1.0)
    return mat

def baseline_neighbor(env):  # 仅保留原矩阵中“有边”的位置，再行归一化
    mat = (env.origin_matrix > 1e-12).astype(float)
    rs = mat.sum(axis=1, keepdims=True)
    mat = mat / (rs + 1e-12)
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

def baseline_random50(env):  # 对角线 0.5，其余随机
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

BASELINE_STRATEGIES = {
    "original": baseline_original,
    "diagonal": baseline_diagonal,
    "neighbor": baseline_neighbor,
    "random50": baseline_random50,
}

# ================== 训练 / 评估 ==================
class CustomCallback(BaseCallback):
    def __init__(self, exp_name="", log_interval=2000, verbose=1):
        super().__init__(verbose)
        self.exp_name = exp_name
        self.log_interval = log_interval
        self.start_time = None
        self.last_save = None

    def _on_training_start(self):
        self.start_time = time.time()
        self.last_save = self.start_time
        print(f"[{self.exp_name}] start training...")

    def _on_step(self):
        if self.n_calls % self.log_interval == 0:
            steps = self.model.num_timesteps
            total_steps = self.model._total_timesteps
            pct = steps / max(total_steps, 1) * 100
            elapsed = time.time() - self.start_time
            remain = elapsed / max(steps, 1) * max(total_steps - steps, 1)
            print(f"[{self.exp_name}] step {steps}/{total_steps}({pct:.1f}%), "
                  f"elapsed={elapsed/60:.1f}m, ETA={remain/60:.1f}m")
            if time.time() - self.last_save > 600:
                ckpt = f"./models/{self.exp_name}_step{steps}.zip"
                self.model.save(ckpt)
                self.last_save = time.time()
                print(f"[{self.exp_name}] checkpoint => {ckpt}")
        return True

def make_env(n_envs=1, seeds=[0], env_kwargs=None):
    def env_fn(seed):
        def _thunk():
            env = CommuneMatrixEnv(**(env_kwargs or {}))
            env.seed(seed)
            return env
        return _thunk
    if n_envs == 1:
        return DummyVecEnv([env_fn(seeds[0])])
    else:
        return SubprocVecEnv([env_fn(s) for s in seeds])

def train_model(exp_cfg):
    algo = exp_cfg['algo']
    exp_id = exp_cfg['exp_id']
    gpu = exp_cfg.get('gpu_id', 0)
    days = exp_cfg.get('days', 20)
    is_states = exp_cfg.get('is_states', False)
    total_timesteps = exp_cfg.get('total_timesteps', 1_000_000)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    set_random_seed(10_000 + exp_id)
    np.random.seed(10_000 + exp_id)
    torch.manual_seed(10_000 + exp_id)

    env_kwargs = {'days': days, 'is_states': is_states}
    vec_env = make_env(n_envs=1, seeds=[10_000 + exp_id], env_kwargs=env_kwargs)

    exp_name = f"{algo}_exp{exp_id}"
    cb = CustomCallback(exp_name=exp_name, log_interval=2000)
    ckpt_cb = CheckpointCallback(save_freq=50_000, save_path=f"./models/{exp_name}_ckpts/", name_prefix="model")

    if algo == "PPO":
        model = PPO("MlpPolicy", vec_env, device="cuda", verbose=0,
                    tensorboard_log=f"./tensorboard/{exp_name}",
                    learning_rate=3e-4, n_steps=2048, batch_size=128,
                    n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
                    ent_coef=0.01,
                    policy_kwargs=dict(net_arch=[dict(pi=[256, 256, 128], vf=[256, 256, 128])]))
    elif algo == "A2C":
        model = A2C("MlpPolicy", vec_env, device="cuda", verbose=0,
                    tensorboard_log=f"./tensorboard/{exp_name}",
                    learning_rate=7e-4, n_steps=5, gamma=0.99,
                    ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5,
                    policy_kwargs=dict(net_arch=[dict(pi=[256, 256, 128], vf=[256, 256, 128])]))
    else:
        raise ValueError("Unsupported algo")

    model.learn(total_timesteps=total_timesteps, callback=[cb, ckpt_cb], progress_bar=True)
    final_path = f"./models/{exp_name}_final.zip"
    model.save(final_path)
    vec_env.close()
    print(f"[{exp_name}] training done => {final_path}")
    return final_path

def evaluate_model(model_path, algo, exp_id, n_episodes=3, days=20, is_states=False):
    model = PPO.load(model_path) if algo == "PPO" else A2C.load(model_path)

    results = {'method': f"{algo}_exp{exp_id}", 'episodes': []}
    for ep in range(n_episodes):
        env = CommuneMatrixEnv(days=days, is_states=is_states)
        obs = env.reset()
        done = False
        ep_reward = 0.0

        ep_data = {
            'hourly_infections': [],
            'hourly_deaths': [],
            'hourly_commute_ratio': [],
            'hourly_reward': [],
            'district_infections': [],   # list[list[float]] (23 x T)
            'policy_matrices': []        # 按天记录策略矩阵
        }
        dist_infs = [[] for _ in range(env.num)]
        t = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_reward += float(reward)

            # 4 条小时级曲线
            ep_data['hourly_reward'].append(float(reward))
            ep_data['hourly_infections'].append(float(info['infections']))
            ep_data['hourly_deaths'].append(float(info['deaths']))

            mat = env.seir_env.matrix
            ori_c = np.sum(env.original_matrix) - np.trace(env.original_matrix)
            now_c = np.sum(mat) - np.trace(mat)
            c_ratio = now_c / (ori_c + 1e-12)
            ep_data['hourly_commute_ratio'].append(float(c_ratio))

            # district infections (E+I+J)
            for d in range(env.num):
                eij = float(env.seir_env.network.nodes[d].state[1:4].sum())
                dist_infs[d].append(eij)

            # 每天记录一次策略矩阵
            if t % 24 == 0:
                ep_data['policy_matrices'].append(mat.copy().tolist())
            t += 1

        ep_data['total_reward'] = float(ep_reward)
        ep_data['district_infections'] = dist_infs
        results['episodes'].append(ep_data)

    out_dir = f"./results/{algo}_exp{exp_id}/"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "evaluation.json"), "w") as f:
        json.dump(results, f, indent=2)
    _save_eval_plots(results, out_dir)
    print(f"[{algo}_exp{exp_id}] evaluate done => {out_dir}/evaluation.json")
    return results

def _save_eval_plots(results, out_dir):
    """保存单实验的 4 合 1 时序图 + 矩阵对比，全部使用 Python 原生类型。"""
    last = results['episodes'][-1]
    T = len(last['hourly_infections'])
    x = np.arange(T)

    plt.figure(figsize=(12, 8))
    ax = plt.subplot(4, 1, 1); ax.plot(x, last['hourly_infections']); ax.set_title("hourly_infections"); ax.grid(alpha=.3)
    ax = plt.subplot(4, 1, 2); ax.plot(x, last['hourly_deaths']);     ax.set_title("hourly_deaths");     ax.grid(alpha=.3)
    ax = plt.subplot(4, 1, 3); ax.plot(x, last['hourly_commute_ratio']); ax.set_title("hourly_commute_ratio"); ax.grid(alpha=.3)
    ax = plt.subplot(4, 1, 4); ax.plot(x, last['hourly_reward']);     ax.set_title("hourly_reward");     ax.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "evaluation_timeseries.png"))
    plt.close()

    # 矩阵首日与末日对比（若有记录）
    if last['policy_matrices']:
        env_tmp = CommuneMatrixEnv()
        orig = env_tmp.original_matrix
        first = np.array(last['policy_matrices'][0])
        final = np.array(last['policy_matrices'][-1])
        change = (final - orig) / (orig + 1e-12)

        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1); sns.heatmap(orig,  cmap="Blues",  cbar=True); plt.title("Original Matrix")
        plt.subplot(1, 3, 2); sns.heatmap(final, cmap="Blues",  cbar=True); plt.title("Final Policy Matrix")
        plt.subplot(1, 3, 3); sns.heatmap(change, cmap="RdBu_r", center=0, cbar=True); plt.title("Relative Change")
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, "matrix_comparison.png")); plt.close()

# ================== 并行训练 / 评估 / 基线 ==================
def parallel_train_models(configs, gpu_list):
    procs = []
    for cfg in configs:
        gpu = gpu_list[cfg['exp_id'] % max(len(gpu_list), 1)]
        cfg = dict(cfg, gpu_id=gpu)
        p = multiprocessing.Process(target=train_model, args=(cfg,))
        p.start()
        procs.append(p)
        time.sleep(1)
    for p in procs:
        p.join()
    print("All training processes done.")

def parallel_evaluate_models(configs):
    tasks = []
    for cfg in configs:
        algo, exp_id = cfg['algo'], cfg['exp_id']
        modelp = f"./models/{algo}_exp{exp_id}_final.zip"
        if not os.path.exists(modelp):
            print(f"[Skip] model not found: {modelp}")
            continue
        tasks.append((modelp, algo, exp_id, 3, cfg.get('days', 20), cfg.get('is_states', False)))
    with Pool(processes=min(multiprocessing.cpu_count(), max(1, len(tasks)))) as pool:
        pool.starmap(evaluate_model, tasks)

def evaluate_all_baselines(days=20, is_states=False):
    for bname, bfunc in BASELINE_STRATEGIES.items():
        out_dir = f"./results/baseline_{bname}/"
        os.makedirs(out_dir, exist_ok=True)
        results = {'method': f"baseline_{bname}", 'episodes': []}
        env = CommuneMatrixEnv(days=days, is_states=is_states)
        obs = env.reset()
        done = False
        ep_reward = 0.0
        ep_data = {
            'hourly_infections': [], 'hourly_deaths': [],
            'hourly_commute_ratio': [], 'hourly_reward': []
        }
        mat_fix = bfunc(env.seir_env)
        t = 0
        while not done:
            action = mat_fix.flatten()
            obs, reward, done, info = env.step(action)
            ep_reward += float(reward)
            ep_data['hourly_reward'].append(float(reward))
            ep_data['hourly_infections'].append(float(info['infections']))
            ep_data['hourly_deaths'].append(float(info['deaths']))
            mat = env.seir_env.matrix
            ori_c = np.sum(env.original_matrix) - np.trace(env.original_matrix)
            now_c = np.sum(mat) - np.trace(mat)
            ep_data['hourly_commute_ratio'].append(float(now_c / (ori_c + 1e-12)))
            t += 1
        ep_data['total_reward'] = float(ep_reward)
        results['episodes'].append(ep_data)
        with open(os.path.join(out_dir, "baseline_evaluation.json"), "w") as f:
            json.dump(results, f, indent=2)

        # 简单时序图
        T = len(ep_data['hourly_infections']); x = np.arange(T)
        plt.figure(figsize=(12, 8))
        for i, (k, ttl) in enumerate([
            ('hourly_infections', 'hourly_infections'),
            ('hourly_deaths', 'hourly_deaths'),
            ('hourly_commute_ratio', 'hourly_commute_ratio'),
            ('hourly_reward', 'hourly_reward')
        ]):
            plt.subplot(4, 1, i + 1); plt.plot(x, ep_data[k]); plt.title(ttl); plt.grid(alpha=.3)
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, "baseline_timeseries.png")); plt.close()
        print(f"[baseline_{bname}] evaluate done => {out_dir}/baseline_evaluation.json")

# ================== 可视化汇总（依赖 advanced_visuals.py） ==================
def run_visualization(args):
    try:
        from advanced_visuals import run_all_visualization_improvements
        run_all_visualization_improvements("./results")
    except Exception as e:
        print(f"advanced_visuals.py 不可用或报错，可忽略或稍后单独运行。错误: {e}")

# ================== 主入口 ==================
def main():
    parser = argparse.ArgumentParser(description="SEIJRD (717 双β) - CommuneMatrix RL (小时步长)")
    parser.add_argument("--mode", type=str, choices=["train", "evaluate", "baseline", "visualize", "all"], default="all")
    parser.add_argument("--algos", nargs="+", default=["PPO", "A2C"], help="算法列表: A2C / PPO")
    parser.add_argument("--exp-start", type=int, default=1)
    parser.add_argument("--exp-count", type=int, default=16, help="默认 16 组实验")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--days", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--is-states", action="store_true", help="是否读取 data/SEIJRD.csv 作为初值")
    args = parser.parse_args()

    # 组装配置
    configs = []
    for algo in args.algos:
        for eid in range(args.exp_start, args.exp_start + args.exp_count):
            configs.append({
                'exp_id': eid,
                'algo': algo,
                'total_timesteps': args.timesteps,
                'days': args.days,
                'is_states': args.is_states
            })

    if args.mode in ("train", "all"):
        print("===== STEP 1: TRAIN =====")
        gpu_list = list(range(NUM_GPUS)) if NUM_GPUS > 0 else [0]
        parallel_train_models(configs, gpu_list)

    if args.mode in ("evaluate", "all"):
        print("===== STEP 2: EVALUATE RL MODELS =====")
        parallel_evaluate_models(configs)

    if args.mode in ("baseline", "all"):
        print("===== STEP 3: EVALUATE BASELINES =====")
        evaluate_all_baselines(days=args.days, is_states=args.is_states)

    if args.mode in ("visualize", "all"):
        print("===== STEP 4: VISUALIZATION =====")
        run_visualization(args)

    print("Done.")

if __name__ == "__main__":
    main()
