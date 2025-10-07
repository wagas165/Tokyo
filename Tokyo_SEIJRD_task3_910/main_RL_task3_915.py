#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Task-3 主程序：按区隔离 (I -> (1-a_i)I) + 双β动力学 + 与 β2 联动
- 一键支持：train / evaluate / baseline / visualize / all
- 评估输出 evaluation.json/xlsx + 每个实验自己的 strategy_comparison.png（对所有 Task-3 baseline 的对照）+ evaluation_timeseries.png（含 cumulative reward）
- 全局可视化复用 advanced_visuals.py（Task-2/Task-3 通用）
"""

import os
import json
import time
import argparse
import numpy as np
import pandas as pd
import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

# RL
import torch
import gym
from gym import spaces
from gym.utils import seeding
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.utils import set_random_seed

# 保证目录
os.makedirs("./models", exist_ok=True)
os.makedirs("./logs", exist_ok=True)
os.makedirs("./results", exist_ok=True)
os.makedirs("./visualizations", exist_ok=True)
os.makedirs("./tensorboard", exist_ok=True)

NUM_GPUS = int(os.environ.get("NUM_GPUS", 8))
MAX_POP = 1e12
DATA_JSON = "data/end_copy.json"

# ===== 双β参数（与717版本风格一致，可按需微调） =====
PARAS = {
    "beta_1": 0.0614,   # E 传播强度
    "beta_2": 0.0696,   # I+J 传播强度
    "gamma_i": 0.0496,
    "gamma_j": 0.0376,
    "sigma_i": 0.01378,
    "sigma_j": 0.03953,
    "mu_i": 2e-5,
    "mu_j": 0.00027,
    "rho": 8.62e-05,    # 场馆内接触概率
    "initial_infect": 500
}

# ===== I 类按区隔离与 β2 的联动系数（默认 0.75） =====
BETA2_K = 0.75

# ===== CSV / 初值读取（与 Task-2 样式对齐） =====
def Input_Population(fp="data/tokyo_population.csv"):
    df = pd.read_csv(fp, header=None, names=["code","ward_jp","ward_en","dummy","population"])
    df["population"] = df["population"].astype(int)
    return df["population"].tolist()

def Input_Matrix(fp="data/tokyo_commuting_flows_with_intra.csv"):
    df = pd.read_csv(fp, index_col=0)
    df = df.apply(pd.to_numeric, errors='coerce')
    return df.values

def Input_Initial(fp="data/SEIJRD.csv"):
    df = pd.read_csv(fp, index_col=None).astype(float)
    return df.values

# ===== 场馆与赛程处理（同一风格） =====
def process_competitions(data_file):
    slot_mapping = {0:(8,11), 1:(13,17), 2:(19,22)}
    year = 2021
    earliest = None
    for venue in data_file:
        for t in venue["time"]:
            ed = datetime.date(year, t["month"], t["day"])
            if earliest is None or ed < earliest:
                earliest = ed
    comps = []
    for venue in data_file:
        vid = venue["venue_id"]
        cap = int(venue["capacity"].replace(',', ''))
        for t in venue["time"]:
            ed = datetime.date(year, t["month"], t["day"])
            diff = (ed - earliest).days
            sh,eh = slot_mapping[t["slot"]]
            st = diff*24 + sh
            edk = diff*24 + eh
            comps.append((vid, st, edk, cap, t["slot"]))
    return comps

class Place:
    def __init__(self, data: Dict):
        self.id = data["venue_id"]
        self.name = data["venue_name"]
        self.capacity = data["capacity"]
        self.agenda = []
        self.endtime = -1
        self.audience_from = []

    def infect(self, paras):
        if len(self.audience_from) == 0:
            return
        seijrd_sum = np.sum(self.audience_from, axis=0)  # [S,E,I,J,R,D]
        infectious_num = seijrd_sum[1] + seijrd_sum[2] + seijrd_sum[3]
        if infectious_num <= 0:
            return
        prob = 1 - (1 - paras["rho"])**infectious_num
        prob = np.clip(prob, 0, 1)
        S_tot = seijrd_sum[0]
        if S_tot < 1e-9:
            return
        new_inf = S_tot * prob
        frac_sus = self.audience_from[:,0] / (S_tot + 1e-12)
        new_each = frac_sus * new_inf
        self.audience_from[:,0] -= new_each
        self.audience_from[:,1] += new_each

def process_places(data_file):
    places = {}
    for d in data_file:
        vid = d["venue_id"]
        places[vid] = Place({
            "venue_id": vid,
            "venue_name": d["venue_name"],
            "capacity": int(d["capacity"].replace(',',''))
        })
    return places

# ===== SEIJRD 节点与网络（双β + 可传入隔离强度对 β2 联动） =====
class SEIR_Node:
    def __init__(self, state, total_nodes):
        self.state = state.astype(np.float64)
        self.total_nodes = total_nodes
        self.from_node = np.zeros((total_nodes, 6))
        self.to_node   = np.zeros((total_nodes, 5))

    def update_seir(self, delta_time=1/2400, times=100, paras=PARAS, beta2_scale=1.0):
        S,E,I,J,R,D = self.state
        N = S+E+I+J+R
        if N <= 0:
            return
        beta1 = paras["beta_1"]
        beta2 = paras["beta_2"] * beta2_scale

        dS = - beta1 * S * E / N - beta2 * S * (I + J) / N
        dE =   beta1 * S * E / N + beta2 * S * (I + J) / N - paras["sigma_i"]*E - paras["sigma_j"]*E
        dI =   paras["sigma_i"]*E - paras["gamma_i"]*I - paras["mu_i"]*I
        dJ =   paras["sigma_j"]*E - paras["gamma_j"]*J - paras["mu_j"]*J
        dR =   paras["gamma_i"]*I + paras["gamma_j"]*J
        dD =   paras["mu_i"]*I + paras["mu_j"]*J

        for _ in range(times):
            self.state += np.array([dS,dE,dI,dJ,dR,dD]) * delta_time

        self.state = np.clip(self.state, 0, MAX_POP)
        # 同步 from_node（用于回家流程中质量守恒）
        Nval = np.sum(self.from_node, axis=1)
        if N > 0:
            ratio = (Nval / N)[:,None]
        else:
            ratio = np.zeros((self.total_nodes,1))
        self.from_node += np.array([dS,dE,dI,dJ,dR,dD])[None,:] * ratio * delta_time * times

class SEIR_Network:
    def __init__(self, num, states, matrix=None, paras=PARAS):
        self.node_num = num
        self.nodes = {i: SEIR_Node(states[i], num) for i in range(num)}
        if matrix is not None:
            rs = np.sum(matrix, axis=1, keepdims=True)
            self.A = matrix / (rs + 1e-12)
        else:
            A = np.random.rand(num, num)
            self.A = A / A.sum(axis=1, keepdims=True)
        self.paras = paras
        self.delta_time = 1/2400

    def morning_commuting(self):
        for i in range(self.node_num):
            SEIJR = self.nodes[i].state[:5]
            for j in range(self.node_num):
                self.nodes[i].to_node[j] = SEIJR * self.A[i][j]
        for i in range(self.node_num):
            self.nodes[i].state[:5] = 0.0
        for i in range(self.node_num):
            for j in range(self.node_num):
                self.nodes[j].state[:5] += self.nodes[i].to_node[j]
                self.nodes[j].from_node[i,:5] = self.nodes[i].to_node[j]

    def evening_commuting(self):
        for i in range(self.node_num):
            for j in range(self.node_num):
                if i == j: continue
                self.nodes[j].state -= self.nodes[j].from_node[i]
                self.nodes[i].state += self.nodes[j].from_node[i]
        for i in range(self.node_num):
            self.nodes[i].from_node = np.zeros((self.node_num,6))
            self.nodes[i].to_node   = np.zeros((self.node_num,5))

    def update_network(self, beta2_scale=1.0):
        for i in range(self.node_num):
            self.nodes[i].update_seir(self.delta_time, 100, self.paras, beta2_scale)

# ===== Env：加载人口/矩阵/赛程，保持与 Task‑2 一致风格 =====
class Env:
    def __init__(self, is_states=False):
        self.is_states = is_states
        self.num = 23
        self.network = None
        self.competitions = []
        self.places = {}
        self.matrix = None
        self.origin_matrix = None
        self.current_tick = 0
        self.paras = PARAS.copy()

    def init_env(self):
        data_json = json.load(open(DATA_JSON, "r", encoding="utf-8"))
        self.competitions = process_competitions(data_json)
        self.places = process_places(data_json)
        for c in self.competitions:
            self.places[c[0]].agenda.append(c)
        for p in self.places.values():
            p.agenda.sort(key=lambda x: x[1])

        if self.is_states:
            init_states = Input_Initial("data/SEIJRD.csv")
        else:
            pops = Input_Population("data/tokyo_population.csv")
            init_states = np.zeros((self.num,6))
            for i in range(self.num):
                S_ = pops[i] - self.paras["initial_infect"]*4.5/self.num
                E_ = self.paras["initial_infect"]*2/self.num
                I_ = self.paras["initial_infect"]*1/self.num
                J_ = self.paras["initial_infect"]*1.5/self.num
                init_states[i] = np.array([S_,E_,I_,J_,0,0], dtype=float)

        try:
            M = Input_Matrix("data/tokyo_commuting_flows_with_intra.csv")
            rs = M.sum(axis=1, keepdims=True)
            M = M / (rs + 1e-12)
            self.origin_matrix = M.copy()
            self.matrix = M.copy()
        except:
            self.origin_matrix = np.eye(self.num)
            self.matrix = np.eye(self.num)

        self.network = SEIR_Network(self.num, init_states, self.matrix, self.paras)
        self.current_tick = 0

    def check_competition_start(self):
        for pid, place in self.places.items():
            if not place.agenda: continue
            comp = place.agenda[0]
            if comp[1] == self.current_tick:
                place.audience_from = np.zeros((self.num,6))
                place.agenda.pop(0)
                place.endtime = comp[2]
                capacity = comp[3]

                sums = [np.sum(self.network.nodes[i].state[:4]) for i in range(self.num)]
                sums = np.array(sums)
                tot = np.sum(sums);
                if tot < 1e-9: tot = 1.0
                for i in range(self.num):
                    portion = self.network.nodes[i].state[:5] * (capacity / tot)
                    portion = np.minimum(portion, self.network.nodes[i].state[:5])
                    place.audience_from[i,:5] = portion
                    self.network.nodes[i].state[:5] -= portion
                place.infect(self.paras)

    def check_competition_end(self):
        for pid, place in self.places.items():
            if place.endtime == self.current_tick:
                for i in range(self.num):
                    self.network.nodes[i].state[:5] += place.audience_from[i,:5]
                place.audience_from = []
                place.endtime = -1

# ====== Task-3 强化学习环境：a_i 日决策（隔离 I），β2 与 mean(a) 联动 ======
MIN_SELF_ACTIVITY = 0.3  # 仅用于通勤代理可视化（与 Task-2 对齐字段名）

class IsolationEnv(gym.Env):
    """
    Step = 1 小时；但仅在每天开始的时刻（hour==0）读取新的 action 向量 a。
    在每天结束（hour==23 -> 0）应用 I <- (1-a) I，并计算当日奖励：
      reward_day = + sum_i (a_i * I_i_before) - cost_coef * sum_i (a_i * I_i_before) [+ commute_term 可选]
    其中 commute_ratio 代理 = 1 - mean(a)
    """
    metadata = {"render.modes": []}

    def __init__(self, days=20, is_states=False,
                 a_cost_coef=1.0, beta2_k=BETA2_K, commute_coef=0.0):
        super().__init__()
        self.days = days
        self.is_states = is_states
        self.a_cost_coef = a_cost_coef
        self.beta2_k = beta2_k
        self.commute_coef = commute_coef

        self.env = Env(is_states=is_states)
        self.env.init_env()
        self.num = self.env.num

        # 观测：23*6 + day + hour
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num*6+2,), dtype=np.float32)
        # 动作：23个区的隔离比例 a_i
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.num,), dtype=np.float32)

        self.current_hour = 0
        self.current_day = 0
        self.episode_steps = 0
        self.max_steps = self.days * 24
        self._today_a = np.zeros(self.num, dtype=np.float32)  # 当日生效的隔离比例

        self.np_random = None

    def seed(self, seed=None):
        self.np_random, seed_ = seeding.np_random(seed)
        return [seed_]

    def reset(self):
        self.env = Env(is_states=self.is_states)
        self.env.init_env()
        self.current_hour = 0
        self.current_day = 0
        self.episode_steps = 0
        self._today_a = np.zeros(self.num, dtype=np.float32)
        return self._get_obs()

    def _get_obs(self):
        arr = []
        for i in range(self.num):
            arr.append(self.env.network.nodes[i].state)
        arr = np.concatenate(arr, axis=0)
        return np.concatenate([arr, [self.current_day, self.current_hour]]).astype(np.float32)

    def step(self, action):
        # 每天开始时刻接收新的 a
        if self.current_hour == 0:
            a = np.clip(action, 0.0, 1.0).astype(np.float32)
            self._today_a = a
        else:
            a = self._today_a

        # --- 比赛与通勤 ---
        self.env.check_competition_start()
        if self.current_hour == 8:
            self.env.network.morning_commuting()

        prev_states = np.array([nd.state.copy() for nd in self.env.network.nodes.values()])

        # β2 与 mean(a) 联动
        beta2_scale = 1.0 - self.beta2_k * float(np.mean(a))
        beta2_scale = max(0.0, beta2_scale)
        self.env.network.update_network(beta2_scale=beta2_scale)

        if self.current_hour == 18:
            self.env.network.evening_commuting()
        self.env.check_competition_end()

        # --- 日终隔离 与 奖励 ---
        reward = 0.0
        if self.current_hour == 23:
            # 在隔离前统计 I（作为隔离收益的基数）
            I_before = np.array([self.env.network.nodes[i].state[2] for i in range(self.num)], dtype=np.float64)
            removed_I = a * I_before

            # 应用隔离 I <- (1-a) I
            for i in range(self.num):
                self.env.network.nodes[i].state[2] = max(0.0, self.env.network.nodes[i].state[2] - removed_I[i])

            # 通勤代理：1 - mean(a)
            commute_ratio = 1.0 - float(np.mean(a))

            reward = float(np.sum(removed_I) - self.a_cost_coef * np.sum(removed_I) + self.commute_coef * commute_ratio)

        # --- 指标信息（按小时统计） ---
        curr_states = np.array([nd.state for nd in self.env.network.nodes.values()])
        prev_infected = np.sum(prev_states[:,1:4])
        curr_infected = np.sum(curr_states[:,1:4])
        hour_infections = max(0.0, curr_infected - prev_infected)

        prev_death = np.sum(prev_states[:,5])
        curr_death = np.sum(curr_states[:,5])
        hour_deaths = max(0.0, curr_death - prev_death)

        info = {
            "infections": float(hour_infections),
            "deaths": float(hour_deaths),
            "commute_ratio": float(1.0 - float(np.mean(a)))  # 仅作可视化代理
        }

        # --- 时间推进 ---
        self.current_hour += 1
        if self.current_hour >= 24:
            self.current_hour = 0
            self.current_day += 1
        self.episode_steps += 1

        done = (self.episode_steps >= self.max_steps)
        obs = self._get_obs()
        return obs, reward, done, info

# ======== 训练与评估（与 Task‑2 风格一致） ========
class ProgressCallback(BaseCallback):
    def __init__(self, exp_id=0, algo="", log_interval=2000, verbose=1):
        super().__init__(verbose)
        self.exp_id=exp_id
        self.algo=algo
        self.log_interval=log_interval
        self.start_time=None
        self.last_save=None

    def _on_training_start(self):
        self.start_time=time.time()
        self.last_save=self.start_time
        print(f"[Exp {self.exp_id}] start {self.algo} training...")

    def _on_step(self):
        if self.n_calls % self.log_interval == 0:
            steps=self.model.num_timesteps
            total=self.model._total_timesteps
            pct=steps/total*100
            el=time.time()-self.start_time
            eta=el/(steps+1e-9)*(total-steps)
            print(f"[Exp {self.exp_id}] {self.algo} {steps}/{total}({pct:.1f}%), elapsed={el/60:.1f}m, ETA={eta/60:.1f}m")
            if time.time()-self.last_save > 600:
                ckpt=f"./models/{self.algo}_exp{self.exp_id}_step{steps}.zip"
                self.model.save(ckpt)
                self.last_save=time.time()
                print(f"[Exp {self.exp_id}] checkpoint => {ckpt}")
        return True

def make_env(n_envs=1, seeds=[0], env_kwargs=None):
    def env_fn(seed):
        def _thunk():
            env = IsolationEnv(**(env_kwargs or {}))
            env.seed(seed)
            return env
        return _thunk
    if n_envs==1:
        return DummyVecEnv([env_fn(seeds[0])])
    else:
        return SubprocVecEnv([env_fn(s) for s in seeds])

def train_one(algo, exp_id, gpu_id, total_timesteps=1_000_000, env_kwargs=None, n_envs=1):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    seed = 10000 + exp_id
    set_random_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    vec_env = make_env(n_envs=n_envs, seeds=[seed+i for i in range(n_envs)], env_kwargs=env_kwargs)
    cb = ProgressCallback(exp_id=exp_id, algo=algo, log_interval=2000)
    ckpt_cb = CheckpointCallback(save_freq=max(1, 50000//n_envs),
                                 save_path=f"./models/{algo}_exp{exp_id}_ckpts/",
                                 name_prefix="model")

    if algo == "PPO":
        model = PPO("MlpPolicy", vec_env, device="cuda", verbose=0,
                    tensorboard_log=f"./tensorboard/{algo}_exp{exp_id}",
                    learning_rate=3e-4, n_steps=2048, batch_size=128,
                    n_epochs=10, gamma=0.99, gae_lambda=0.95,
                    clip_range=0.2, ent_coef=0.01,
                    policy_kwargs=dict(net_arch=[dict(pi=[256,256,128], vf=[256,256,128])]))
    elif algo == "A2C":
        model = A2C("MlpPolicy", vec_env, device="cuda", verbose=0,
                    tensorboard_log=f"./tensorboard/{algo}_exp{exp_id}",
                    learning_rate=7e-4, n_steps=5, gamma=0.99,
                    ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5,
                    policy_kwargs=dict(net_arch=[dict(pi=[256,256,128], vf=[256,256,128])]))
    else:
        raise ValueError("Unsupported algo")

    model.learn(total_timesteps=total_timesteps, callback=[cb, ckpt_cb], progress_bar=True)
    final_path = f"./models/{algo}_exp{exp_id}_final.zip"
    model.save(final_path)
    vec_env.close()
    print(f"[Exp {exp_id}] training done => {final_path}")
    return final_path

# ===== Task-3 基线（常数隔离） =====
BASELINE_TASK3 = {
    "constant_80": lambda n: np.ones(n)*0.8,
    "constant_50": lambda n: np.ones(n)*0.5,
    "constant_30": lambda n: np.ones(n)*0.3,
    "no_isolation": lambda n: np.zeros(n)
}

def evaluate_baseline(bname, days=20, a_cost_coef=1.0, beta2_k=BETA2_K, commute_coef=0.0, is_states=False):
    out_dir = f"./results/baseline_{bname}/"
    os.makedirs(out_dir, exist_ok=True)
    env = IsolationEnv(days=days, is_states=is_states, a_cost_coef=a_cost_coef, beta2_k=beta2_k, commute_coef=commute_coef)
    obs = env.reset()

    a_vec = BASELINE_TASK3[bname](env.num)
    done=False
    ep = {
        "hourly_infections":[], "hourly_deaths":[],
        "hourly_commute_ratio":[], "hourly_reward":[],
        "isolation_series":[]
    }
    step=0
    while not done:
        action = a_vec  # 每小时都传，但只有 hour==0 时生效
        obs, rew, done, info = env.step(action)
        ep["hourly_infections"].append(float(info["infections"]))
        ep["hourly_deaths"].append(float(info["deaths"]))
        ep["hourly_commute_ratio"].append(float(info["commute_ratio"]))
        ep["hourly_reward"].append(float(rew))
        if step % 24 == 0:
            ep["isolation_series"].append(a_vec.tolist())
        step += 1

    results = {"method": f"baseline_{bname}", "episodes":[ep]}
    with open(os.path.join(out_dir,"baseline_evaluation.json"), "w") as f:
        json.dump(results, f, indent=2)
    # 基线简单时序图
    _plot_last_episode_time_series(ep, os.path.join(out_dir,"baseline_timeseries.png"))
    print(f"Baseline {bname} evaluate done => {out_dir}")
    return results

# ===== RL 评估（生成每实验自己的对比图 + 时序图） =====
# ==== REPLACE evaluate_model in main_RL_task3_910.py ====
def evaluate_model(cfg,
                   episodes: int = 3,
                   device: str = "cpu"):
    """
    评估单个实验配置：
      - 始终在 CPU 上推理（device='cpu'），避免 GPU OOM
      - 若外部不慎要求 CUDA，检测到 OOM 自动回退 CPU
    必要键：cfg['algo'] in {'PPO','A2C'}, cfg['exp_id'](int)
    可选键：cfg['model_path']、cfg['days']、cfg['is_states']...
    """
    import os, gc, json
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from stable_baselines3 import PPO, A2C
    import torch

    algo = cfg.get('algo', 'A2C')
    exp_id = cfg.get('exp_id', 0)
    days = int(cfg.get('days', 20))
    is_states = bool(cfg.get('is_states', False))
    model_path = cfg.get('model_path',
                         f"./models/{algo}_exp{exp_id}_final.zip")

    # ---- 关键：强制 CPU 评估 ----
    load_cls = PPO if algo == "PPO" else A2C
    try:
        model = load_cls.load(model_path, device=device)  # 默认 'cpu'
    except RuntimeError as e:
        # 如果有外部错误指定为 cuda，兜底
        if "CUDA" in str(e).upper():
            print(f"[Exp {exp_id}] CUDA OOM detected. Falling back to CPU...")
            model = load_cls.load(model_path, device="cpu")
            device = "cpu"
        else:
            raise

    results = {
        'method': f"{algo}_exp{exp_id}",
        'episodes': []
    }

    # 环境构造函数名/参数按你 task3 的实现来；以下保持常用写法：
    from main_RL_task3_910 import CommuneMatrixEnv  # 如果放同文件，请删掉这行
    for ep in range(episodes):
        env = CommuneMatrixEnv(days=days, is_states=is_states)
        obs = env.reset()
        done = False

        ep_data = {
            'hourly_infections': [],
            'hourly_deaths': [],
            'hourly_commute_ratio': [],
            'hourly_reward': [],
        }
        ep_reward = 0.0

        with torch.no_grad():
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                ep_reward += float(reward)

                # 记录
                ep_data['hourly_infections'].append(float(info.get('infections', 0.0)))
                ep_data['hourly_deaths'].append(float(info.get('deaths', 0.0)))
                mat = env.seir_env.matrix
                orig_c = np.sum(env.original_matrix) - np.trace(env.original_matrix)
                now_c = np.sum(mat) - np.trace(mat)
                c_ratio = now_c / (orig_c + 1e-12)
                ep_data['hourly_commute_ratio'].append(float(c_ratio))
                ep_data['hourly_reward'].append(float(reward))

        ep_data['total_reward'] = float(ep_reward)
        results['episodes'].append(ep_data)

    # —— 保存 JSON + XLSX + 单集时序图（与现有流程一致）——
    out_dir = f"./results/{algo}_exp{exp_id}/"
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "evaluation.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    xlsx_path = os.path.join(out_dir, "evaluation.xlsx")
    with pd.ExcelWriter(xlsx_path) as writer:
        for i, ep in enumerate(results['episodes'], start=1):
            df = pd.DataFrame({
                "hourly_infections": ep['hourly_infections'],
                "hourly_deaths": ep['hourly_deaths'],
                "hourly_commute_ratio": ep['hourly_commute_ratio'],
                "hourly_reward": ep['hourly_reward']
            })
            df.to_excel(writer, sheet_name=f"episode_{i}", index=False)

    last_ep = results['episodes'][-1]
    t = np.arange(len(last_ep['hourly_infections']))
    plt.figure(figsize=(12, 8))
    plt.subplot(4,1,1); plt.plot(t, last_ep['hourly_infections']); plt.title("hourly_infections"); plt.grid(alpha=.3)
    plt.subplot(4,1,2); plt.plot(t, last_ep['hourly_deaths']); plt.title("hourly_deaths"); plt.grid(alpha=.3)
    plt.subplot(4,1,3); plt.plot(t, last_ep['hourly_commute_ratio']); plt.title("hourly_commute_ratio"); plt.grid(alpha=.3)
    plt.subplot(4,1,4); plt.plot(t, last_ep['hourly_reward']); plt.title("hourly_reward"); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "evaluation_timeseries.png")); plt.close()

    # 资源清理
    del model; gc.collect()
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"[Exp {exp_id}] CPU evaluation done → {json_path}")
    return results


def _plot_last_episode_time_series(last_ep: Dict, png_path: str):
    x = np.arange(len(last_ep["hourly_infections"]))
    cum_inf  = np.cumsum(last_ep["hourly_infections"])
    cum_death= np.cumsum(last_ep["hourly_deaths"])
    avg_comm = pd.Series(last_ep["hourly_commute_ratio"]).expanding().mean().values
    cum_rew  = np.cumsum(last_ep["hourly_reward"])

    plt.figure(figsize=(12,8))
    plt.subplot(4,1,1); plt.plot(x, cum_inf);  plt.title("Cumulative Infections"); plt.grid(alpha=.3)
    plt.subplot(4,1,2); plt.plot(x, cum_death);plt.title("Cumulative Deaths");     plt.grid(alpha=.3)
    plt.subplot(4,1,3); plt.plot(x, avg_comm); plt.title("Average Commute Ratio");  plt.grid(alpha=.3)
    plt.subplot(4,1,4); plt.plot(x, cum_rew);  plt.title("Cumulative Reward");      plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(png_path); plt.close()

def _draw_strategy_comparison_single_experiment(rl_last_ep: Dict, out_path: str):
    # 读取所有 Task-3 baselines 的 last episode
    base_list = []
    for b in BASELINE_TASK3.keys():
        bpath = f"./results/baseline_{b}/baseline_evaluation.json"
        if os.path.exists(bpath):
            with open(bpath,"r") as f:
                data = json.load(f)
            if data["episodes"]:
                base_list.append((b, data["episodes"][-1]))

    def _series(ep):
        return (
            np.cumsum(ep["hourly_infections"]),
            np.cumsum(ep["hourly_deaths"]),
            pd.Series(ep["hourly_commute_ratio"]).expanding().mean().values,
            np.cumsum(ep["hourly_reward"])
        )

    plt.figure(figsize=(16,18))
    titles = ["Cumulative Infections","Cumulative Deaths","Average Commute Ratio","Cumulative Reward"]
    ep_metrics = _series(rl_last_ep)

    for i in range(4):
        plt.subplot(4,1,i+1)
        plt.plot(ep_metrics[i], label="RL")
        for bname, bep in base_list:
            bmet = _series(bep)
            plt.plot(bmet[i], label=f"baseline_{bname}", linestyle="--", linewidth=2)
        plt.title(titles[i]); plt.grid(alpha=.3); plt.legend()
    plt.tight_layout(); plt.savefig(out_path); plt.close()

# ====== 并行调度 & 主入口 ======
def parallel_train(configs, gpu_list):
    import multiprocessing
    procs=[]
    for cfg in configs:
        eid = cfg["exp_id"]; algo = cfg["algo"]
        gpu = gpu_list[eid % max(1,len(gpu_list))]
        p = multiprocessing.Process(target=train_one,
            args=(algo, eid, gpu, cfg.get("total_timesteps",1_000_000),
                  {"days":cfg.get("days",20),
                   "is_states":cfg.get("is_states",False),
                   "a_cost_coef":cfg.get("a_cost_coef",1.0),
                   "beta2_k":cfg.get("beta2_k",BETA2_K),
                   "commute_coef":cfg.get("commute_coef",0.0)},
                  cfg.get("n_envs",1)))
        p.start(); procs.append(p); time.sleep(2)
    for p in procs: p.join()
    print("All training done.")

# ==== REPLACE parallel_evaluate in main_RL_task3_910.py ====
def parallel_evaluate(cfgs,
                      device: str = "cpu",
                      max_procs: int = None,
                      episodes: int = 3):
    """
    评估多个实验：
      - 默认 device='cpu'：安全并行
      - 若 device='cuda'：为防 OOM，改为顺序评估
    """
    import multiprocessing as mp
    import os, torch

    # 评估使用 CPU 更安全：每进程不初始化 CUDA 上下文
    if device.lower() == "cpu":
        if max_procs is None:
            max_procs = min(mp.cpu_count(), max(1, len(cfgs)))
        try:
            torch.set_num_threads(1)
        except Exception:
            pass

        tasks = []
        for cfg in cfgs:
            c = dict(cfg)  # 拷贝，防止原地改
            tasks.append((c, episodes, "cpu"))

        with mp.Pool(processes=max_procs, maxtasksperchild=1) as pool:
            pool.starmap(lambda c, e, dev: evaluate_model(c, episodes=e, device=dev), tasks)

    else:
        # GPU：顺序评估，避免并发占满同一块显存
        for cfg in cfgs:
            evaluate_model(cfg, episodes=episodes, device="cuda")


def eval_all_baselines(days=20, a_cost_coef=1.0, beta2_k=BETA2_K, commute_coef=0.0, is_states=False):
    for b in BASELINE_TASK3.keys():
        evaluate_baseline(b, days=days, a_cost_coef=a_cost_coef, beta2_k=beta2_k, commute_coef=commute_coef, is_states=is_states)

def build_configs(algos, exp_start, exp_count, days, seed=0):
    cfgs=[]
    eid_list = list(range(exp_start, exp_start+exp_count))
    gpu_list = list(range(NUM_GPUS)) if NUM_GPUS>0 else [0]
    for algo in algos:
        for eid in eid_list:
            cfgs.append({
                "algo":algo, "exp_id":eid, "days":days, "seed": seed+eid,
                "n_envs":1, "total_timesteps":1_000_000,
                "a_cost_coef":1.0, "beta2_k":BETA2_K, "commute_coef":0.0,
                "is_states":False
            })
    return cfgs, gpu_list

def main():
    parser = argparse.ArgumentParser(description="Task-3 Isolation + double-beta RL")
    parser.add_argument("--mode", type=str, choices=["train","evaluate","baseline","visualize","all"], default="all")
    parser.add_argument("--algos", nargs="+", default=["PPO","A2C"])
    parser.add_argument("--exp-start", type=int, default=1)
    parser.add_argument("--exp-count", type=int, default=16)
    parser.add_argument("--days", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-device", type=str, choices=["cpu", "cuda"], default="cpu",
                        help="Device for evaluation (default: cpu)")
    parser.add_argument("--eval-procs", type=int, default=None,
                        help="Max processes for CPU evaluation (default: min(cpu_count, N))")

    args = parser.parse_args()

    cfgs, gpu_list = build_configs(args.algos, args.exp_start, args.exp_count, args.days, args.seed)

    if args.mode in ("train","all"):
        parallel_train(cfgs, gpu_list)

    if args.mode in ("evaluate","all"):
        parallel_evaluate(cfgs, device=args.eval_device, max_procs=args.eval_procs, episodes=3)

    if args.mode in ("baseline","all"):
        eval_all_baselines(days=args.days)

    if args.mode in ("visualize","all"):
        from advanced_visuals import run_all_visualization_improvements
        run_all_visualization_improvements("./results")

if __name__ == "__main__":
    main()
