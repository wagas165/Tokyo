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

matplotlib.use("Agg")  # Use non-interactive backend
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
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

# Ensure output directories exist
os.makedirs("./models", exist_ok=True)
os.makedirs("./logs", exist_ok=True)
os.makedirs("./results", exist_ok=True)
os.makedirs("./visualizations", exist_ok=True)
os.makedirs("./tensorboard", exist_ok=True)

# Global configuration
NUM_GPUS = 8
NUM_CPU_CORES = 140
MAX_POP = 1e12

# ======== 参数：对齐老版的 SEIJRD 模型参数 ========
SEIJRD_PARAMS = {
    "beta": 0.155,  # 传染率
    "sigma_i": 0.0299,  # E -> I 转换率
    "sigma_j": 0.0156,  # E -> J 转换率
    "gamma_i": 0.079,  # I -> R 康复率
    "gamma_j": 0.031,  # J -> R 康复率
    "mu_i": 1.95e-5,  # I -> D 死亡率
    "mu_j": 0.00025,  # J -> D 死亡率
    "rho": 1e-4  # 在场馆内的接触传染概率基数
}

MIN_SELF_ACTIVITY = 0.3  # Minimum proportion of population that stays in their home area
FILE_PATH = "data/end_copy.json"  # Competition data file


# ====================== DATA PROCESSING ======================
def process_competitions(data_file):
    """Process competition data from JSON file (保持不变，可与老版对应)"""
    slot_mapping = {0: (8, 11), 1: (13, 17), 2: (19, 22)}
    year = 2021
    earliest_date = None

    for venue in data_file:
        for t in venue["time"]:
            event_date = datetime.date(year, t["month"], t["day"])
            if earliest_date is None or event_date < earliest_date:
                earliest_date = event_date

    competitions = []
    for venue in data_file:
        vid = venue["venue_id"]
        capacity = int(venue["capacity"].replace(',', ''))
        for t in venue["time"]:
            event_date = datetime.date(year, t["month"], t["day"])
            days_diff = (event_date - earliest_date).days
            slot = t["slot"]
            start_hour, end_hour = slot_mapping[slot]
            start_tick = days_diff * 24 + start_hour
            end_tick = days_diff * 24 + end_hour
            competitions.append((vid, start_tick, end_tick, capacity, slot))

    return competitions


def process_places(data_file):
    """Process venue data from JSON file"""
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


# ====================== SEIJRD MODEL CLASSES ======================
class Place:
    """Venue class for SEIJRD model"""

    def __init__(self, data: Dict):
        self.id = data["venue_id"]
        self.name = data["venue_name"]
        self.capacity = data["capacity"]
        self.agenda = []
        self.endtime = -1
        self.audience_from = []

    def infect(self, paras=SEIJRD_PARAMS):
        """
        修正Bug1：按照老代码做法，将 E + I + J 都视为可传染人群。
        """
        if len(self.audience_from) == 0:
            return

        seijrd_sum = np.sum(self.audience_from, axis=0)  # [S, E, I, J, R, D]
        # E + I + J 皆视为具有传染力
        infectious_num = seijrd_sum[1] + seijrd_sum[2] + seijrd_sum[3]
        if infectious_num <= 0:
            return

        # 计算新感染概率
        try:
            prob = 1 - (1 - paras["rho"]) ** infectious_num
        except OverflowError:
            prob = 1.0
        prob = np.clip(prob, 0, 1)

        # 计算总可被感染的易感者
        S_tot = seijrd_sum[0]
        if S_tot < 1e-9:
            return
        new_inf = S_tot * prob

        # 按各节点易感者占比，分配新感染
        frac_sus = self.audience_from[:, 0] / (S_tot + 1e-12)
        new_inf_each = frac_sus * new_inf

        self.audience_from[:, 0] -= new_inf_each
        self.audience_from[:, 1] += new_inf_each


class SEIJRD_Node:
    """Node representing a geographic area with SEIJRD dynamics"""

    def __init__(self, state, total_nodes):
        self.state = state.astype(np.float64)
        self.from_node = np.zeros((total_nodes, 6))
        self.total_nodes = total_nodes

    def update_seijrd(self, dt=1 / 2400, times=100, paras=SEIJRD_PARAMS):
        """
        修正Bug2：在这里使用老代码 ODE 形式，循环迭代 times 次，每次 dt=1/2400
        """
        for _ in range(times):
            S, E, I, J, R, D_ = self.state
            N = S + E + I + J + R  # D是否包含看需要，此处跟老代码保持一致

            if N < 1e-9:
                continue

            beta = paras["beta"]
            sigma_i = paras["sigma_i"]
            sigma_j = paras["sigma_j"]
            gamma_i = paras["gamma_i"]
            gamma_j = paras["gamma_j"]
            mu_i = paras["mu_i"]
            mu_j = paras["mu_j"]

            # 老代码中的 ODE (SEIJRD_Model_Modify_Matrix.py 类似)
            dS = - beta * S * E / N
            dE = beta * S * E / N - sigma_i * E - sigma_j * E
            dI = sigma_i * E - gamma_i * I - mu_i * I
            dJ = sigma_j * E - gamma_j * J - mu_j * J
            dR = gamma_i * I + gamma_j * J
            dD = mu_i * I + mu_j * J

            self.state[0] += dS * dt
            self.state[1] += dE * dt
            self.state[2] += dI * dt
            self.state[3] += dJ * dt
            self.state[4] += dR * dt
            self.state[5] += dD * dt

        self.state = np.clip(self.state, 0, MAX_POP)


class SEIJRD_Network:
    """Network of SEIJRD nodes with commuting dynamics"""

    def __init__(self, num_nodes, states):
        self.node_num = num_nodes
        self.nodes = {}
        for i in range(num_nodes):
            self.nodes[i] = SEIJRD_Node(states[i], num_nodes)

        # 初始化通勤矩阵：可从外部 env.modify_matrix() 来覆盖
        self.A = np.eye(num_nodes)
        self.delta_time = 1 / 2400

    def morning_commuting(self):
        """Simulate morning commuting between nodes"""
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
        """Simulate evening commuting between nodes"""
        for i in range(self.node_num):
            for j in range(self.node_num):
                if i == j:
                    continue
                self.nodes[j].state += self.nodes[i].from_node[j]
                self.nodes[j].state = np.clip(self.nodes[j].state, 0, MAX_POP)
            self.nodes[i].from_node = np.zeros((self.node_num, 6))

    def update_network(self):
        """Update SEIJRD dynamics for all nodes"""
        for i in range(self.node_num):
            self.nodes[i].update_seijrd(dt=1 / 2400, times=100, paras=SEIJRD_PARAMS)


class Env:
    """Base environment for SEIJRD simulation"""

    def __init__(self):
        self.places = {}
        self.competitions = []
        self.network = None
        self.capacity_strategy = None
        self.current_tick = 0
        self.origin_matrix = None
        self.matrix = None
        self.num = 0
        self.states_history = []

    def init_env(self):
        """Initialize environment with competition data and SEIJRD states"""
        try:
            data_json = json.load(open(FILE_PATH, 'r', encoding='utf-8'))
            self.competitions = process_competitions(data_json)
            self.places = process_places(data_json)

            for comp in self.competitions:
                vid = comp[0]
                self.places[vid].agenda.append(comp)
            for p in self.places.values():
                p.agenda.sort(key=lambda x: x[1])

            # 初始化 23 区的 SEIJRD 状态 [S, E, I, J, R, D]
            # 你也可根据实际初值或 CSV 调整
            self.num = 23
            init_states = []
            for i in range(self.num):
                # 示例：与老代码类似，可自行修改
                init_states.append(np.array([92200, 800, 0, 0, 0, 0]))

            init_states = np.array(init_states)
            self.network = SEIJRD_Network(self.num, init_states)

            # 加载或生成通勤矩阵
            try:
                matrix = self.load_matrix()
                self.origin_matrix = matrix.copy()
                self.matrix = matrix.copy()
                self.network.A = matrix.copy()
            except:
                print("Warning: Could not load matrix. Using identity matrix.")
                self.origin_matrix = np.eye(self.num)
                self.matrix = np.eye(self.num)
                self.network.A = self.matrix.copy()

            self.current_tick = 0
            self.states_history = []

        except Exception as e:
            print(f"Error initializing environment: {e}")
            raise

    def load_matrix(self):
        """Load commuting matrix from file (与老代码保持相同路径)"""
        df = pd.read_csv("data/tokyo_commuting_flows_with_intra.csv", index_col=0)
        df = df.apply(pd.to_numeric, errors='coerce')
        matrix = df.values

        # Normalize rows to sum to 1
        matrix = matrix / np.sum(matrix, axis=1, keepdims=True)
        return matrix

    def check_competition_start(self):
        """Check if any competitions start at current tick"""
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

                N_list = [np.sum(self.network.nodes[i].state[:5]) for i in range(self.network.node_num)]
                sumN = np.sum(N_list) if np.sum(N_list) > 0 else 1.0
                for i in range(self.network.node_num):
                    frac = (self.network.nodes[i].state[:5] / sumN) * actual_cap
                    frac = np.minimum(frac, self.network.nodes[i].state[:5])
                    moved_6 = np.zeros(6)
                    moved_6[:5] = frac
                    self.network.nodes[i].state[:5] -= frac
                    place.audience_from[i] = moved_6

                # 在进入场馆后，立刻做一次 infect（可选，看老代码的顺序）
                place.infect(SEIJRD_PARAMS)

    def check_competition_end(self):
        """Check if any competitions end at current tick"""
        for pl in self.places.values():
            if pl.endtime == self.current_tick:
                for i in range(self.network.node_num):
                    self.network.nodes[i].state[:5] += pl.audience_from[i][:5]
                pl.audience_from = []
                pl.endtime = -1

    def modify_matrix(self, matrix):
        """Modify the commuting matrix"""
        self.network.A = matrix.copy()
        self.matrix = matrix.copy()

        # 同老代码：根据通勤矩阵修改后的总流动比，动态调整 beta
        ratio = (np.sum(self.matrix) - np.trace(self.matrix)) / (
                    np.sum(self.origin_matrix) - np.trace(self.origin_matrix))
        # 当 ratio < 0.XX 时，可强行剪切到合理范围，也可直接套用老代码
        # 这里只是示例做法
        new_beta = 0.05 + (0.155 - 0.05) * ratio
        SEIJRD_PARAMS['beta'] = max(new_beta, 0.01)  # 保底


class CommuneMatrixEnv(gym.Env):
    """Reinforcement Learning environment for optimizing commuting matrix (RL2)"""

    def __init__(self, days=20, monitor=False, logdir=None):
        super().__init__()
        self.days = days
        self.seijrd_env = Env()
        self.seijrd_env.init_env()
        self.n = self.seijrd_env.num

        # Observation space: SEIJRD states + day + hour
        obs_dim = 6 * self.n + 2
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # Action space: complete matrix (flatten 23×23 = 529)
        self.action_space = spaces.Box(
            low=0, high=1, shape=(self.n * self.n,), dtype=np.float32)

        self.original_matrix = self.seijrd_env.origin_matrix.copy()
        self.current_day = 0
        self.np_random = None
        self.seed_value = None

    def seed(self, seed=None):
        """Set random seed"""
        self.np_random, seed_ = seeding.np_random(seed)
        self.seed_value = seed_
        return [seed_]

    def reset(self):
        """Reset environment to initial state"""
        self.seijrd_env = Env()
        self.seijrd_env.init_env()
        self.original_matrix = self.seijrd_env.origin_matrix.copy()
        self.current_day = 0
        return self._get_observation()

    def _get_observation(self):
        """Construct current observation vector"""
        states = np.array([self.seijrd_env.network.nodes[i].state for i in range(self.n)])
        day = self.current_day
        hour = self.seijrd_env.current_tick % 24

        return np.concatenate([states.flatten(), [day, hour]]).astype(np.float32)

    def _process_action_to_matrix(self, action):
        """Process raw action to a valid commuting matrix"""
        matrix = action.reshape(self.n, self.n)
        matrix = np.maximum(matrix, 0.0)

        # Row normalize
        row_sums = matrix.sum(axis=1, keepdims=True)
        matrix = matrix / (row_sums + 1e-10)

        # Minimum self-activity
        for i in range(self.n):
            if matrix[i, i] < MIN_SELF_ACTIVITY:
                deficit = MIN_SELF_ACTIVITY - matrix[i, i]
                non_diag_sum = 1.0 - matrix[i, i]
                if non_diag_sum > 0:
                    scale_factor = (non_diag_sum - deficit) / non_diag_sum
                    for j in range(self.n):
                        if j != i:
                            matrix[i, j] *= scale_factor
                    matrix[i, i] = MIN_SELF_ACTIVITY
        return matrix

    def step(self, action):
        """Execute one step (simulating one day, 24 hours)"""
        matrix = self._process_action_to_matrix(action)
        prev_states = np.array([self.seijrd_env.network.nodes[i].state.copy() for i in range(self.n)])

        # 1) 修改通勤矩阵
        self.seijrd_env.modify_matrix(matrix)

        # 2) 进行一天(24小时)的循环：Bug4修正，顺序对齐老代码
        start_tick = self.seijrd_env.current_tick
        for hour in range(24):
            current_tick = start_tick + hour
            self.seijrd_env.current_tick = current_tick

            # 比赛开始检查
            self.seijrd_env.check_competition_start()

            # 早上8点
            if current_tick % 24 == 8:
                self.seijrd_env.network.morning_commuting()

            # 更新SEIJRD网络
            self.seijrd_env.network.update_network()

            # 晚上18点
            if current_tick % 24 == 18:
                self.seijrd_env.network.evening_commuting()

            # 比赛结束检查
            self.seijrd_env.check_competition_end()

        # 3) 计算奖励
        current_states = np.array([self.seijrd_env.network.nodes[i].state for i in range(self.n)])
        reward = self._calculate_reward(prev_states, current_states, matrix)

        # 4) 更新 day 计数
        self.current_day += 1
        done = (self.current_day >= self.days)

        # 构造 observation
        obs = self._get_observation()

        # 收集信息
        info = {
            'day': self.current_day,
            'infections': np.sum(current_states[:, 1:4]) - np.sum(prev_states[:, 1:4]),
            'deaths': np.sum(current_states[:, 5]) - np.sum(prev_states[:, 5]),
            'matrix_change': np.linalg.norm(matrix - self.original_matrix),
            'matrix': matrix
        }

        return obs, reward, done, info

    def _calculate_reward(self, previous_states, current_states, modified_matrix):
        """
        多指标综合奖励，可根据需要调参
        """
        # 1. 新增感染惩罚
        prev_infected = np.sum(previous_states[:, 1:4])
        curr_infected = np.sum(current_states[:, 1:4])
        new_infections = max(0, curr_infected - prev_infected)

        # 2. 通勤维持（原始Beta思路）
        original_commute = np.sum(self.original_matrix) - np.trace(self.original_matrix)
        current_commute = np.sum(modified_matrix) - np.trace(modified_matrix)
        commute_ratio = current_commute / (original_commute + 1e-12)

        # 3. 死亡惩罚
        prev_deaths = np.sum(previous_states[:, 5])
        curr_deaths = np.sum(current_states[:, 5])
        new_deaths = max(0, curr_deaths - prev_deaths)

        total_population = np.sum(current_states[:, :5])

        # 这里可自由设计权重
        w1 = 1.0
        w2 = 0.5
        w3 = 3.0

        # 新增感染 & 死亡都是负向
        reward = (- w1 * new_infections
                  + w2 * total_population * commute_ratio
                  - w3 * new_deaths)

        return reward


# ====================== TRAINING HELPERS & MAIN (与原代码一致) ======================
class TrainingProgressCallback(BaseCallback):
    """Callback for tracking and logging training progress"""

    def __init__(self, exp_id=0, algo="", verbose=1, log_interval=1000,
                 save_interval=50000, tensorboard_dir=None):
        super().__init__(verbose)
        self.exp_id = exp_id
        self.algo = algo
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.start_time = None
        self.last_save_time = None

        # Setup tensorboard logging
        if tensorboard_dir:
            log_dir = f"{tensorboard_dir}/{algo}_exp{exp_id}"
            # 改名为 self.tb_logger 避免与 BaseCallback.logger 冲突
            self.tb_logger = configure(log_dir, ["tensorboard"])
        else:
            self.tb_logger = None

    def _on_training_start(self):
        self.start_time = time.time()
        self.last_save_time = self.start_time
        print(f"[Exp {self.exp_id}] Starting {self.algo} training...")

    def _on_step(self):
        if self.n_calls % self.log_interval == 0:
            elapsed = time.time() - self.start_time
            steps = self.model.num_timesteps
            total_steps = self.model._total_timesteps

            progress_pct = steps / total_steps * 100
            remaining = elapsed / (steps + 1) * (total_steps - steps)

            # 从 model 的 logger 中获取实时值
            metrics = {}
            for key in ["train/policy_loss", "train/value_loss", "train/explained_variance"]:
                if key in self.model.logger.name_to_value:
                    metrics[key] = self.model.logger.name_to_value[key]

            # 打印
            metrics_str = ", ".join([f"{k.split('/')[-1]}={v:.4f}" for k, v in metrics.items()])
            print(f"[Exp {self.exp_id}] {self.algo} step {steps}/{total_steps} "
                  f"({progress_pct:.1f}%), elapsed: {elapsed / 60:.1f}m, "
                  f"ETA: {remaining / 60:.1f}m, {metrics_str}")

            # 如果想要自定义地往我们手动创建的 tb_logger 写一些统计
            if self.tb_logger is not None:
                self.tb_logger.record("time/progress_percent", progress_pct)
                self.tb_logger.record("time/elapsed_seconds", elapsed)
                self.tb_logger.record("time/remaining_seconds", remaining)
                for k, v in metrics.items():
                    self.tb_logger.record(k, v)
                self.tb_logger.dump(self.n_calls)

            # 定时保存
            if time.time() - self.last_save_time > 600:  # Save every 10 min
                self.last_save_time = time.time()
                checkpoint_path = f"./models/{self.algo}_exp{self.exp_id}_step{steps}.zip"
                self.model.save(checkpoint_path)
                print(f"[Exp {self.exp_id}] Saved checkpoint: {checkpoint_path}")

        return True


def make_env(seed=0, env_kwargs=None):
    """Create and seed a CommuneMatrixEnv environment"""

    def _init():
        env = CommuneMatrixEnv(**(env_kwargs or {}))
        env.seed(seed)
        return env

    return _init


def make_vec_env(n_envs, seeds=None, env_kwargs=None):
    """Create a vectorized environment for parallel training"""
    if seeds is None:
        seeds = list(range(n_envs))

    env_fns = [make_env(seed=seeds[i], env_kwargs=env_kwargs) for i in range(n_envs)]

    if n_envs == 1:
        return DummyVecEnv(env_fns)
    else:
        return SubprocVecEnv(env_fns)


def train_model(algo, exp_id, gpu_id, n_envs=4, total_timesteps=1000000,
                env_kwargs=None, seed=0, log_interval=1000):
    """Train a model on a specific GPU"""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    set_random_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    vec_env = make_vec_env(n_envs=n_envs,
                           seeds=[seed + i for i in range(n_envs)],
                           env_kwargs=env_kwargs)

    callbacks = [
        TrainingProgressCallback(
            exp_id=exp_id,
            algo=algo,
            log_interval=log_interval,
            tensorboard_dir="./tensorboard"
        ),
        CheckpointCallback(
            save_freq=50000 // n_envs,
            save_path=f"./models/{algo}_exp{exp_id}_checkpoints/",
            name_prefix="model",
            save_replay_buffer=False,
            save_vecnormalize=False,
        )
    ]

    if algo == "PPO":
        model = PPO(
            "MlpPolicy",
            vec_env,
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
            policy_kwargs=dict(
                net_arch=[dict(pi=[256, 256, 128], vf=[256, 256, 128])]
            )
        )
    elif algo == "A2C":
        model = A2C(
            "MlpPolicy",
            vec_env,
            device="cuda",
            verbose=0,
            tensorboard_log=f"./tensorboard/{algo}_exp{exp_id}",
            learning_rate=7e-4,
            n_steps=5,
            gamma=0.99,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(
                net_arch=[dict(pi=[256, 256, 128], vf=[256, 256, 128])]
            )
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algo}")

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True
        )

        final_model_path = f"./models/{algo}_exp{exp_id}_final.zip"
        model.save(final_model_path)
        print(f"[Exp {exp_id}] Training complete. Model saved to {final_model_path}")

        return model, final_model_path

    except Exception as e:
        print(f"[Exp {exp_id}] Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None
    finally:
        vec_env.close()


def evaluate_model(model_path, algo, exp_id, n_episodes=5):
    """Evaluate a trained model"""
    if algo == "PPO":
        model = PPO.load(model_path)
    elif algo == "A2C":
        model = A2C.load(model_path)
    else:
        raise ValueError(f"Unsupported algorithm: {algo}")

    results = {
        'rewards': [],
        'infections': [],
        'deaths': [],
        'commute_ratios': [],
        'matrices': []
    }

    for episode in range(n_episodes):
        env = CommuneMatrixEnv()
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_infections = 0
        episode_deaths = 0
        day_matrices = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            episode_infections += info['infections']
            episode_deaths += info['deaths']
            day_matrices.append(info['matrix'])

        final_matrix = day_matrices[-1]
        original_matrix = env.original_matrix
        orig_commute = np.sum(original_matrix) - np.trace(original_matrix)
        final_commute = np.sum(final_matrix) - np.trace(final_matrix)
        commute_ratio = final_commute / (orig_commute + 1e-10)

        results['rewards'].append(episode_reward)
        results['infections'].append(episode_infections)
        results['deaths'].append(episode_deaths)
        results['commute_ratios'].append(commute_ratio)
        results['matrices'].append(day_matrices)

        print(f"Episode {episode + 1}: Reward={episode_reward:.2f}, "
              f"Infections={episode_infections:.0f}, "
              f"Deaths={episode_deaths:.0f}, "
              f"Commute Ratio={commute_ratio:.2f}")

    results_dir = f"./results/{algo}_exp{exp_id}/"
    os.makedirs(results_dir, exist_ok=True)

    summary = {
        'mean_reward': np.mean(results['rewards']),
        'std_reward': np.std(results['rewards']),
        'mean_infections': np.mean(results['infections']),
        'std_infections': np.std(results['infections']),
        'mean_deaths': np.mean(results['deaths']),
        'std_deaths': np.std(results['deaths']),
        'mean_commute_ratio': np.mean(results['commute_ratios']),
        'std_commute_ratio': np.std(results['commute_ratios']),
    }

    with open(f"{results_dir}/summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    visualize_evaluation(results, algo, exp_id)
    return results, summary


def visualize_evaluation(results, algo, exp_id):
    results_dir = f"./visualizations/{algo}_exp{exp_id}/"
    os.makedirs(results_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    metrics = ['rewards', 'infections', 'deaths', 'commute_ratios']
    labels = ['Total Reward', 'Total Infections', 'Total Deaths', 'Commute Ratio']

    for i, (metric, label) in enumerate(zip(metrics, labels)):
        mean_val = np.mean(results[metric])
        std_val = np.std(results[metric])

        plt.subplot(2, 2, i + 1)
        plt.bar(0, mean_val, yerr=std_val, width=0.5, capsize=10, alpha=0.7)
        plt.title(label)
        plt.grid(True, alpha=0.3)
        plt.xticks([0], [algo])

    plt.tight_layout()
    plt.savefig(f"{results_dir}/metrics_summary.png")
    plt.close()

    if len(results['matrices']) > 0:
        matrix_sequence = results['matrices'][-1]

        plt.figure(figsize=(15, 5))
        env = CommuneMatrixEnv()
        original_matrix = env.original_matrix

        plt.subplot(1, 3, 1)
        sns.heatmap(original_matrix, cmap="Blues", annot=False)
        plt.title("Original Commuting Matrix")

        final_matrix = matrix_sequence[-1]
        plt.subplot(1, 3, 2)
        sns.heatmap(final_matrix, cmap="Blues", annot=False)
        plt.title(f"Final Day Matrix ({algo})")

        change_matrix = (final_matrix - original_matrix) / (original_matrix + 1e-10)
        plt.subplot(1, 3, 3)
        sns.heatmap(change_matrix, cmap="RdBu_r", center=0, annot=False)
        plt.title("Relative Change (blue=decrease)")

        plt.tight_layout()
        plt.savefig(f"{results_dir}/matrix_comparison.png")
        plt.close()

        np.savetxt(f"{results_dir}/original_matrix.csv", original_matrix, delimiter=",")
        np.savetxt(f"{results_dir}/final_matrix.csv", final_matrix, delimiter=",")
        np.savetxt(f"{results_dir}/change_matrix.csv", change_matrix, delimiter=",")


def parallel_train_models(configs, gpu_allocation):
    processes = []
    for config in configs:
        exp_id = config['exp_id']
        algo = config['algo']
        gpu_id = gpu_allocation[exp_id % len(gpu_allocation)]

        train_args = (
            algo,
            exp_id,
            gpu_id,
            config.get('n_envs', 4),
            config.get('total_timesteps', 1000000),
            {'days': config.get('days', 20)},
            config.get('seed', exp_id),
            config.get('log_interval', 1000)
        )

        p = multiprocessing.Process(target=train_model, args=train_args)
        p.start()
        processes.append(p)
        time.sleep(2)

    for p in processes:
        p.join()

    print("All training processes completed")


def parallel_evaluate_models(configs):
    with Pool(processes=min(multiprocessing.cpu_count(), len(configs))) as pool:
        eval_args = []
        for config in configs:
            exp_id = config['exp_id']
            algo = config['algo']
            model_path = f"./models/{algo}_exp{exp_id}_final.zip"
            if not os.path.exists(model_path):
                print(f"Model not found: {model_path}")
                continue
            eval_args.append((model_path, algo, exp_id, 5))

        pool.starmap(evaluate_model, eval_args)


def main():
    import argparse, os
    from multiprocessing import cpu_count

    parser = argparse.ArgumentParser(description="SEIJRD Commune Matrix Optimization with RL")
    parser.add_argument(
        "--mode", type=str, choices=["train", "evaluate", "visualize", "all"],
        default="all",
        help="Operation mode: train, evaluate, visualize, or all (default: all)"
    )
    parser.add_argument("--algos", type=str, nargs="+", default=["PPO", "A2C"],
                        help="Algorithms to use (PPO, A2C)")
    parser.add_argument("--exp-start", type=int, default=1,
                        help="Starting experiment ID")
    parser.add_argument("--exp-count", type=int, default=8,
                        help="Number of experiments to run")
    parser.add_argument("--timesteps", type=int, default=1000000,
                        help="Total timesteps for training")
    parser.add_argument("--days", type=int, default=20,
                        help="Simulation days per episode")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed")
    parser.add_argument("--exp-ids", type=int, nargs="+",
                        help="Specific experiment IDs to evaluate or visualize")

    args = parser.parse_args()

    # build experiment configurations
    exp_ids = args.exp_ids or list(range(args.exp_start, args.exp_start + args.exp_count))
    configs = [
        {
            'exp_id': eid,
            'algo': algo,
            'total_timesteps': args.timesteps,
            'days': args.days,
            'seed': args.seed + eid,
            'n_envs': 8
        }
        for algo in args.algos for eid in exp_ids
    ]

    # Run train
    if args.mode in ("train", "all"):
        gpu_allocation = list(range(NUM_GPUS))
        print(f"Starting training with {len(configs)} configurations on {NUM_GPUS} GPUs")
        parallel_train_models(configs, gpu_allocation)

    # Run evaluation
    if args.mode in ("evaluate", "all"):
        print(f"Evaluating {len(configs)} models")
        parallel_evaluate_models(configs)

    # Run visualization
    if args.mode in ("visualize", "all"):
        for cfg in configs:
            algo = cfg['algo']
            exp_id = cfg['exp_id']
            path = f"./models/{algo}_exp{exp_id}_final.zip"
            if os.path.exists(path):
                print(f"Creating visualizations for {algo} experiment {exp_id}")
                evaluate_model(path, algo, exp_id, n_episodes=3)
            else:
                print(f"Model not found: {path}")

    if args.mode not in ("train", "evaluate", "visualize", "all"):
        parser.print_help()



if __name__ == "__main__":
    main()
