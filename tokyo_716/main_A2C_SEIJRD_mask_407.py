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
matplotlib.use("Agg")  # 非交互后端
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

# =========== Global Config ===========
NUM = 50          # Node数
MAX_POP = 1e12
TOTAL_TIMESTEPS = 500000
LOSS_LOG_INTERVAL = 10000

# Baseline策略：对 venue_id in [3,5,24,37,40] => 0.5, 否则 => 0
BASELINE_SPECIAL_VIDS = [3,5,24,37,40]

# RL 中 SEIJRD 参数
paras = {
    "beta_1" : 0.0614,
    "beta_2" : 0.0696,
    "gamma_i": 0.0496,
    "gamma_j": 0.0376,
    "sigma_i": 0.01378,
    "sigma_j": 0.03953,
    "mu_i": 2e-5,
    "mu_j": 0.00027,
    "rho": 8.62e-05,  # 接触传染概率
    "initial_infect": 500,  # 初始感染人数
}

# 实验配置(含 delta)
# 定义基础参数
alpha = 1.0e-8
gamma = 1.0e-9

# 定义beta值范围（30个值，从8.0e-7到1.5e-6）
beta_start = 8.000e-07
beta_end = 1.500e-06
beta_values = np.linspace(beta_start, beta_end, 10)

# 定义delta值范围(10个值，从10^-9到10^-5，对数等分)
delta_values = [7e-10,5e-10, 3e-10, 1e-10,8e-11,5e-11]

# 生成所有实验配置
experiments = []
exp_id = 1

for beta in beta_values:
    for delta in delta_values:
        experiments.append({
            "exp_id": exp_id,
            "gpu_id": (exp_id - 1) % 8,  # 循环分配到8个GPU
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "delta": delta,
        })
        exp_id += 1

file_path = "data/end_copy.json"   # 场馆数据
# baseline_csv_path = "data/daily_overall_EI_delta.csv"  # 可能不用了

# =============== Data Processing ===============
def process_competitions(data_file):
    slot_mapping = {0: (8, 11), 1: (13,17), 2: (19,22)}
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
            slot = t["slot"]
            start_hour, end_hour = slot_mapping[slot]
            start_tick = days_diff*24 + start_hour
            end_tick   = days_diff*24 + end_hour
            comps.append((vid, start_tick, end_tick, capacity, slot))
    return comps

class Place:
    def __init__(self, data: Dict):
        self.id = data["venue_id"]
        self.name = data["venue_name"]
        self.capacity = data["capacity"]
        self.agenda = []
        self.endtime = -1
        self.audience_from = []

    def infect(self):
        if len(self.audience_from) == 0:
            return
        seijrd_sum = np.sum(self.audience_from, axis=0)
        E_tot = seijrd_sum[1]
        S_tot = seijrd_sum[0]
        if S_tot <= 0: return
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

# =============== SEIJRD Node/Network ===============
class SEIJRD_Node:
    def __init__(self, state, total_nodes):
        self.state = state.astype(np.float64)
        self.from_node = np.zeros((total_nodes,6))
        self.total_nodes = total_nodes

    def update_seijrd(self, delta_time=0.01, times=100, paras=paras):
        N = sum(self.state)
        SEIJRD = self.state.copy()

        if N > 0:
            dS = -paras["beta_1"] * SEIJRD[0] * SEIJRD[1] / N \
                 - paras["beta_2"] * SEIJRD[0] * (SEIJRD[2] + SEIJRD[3]) / N

            dE = paras["beta_1"] * SEIJRD[0] * SEIJRD[1] / N \
                 + paras["beta_2"] * SEIJRD[0] * (SEIJRD[2] + SEIJRD[3]) / N \
                 - paras["sigma_i"] * SEIJRD[1] - paras["sigma_j"] * SEIJRD[1]

            dI = paras["sigma_i"] * SEIJRD[1] - paras["gamma_i"] * SEIJRD[2] - paras["mu_i"] * SEIJRD[2]
            dJ = paras["sigma_j"] * SEIJRD[1] - paras["gamma_j"] * SEIJRD[3] - paras["mu_j"] * SEIJRD[3]
            dR = paras["gamma_i"] * SEIJRD[2] + paras["gamma_j"] * SEIJRD[3]
            dD = paras["mu_i"] * SEIJRD[2] + paras["mu_j"] * SEIJRD[3]
        else:
            dS = dE = dI = dJ = dR = dD = 0.0

        self.state += np.array([dS, dE, dI, dJ, dR, dD]) * delta_time * times

        Nvalue = np.sum(self.from_node, axis=1)
        self.from_node += (
                np.array([dS, dE, dI, dJ, dR, dD])[None, :]
                * ((Nvalue / N)[:, None])
                * delta_time
                * times
        )


class SEIJRD_Network:
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
                if i==j: continue
                a_ij = self.A[i][j]
                move_pop = np.minimum(arr*a_ij, self.nodes[i].state)
                self.nodes[j].from_node[i] = move_pop
                self.nodes[i].state -= move_pop
            self.nodes[i].state = np.clip(self.nodes[i].state, 0, MAX_POP)

    def evening_commuting(self):
        for i in range(self.node_num):
            for j in range(self.node_num):
                if i==j: continue
                self.nodes[j].state += self.nodes[i].from_node[j]
                self.nodes[j].state = np.clip(self.nodes[j].state, 0, MAX_POP)
            self.nodes[i].from_node = np.zeros((self.node_num,6))

    def update_network(self):
        for i in range(self.node_num):
            self.nodes[i].update_seijrd(self.delta_time, 100)

# =============== Basic Env (for hour simulation) ===============
class Env:
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

        # init states
        init_states = []
        for i in range(NUM):
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
                    self.network.nodes[i].state[:5] += pl.audience_from[i][:5]
                pl.audience_from = []
                pl.endtime = -1

# =============== 1. 先跑 baseline 策略 & 存储每日(infection, revenue) ===============
def precompute_baseline(days=20, n_venues=41, alpha=3e-8):
    """
    baseline策略: venue in [3,5,24,37,40] => 0.5, 否则=>0
    跑在同样的Env设定下, 得到每天的 infection & revenue
    Return 2 dict: day->baseInf, day->baseRev
    """
    # 构造一个相同 days 的临时env
    env_ = MultiVenueSEIREnv(days=days, n_venues=n_venues, alpha=alpha, beta=0.0, gamma=0.0, delta=0.0,
                             # 这里临时设置 beta=0, gamma=0, delta=0, 仅为不影响日常计算
                             # 因为我们只想拿到daily_newI, daily_revenue
                             # (也可以保留,无所谓,只要日常yield都能拿到)
                            baseline_mode=True)
    # baseline_mode是我们加进来的标记(见下)
    env_.reset()

    base_action_2d = np.zeros((n_venues,3), dtype=np.float32)
    for vid in BASELINE_SPECIAL_VIDS:
        base_action_2d[vid-1] = 0.5

    day2baseInf = {}
    day2baseRev = {}

    done=False
    day=0
    obs=env_.reset()
    while not done:
        action = base_action_2d.reshape(-1)
        obs, rew, done, info = env_.step(action)
        # info中可以得到 daily_newI, daily_revenue
        # 但我们没直接存 daily_revenue in info, 需要在 env step() 里加
        # (见下MultiVenueSEIREnv 里, baseline_mode时可把 daily_newI/daily_revenue放 info)
        day2baseInf[day] = info["daily_infection"]
        day2baseRev[day] = info["daily_revenue"]
        day+=1

    return day2baseInf, day2baseRev

# =============== 2. 自定义 Gym Env => 引入baseline对比 ===============
class MultiVenueSEIREnv(gym.Env):
    """
    - Each step => 1 day
    - daily_reward = (daily_revenue - baseRev) - [beta*(exceedI) + gamma*(exceedE)]
      其中 exceedI = max(0, daily_newI - baseInf)
    - baseline数据由 precompute_baseline() 提前算好
    - final reward 还可以加 delta*(drate - irate) (可选)
    - baseline_mode: 若True, step()里把 daily_infection/daily_revenue写进info，用于外部记录
    """
    def __init__(self, days=20, n_venues=41,
                 alpha=3e-8, beta=8.5e-7, gamma=1e-9, delta=1e-3,
                 baseline_mode=False,
                 baseline_infection=None,
                 baseline_revenue=None):
        super(MultiVenueSEIREnv, self).__init__()
        self.days = days
        self.n_venues = n_venues
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma
        self.delta = delta
        self.baseline_mode = baseline_mode

        self.seir_env = Env()
        self.seir_env.init_env()

        # baseline 结果, 如果不为 None, 则step()会对比
        self.base_infection_data = baseline_infection  or {}
        self.base_revenue_data   = baseline_revenue    or {}

        self.action_space = spaces.Box(low=0, high=1,
                                       shape=(n_venues*3,),
                                       dtype=np.float32)
        self.observation_space = spaces.Box(low=0.0, high=MAX_POP,
                                            shape=(6*NUM + n_venues*3,),
                                            dtype=np.float32)

        self.current_day = 0
        self.prev_E, self.prev_I = self._get_total_EI()
        self.state = self._get_observation()

        self.cum_rl_revenue = 0.0
        self.cum_rl_infection = 0.0
        self.cum_base_revenue = 0.0
        self.cum_base_infection = 0.0

        self.np_random = None
        self.seed_value = None

    def _get_total_EI(self):
        E,I = 0.0,0.0
        for i in range(NUM):
            st = self.seir_env.network.nodes[i].state
            E += st[1]
            I += st[2]
        return E,I

    def _generate_mask(self):
        mask = np.zeros((self.n_venues,3), dtype=np.float32)
        for vid, place in self.seir_env.places.items():
            for comp in place.agenda:
                start_tick = comp[1]
                d_comp = start_tick // 24
                if d_comp == self.current_day:
                    slot_id = comp[4]
                    mask[vid-1, slot_id] = 1.0
        return mask

    def _get_observation(self):
        obs_list = []
        for i in range(NUM):
            st = self.seir_env.network.nodes[i].state
            obs_list.extend(np.clip(st,0,MAX_POP).tolist())
        mask_2d = self._generate_mask()
        obs_list.extend(mask_2d.flatten().tolist())
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
        self.cum_rl_revenue = 0.0
        self.cum_rl_infection = 0.0
        self.cum_base_revenue = 0.0
        self.cum_base_infection = 0.0
        self.state = self._get_observation()
        return self.state

    def step(self, action):
        obs_len = 6*NUM
        mask_len = self.n_venues*3
        mask_flat = self.state[obs_len : obs_len+mask_len]
        mask_2d = mask_flat.reshape((self.n_venues,3))

        action = np.clip(action, 0,1).reshape((self.n_venues,3))
        effective_action = action * mask_2d
        self.seir_env.capacity_strategy = effective_action.flatten()

        # run day
        start_tick = self.current_day*24
        end_tick   = (self.current_day+1)*24
        for hour in range(start_tick, end_tick):
            self.seir_env.current_tick = hour
            self.seir_env.check_competition_start()
            if hour%24==8:
                self.seir_env.network.morning_commuting()
            self.seir_env.network.update_network()
            if hour%24==18:
                self.seir_env.network.evening_commuting()
            self.seir_env.check_competition_end()

        # daily new E/I
        curE, curI = self._get_total_EI()
        daily_newE = curE - self.prev_E
        daily_newI = curI - self.prev_I
        self.prev_E, self.prev_I = curE, curI

        # daily revenue
        daily_revenue=0.0
        for pid, place in self.seir_env.places.items():
            ratio_3 = effective_action[pid-1,:].sum()
            daily_revenue += ratio_3 * place.capacity
        daily_revenue *= self.alpha

        # baseline data
        base_inf = self.base_infection_data.get(self.current_day, 0.0)
        base_rev = self.base_revenue_data.get(self.current_day, 0.0)

        # 计算差
        inf_diff = daily_newI - base_inf
        rev_diff = daily_revenue - base_rev
        exceedI  = max(0.0, inf_diff)  # 只罚感染超过baseline的那部分
        # 你若有 E => 同理 exceedE = ...
        # 这里示例只对I做惩罚:
        # daily_reward = rev_diff - self.beta*exceedI
        # 如果你还有 exceedE => daily_reward = rev_diff - (beta*exceedI + gamma*exceedE)
        # 这看你具体逻辑

        # 若你要 E 也对比: 先看 run_baseline(...) 里有没有算 daily_newE, 这里留给你扩展:
        exceedE = 0.0
        # daily_newE baseline也可存, 这里先略.

        daily_reward = rev_diff - (self.beta*exceedI + self.gamma*exceedE)

        # 更新 RL/BASE 累计
        self.cum_rl_revenue += daily_revenue
        self.cum_rl_infection += daily_newI
        self.cum_base_revenue += base_rev
        self.cum_base_infection += base_inf

        self.current_day += 1
        done = (self.current_day>=self.days)
        info = {
            "day": self.current_day,
            "newI": daily_newI,
            "daily_infection": daily_newI,   # baseline_mode用
            "daily_revenue": daily_revenue, # baseline_mode用
        }

        if done and not self.baseline_mode:
            # final delta reward
            drate = 0.0
            irate = 0.0
            # baseline 累计
            if self.cum_base_revenue>0:
                drate = (self.cum_rl_revenue - self.cum_base_revenue)/(self.cum_base_revenue+1e-9)
            if self.cum_base_infection>0:
                irate = (self.cum_rl_infection - self.cum_base_infection)/(self.cum_base_infection+1e-9)
            final_bonus = self.delta*(drate - irate)
            daily_reward += final_bonus

            ratio_RI = 0.0
            if abs(irate)>1e-9:
                ratio_RI = drate/irate
            info["final_ratio_RI"] = ratio_RI
            info["cumulative_reward"] = self.cum_rl_revenue

        self.state = self._get_observation()
        return self.state, daily_reward, done, info


# =============== 先计算 baseline, 再训练 RL ===============
def run_baseline_precompute(days=20, n_venues=41, alpha=3e-8):
    """
    计算 baseline 每天感染/收益
    """
    inf_dict, rev_dict = precompute_baseline(days, n_venues, alpha=alpha)
    return inf_dict, rev_dict

# =============== Compare capacity strategies (unchanged) ===============
def run_baseline_str(env):
    """
    如果你想在可视化对比时还跑一遍 baseline:
    """
    env.reset()
    baseline_action_2d = np.zeros((env.n_venues,3), dtype=np.float32)
    for vid in BASELINE_SPECIAL_VIDS:
        baseline_action_2d[vid-1] = 0.5

    records = {"day":[],"new_I":[],"reward":[]}
    done=False
    day=0
    obs=env.reset()
    while not done:
        action= baseline_action_2d.reshape(-1)
        obs, rew, done, info= env.step(action)
        records["day"].append(day)
        records["new_I"].append(info["newI"])
        records["reward"].append(rew)
        day+=1
    df = pd.DataFrame(records)
    return df

def run_constant_ratio(env, ratio):
    env.reset()
    const_act_2d = np.ones((env.n_venues,3), dtype=np.float32)*ratio
    rec = {"day":[],"new_I":[],"reward":[]}
    done=False
    day=0
    obs=env.reset()
    while not done:
        action= const_act_2d.reshape(-1)
        obs, rew, done, info= env.step(action)
        rec["day"].append(day)
        rec["new_I"].append(info["newI"])
        rec["reward"].append(rew)
        day+=1
    df = pd.DataFrame(rec)
    return df

def compare_capacity_strategies(env, model, exp_id, algo, alpha, beta, gamma, delta):
    baseline_df = run_baseline_str(env)
    baseline_df["cum_new_I"] = baseline_df["new_I"].cumsum()
    baseline_df["cum_reward"]= baseline_df["reward"].cumsum()

    ratio_list = [0.0, 0.3, 0.5, 1.0]
    ratio_data = {}
    for r in ratio_list:
        cdf = run_constant_ratio(env, r)
        cdf["cum_new_I"]= cdf["new_I"].cumsum()
        cdf["cum_reward"]= cdf["reward"].cumsum()
        ratio_data[r]= cdf

    # RL
    env.reset()
    rl_dict= {"day":[],"new_I":[],"reward":[]}
    obs=env.reset()
    done=False
    day=0
    while not done:
        act,_= model.predict(obs, deterministic=True)
        obs, rew, done, info= env.step(act)
        rl_dict["day"].append(day)
        rl_dict["new_I"].append(info["newI"])
        rl_dict["reward"].append(rew)
        day+=1
    rl_df= pd.DataFrame(rl_dict)
    rl_df["cum_new_I"]= rl_df["new_I"].cumsum()
    rl_df["cum_reward"]= rl_df["reward"].cumsum()

    fig,axes= plt.subplots(2,2, figsize=(12,10))
    ax1,ax2,ax3,ax4= axes.flatten()

    ax1.plot(baseline_df["day"], baseline_df["new_I"], label="Baseline", lw=2)
    for r,dfc in ratio_data.items():
        ax1.plot(dfc["day"], dfc["new_I"], label=f"ratio {r}")
    ax1.plot(rl_df["day"], rl_df["new_I"], label="RL", lw=2)
    ax1.set_title("Daily NewI")
    ax1.legend(); ax1.grid(True)

    ax2.plot(baseline_df["day"], baseline_df["cum_new_I"], label="Baseline", lw=2)
    for r,dfc in ratio_data.items():
        ax2.plot(dfc["day"], dfc["cum_new_I"], label=f"ratio {r}")
    ax2.plot(rl_df["day"], rl_df["cum_new_I"], label="RL", lw=2)
    ax2.set_title("Cumulative NewI")
    ax2.legend(); ax2.grid(True)

    ax3.plot(baseline_df["day"], baseline_df["reward"], label="Baseline", lw=2)
    for r,dfc in ratio_data.items():
        ax3.plot(dfc["day"], dfc["reward"], label=f"ratio {r}")
    ax3.plot(rl_df["day"], rl_df["reward"], label="RL", lw=2)
    ax3.set_title("Daily Reward")
    ax3.legend(); ax3.grid(True)

    ax4.plot(baseline_df["day"], baseline_df["cum_reward"], label="Baseline", lw=2)
    for r,dfc in ratio_data.items():
        ax4.plot(dfc["day"], dfc["cum_reward"], label=f"ratio {r}")
    ax4.plot(rl_df["day"], rl_df["cum_reward"], label="RL", lw=2)
    ax4.set_title("Cumulative Reward")
    ax4.legend(); ax4.grid(True)

    fig.suptitle(f"Strategy Compare (Exp {exp_id}, {algo}, α={alpha}, β={beta}, γ={gamma}, δ={delta})")
    plt.tight_layout()
    os.makedirs("./visualizations", exist_ok=True)
    outpng = f"./visualizations/exp_{exp_id}_{algo}_strategy_compare.png"
    plt.savefig(outpng)
    plt.close()
    print(f"Saved strategy comparison => {outpng}")

# =============== Callback for loss logging ===============
class LossTrackingCallback(BaseCallback):
    def __init__(self, exp_id=0, total_timesteps=TOTAL_TIMESTEPS,
                 log_interval=LOSS_LOG_INTERVAL, verbose=1):
        super(LossTrackingCallback, self).__init__(verbose)
        self.exp_id=exp_id
        self.total_timesteps=total_timesteps
        self.log_interval=log_interval
        self.start_time=None
        self.loss_data= {
            "timesteps": [],
            "policy_loss":[],
            "value_loss":[],
            "elapsed_time": [],
        }

    def _on_training_start(self):
        self.start_time= time.time()
        print(f"[Exp {self.exp_id}] Start training, record losses every {self.log_interval} steps...")

    def _on_step(self):
        steps= self.model.num_timesteps
        if steps%self.log_interval==0:
            elapsed= time.time()-self.start_time
            pol_loss= self.model.logger.name_to_value.get("train/policy_loss",None)
            val_loss= self.model.logger.name_to_value.get("train/value_loss",None)
            if pol_loss is not None and val_loss is not None:
                self.loss_data["timesteps"].append(steps)
                self.loss_data["policy_loss"].append(pol_loss)
                self.loss_data["value_loss"].append(val_loss)
                self.loss_data["elapsed_time"].append(elapsed)

                fraction= steps/self.total_timesteps
                eta= elapsed/fraction - elapsed if fraction>0 else 0
                print(f"[Exp {self.exp_id}] steps={steps}/{self.total_timesteps}, pol_loss={pol_loss:.2e}, val_loss={val_loss:.2e}, ETA={eta/60:.1f}m")

                self._save_loss_data()
        return True

    def _save_loss_data(self):
        outdir= f"./loss_results/exp_{self.exp_id}"
        os.makedirs(outdir,exist_ok=True)
        df= pd.DataFrame(self.loss_data)
        csv_path= os.path.join(outdir,"loss_data.csv")
        df.to_csv(csv_path, index=False)
        print(f"[Exp {self.exp_id}] Save to {csv_path}")

# =============== Train + Visualization ===============
def make_env(env_id, days, n_venues, alpha, beta, gamma, delta,
             baseline_infection, baseline_revenue):
    def _init():
        e= MultiVenueSEIREnv(days=days, n_venues=n_venues,
                             alpha=alpha, beta=beta, gamma=gamma, delta=delta,
                             baseline_infection= baseline_infection,
                             baseline_revenue= baseline_revenue,
                             baseline_mode=False)
        e.seed(env_id)
        return e
    return _init

def train_experiment(exp_id, gpu_id, alpha, beta, gamma, delta,
                     total_timesteps=TOTAL_TIMESTEPS,
                     log_interval=LOSS_LOG_INTERVAL,
                     days=20, n_venues=41):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # 1) 先算 baseline
    print(f"[Exp {exp_id}] Precompute baseline ...")
    base_infection_dict, base_revenue_dict = precompute_baseline(days, n_venues, alpha=alpha)

    # 2) 构造多进程env, 传入 baseline dict
    num_envs=4
    env_fns= [make_env(i, days, n_venues, alpha, beta, gamma, delta,
                       base_infection_dict, base_revenue_dict)
              for i in range(num_envs)]
    vec_env = SubprocVecEnv(env_fns)

    # 3) callback
    logdir= f"./logs/exp_{exp_id}_A2C_a{alpha}_b{beta}_g{gamma}_d{delta}"
    callback = LossTrackingCallback(exp_id=exp_id,
                                    total_timesteps=total_timesteps,
                                    log_interval=log_interval)
    # 4) train
    model= A2C("MlpPolicy", vec_env, verbose=1, tensorboard_log=logdir)
    model.learn(total_timesteps=total_timesteps,
                callback=callback,
                progress_bar=True)

    os.makedirs("./results", exist_ok=True)
    model_path= f"./results/exp_{exp_id}_A2C_a{alpha}_b{beta}_g{gamma}_d{delta}.zip"
    model.save(model_path)
    vec_env.close()

    print(f"[Exp {exp_id}] Done training => {model_path}")
    # 5) Visualization
    # 仅训练时不用对比. 之后run_experiments全跑完再统一对比也可

def visualize_model_results(exp_id, alpha, beta, gamma, delta, days=20, n_venues=41):
    """
    1) Load final model
    2) Rebuild an env with the same baseline data
    3) Plot daily newI, daily newE, actions heatmap ...
    4) Compare capacity strategies
    """
    model_path= f"./results/exp_{exp_id}_A2C_a{alpha}_b{beta}_g{gamma}_d{delta}.zip"
    if not os.path.exists(model_path):
        print(f"[Exp {exp_id}] Model file not found => {model_path}")
        return

    # Recompute baseline dictionary
    base_infection_dict, base_revenue_dict = precompute_baseline(days, n_venues, alpha)

    model= A2C.load(model_path)
    # 构造单env
    env= MultiVenueSEIREnv(days=days, n_venues=n_venues,
                           alpha=alpha, beta=beta, gamma=gamma, delta=delta,
                           baseline_infection= base_infection_dict,
                           baseline_revenue=   base_revenue_dict,
                           baseline_mode=False)

    # rollout
    data_rec= {"day":[],"newI":[],"newE":[],"reward":[],"actions":[]}
    obs= env.reset()
    done=False
    day=0
    while not done:
        act,_= model.predict(obs, deterministic=True)
        obs, rew, done, info= env.step(act)
        data_rec["day"].append(day)
        data_rec["newI"].append(info["newI"])
        data_rec["newE"].append(info.get("newE",0)) # if not in info => 0
        data_rec["reward"].append(rew)
        data_rec["actions"].append(act)
        day+=1
    df= pd.DataFrame(data_rec)

    # plot daily newI
    plt.figure(figsize=(10,6))
    plt.plot(df["day"], df["newI"], 'r-', label="daily newI")
    plt.plot(df["day"], df["newE"], 'y-', label="daily newE")
    plt.title(f"Exp {exp_id} - daily newI & newE")
    plt.legend(); plt.grid(True)
    os.makedirs("./visualizations", exist_ok=True)
    plt.savefig(f"./visualizations/exp_{exp_id}_newIE.png")
    plt.close()

    # heatmap
    acts_arr= np.array(df["actions"].tolist())
    plt.figure(figsize=(12,8))
    sns.heatmap(acts_arr.T, cmap="YlOrRd",
                xticklabels=range(df.shape[0]),
                yticklabels=range(acts_arr.shape[1]))
    plt.title(f"Exp {exp_id} - Action Heatmap")
    plt.xlabel("Days")
    plt.ylabel("Action Dimension")
    plt.savefig(f"./visualizations/exp_{exp_id}_policy.png")
    plt.close()

    # compare strategies
    compare_capacity_strategies(env, model, exp_id, "A2C", alpha, beta, gamma, delta)

def run_experiments():
    procs=[]
    for cfg in experiments:
        exp_id= cfg["exp_id"]
        gpu_id= cfg["gpu_id"]
        alpha= cfg["alpha"]
        beta=  cfg["beta"]
        gamma= cfg["gamma"]
        delta= cfg["delta"]

        p= multiprocessing.Process(
            target=train_experiment,
            args=(exp_id, gpu_id, alpha, beta, gamma, delta,
                  TOTAL_TIMESTEPS, LOSS_LOG_INTERVAL,20,41)
        )
        p.start()
        procs.append(p)
    for p in procs:
        p.join()

    # 全部训练结束后,可再做可视化
    for cfg in experiments:
        exp_id= cfg["exp_id"]
        alpha= cfg["alpha"]
        beta=  cfg["beta"]
        gamma= cfg["gamma"]
        delta= cfg["delta"]
        visualize_model_results(exp_id, alpha, beta, gamma, delta)

    print("All done !")


if __name__=="__main__":
    run_experiments()
