#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
main_RL_910.py

目标：
- 将动力学更新为“717 版本”的双 β ODE（beta_1 作用于 E，beta_2 作用于 I/J 的传染项）；
- 其余训练/评估/可视化产物与 `main_RL_58.py` 对齐：小时粒度 step、JSON/XLSX 结构、
  多目标奖励/自适应占位、阈值策略、离散动作模板；
- 兼容 `advanced_visuals.py` 的读取与出图；
- 额外修复：补上小时环境中比赛 tick 的推进（current_tick）。

依赖：
- stable-baselines3, gym, numpy, pandas, matplotlib, seaborn
- 数据文件：
  data/end_copy.json
  data/tokyo_commuting_flows_with_intra.csv
  data/tokyo_population.csv
  （可选）data/SEIJRD.csv
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

import warnings
warnings.filterwarnings('ignore')

# ------------------- 目录与硬件参数 -------------------
os.makedirs("./models", exist_ok=True)
os.makedirs("./logs", exist_ok=True)
os.makedirs("./results", exist_ok=True)
os.makedirs("./visualizations", exist_ok=True)
os.makedirs("./tensorboard", exist_ok=True)
os.makedirs("./configs", exist_ok=True)
os.makedirs("./interactive", exist_ok=True)

NUM_GPUS = int(os.environ.get("NUM_GPUS", "8"))
MAX_POP = 1e12

# ------------------- 717版动力学参数（双β） -------------------
backup_paras = {
    "beta_1" : 0.0614,  # E 的传染项系数
    "beta_2" : 0.0696,  # I/J 的传染项系数
    "gamma_i": 0.0496,
    "gamma_j": 0.0376,
    "sigma_i": 0.01378,
    "sigma_j": 0.03953,
    "mu_i": 2e-5,
    "mu_j": 0.00027,
    "rho": 8.62e-05,         # 场馆内接触传染概率
    "initial_infect": 500,   # 初始感染人数（仅在非CSV初值时使用）
}

DEFAULT_IS_STATES = False
MIN_SELF_ACTIVITY = 0.3  # 行为约束：对角线最低比例
file_path = "data/end_copy.json"

# =======================================================
# 数据加载辅助
# =======================================================
def Input_Population(fp="data/tokyo_population.csv"):
    df = pd.read_csv(fp, header=None,
                     names=["code","ward_jp","ward_en","dummy","population"])
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

def process_competitions(data_file):
    """将 end_copy.json 的场馆-场次转化为 (venue_id, start_tick, end_tick, capacity, slot) 列表"""
    slot_mapping = {0:(8,11), 1:(13,17), 2:(19,22)}
    year=2021
    earliest=None
    for venue in data_file:
        for t in venue["time"]:
            edate = datetime.date(year, t["month"], t["day"])
            if earliest is None or edate<earliest:
                earliest=edate
    comps=[]
    for venue in data_file:
        vid = venue["venue_id"]
        cap = int(venue["capacity"].replace(',',''))
        for t in venue["time"]:
            edate = datetime.date(year, t["month"], t["day"])
            diff=(edate-earliest).days
            slot=t["slot"]
            sh,eh=slot_mapping[slot]
            st_tick = diff*24+sh
            ed_tick = diff*24+eh
            comps.append((vid,st_tick,ed_tick,cap,slot))
    return comps

def process_places(data_file):
    places={}
    for d in data_file:
        vid = d["venue_id"]
        place_info={
            "venue_id": vid,
            "venue_name": d["venue_name"],
            "capacity": int(d["capacity"].replace(',',''))
        }
        places[vid]=Place(place_info)
    return places

# =======================================================
# 场馆/传播
# =======================================================
class Place:
    def __init__(self, data: Dict):
        self.id = data["venue_id"]
        self.name= data["venue_name"]
        self.capacity=data["capacity"]
        self.agenda=[]
        self.endtime=None
        self.audience_from=[]  # shape=(num_nodes, 6)

    def infect(self, paras):
        """
        E+I+J 皆可传染；参数中使用 rho（场馆接触传染概率）。
        """
        if len(self.audience_from)==0:
            return
        arr = np.sum(self.audience_from, axis=0)  # [S,E,I,J,R,D]
        inf_num = arr[1]+arr[2]+arr[3]
        if inf_num<=0:
            return
        prob = 1-(1-paras["rho"])**inf_num
        prob=np.clip(prob,0,1)

        S_tot=arr[0]
        if S_tot<1e-9:
            return
        new_inf=S_tot*prob

        ratio=self.audience_from[:,0]/(S_tot+1e-12)
        new_inf_each=ratio*new_inf
        self.audience_from[:,0]-=new_inf_each
        self.audience_from[:,1]+=new_inf_each

# =======================================================
# 网络与节点（717版双β ODE）
# =======================================================
class SEIR_Node:
    """
    注意：这里仍沿用历史命名 SEIR_Node/SEIR_Network，但状态是 SEIJRD 六仓。
    """
    def __init__(self, state, total_nodes):
        self.state=state.astype(float)   # [S,E,I,J,R,D]
        self.total_nodes=total_nodes
        self.from_node=np.zeros((total_nodes,6))
        self.to_node=np.zeros((total_nodes,5))

    def update_seir(self, delta_time=1/2400, times=100, paras=backup_paras):
        """
        717 版双 β ODE：
            dS = - beta_1 * S * E / N - beta_2 * S * (I+J) / N
            dE = + beta_1 * S * E / N + beta_2 * S * (I+J) / N - sigma_i*E - sigma_j*E
            dI = + sigma_i*E - gamma_i*I - mu_i*I
            dJ = + sigma_j*E - gamma_j*J - mu_j*J
            dR = + gamma_i*I + gamma_j*J
            dD = + mu_i*I + mu_j*J

        为保证与旧版一致，这里采用“先算导数，再累加 times 次”的显式 Euler 近似；
        同时把 from_node（白天流动的量）按比例同步更新。
        """
        S,E,I,J,R,D_ = self.state
        N = S+E+I+J+R  # D 不参与接触
        if N>0:
            dS = - paras["beta_1"]*S*E/N - paras["beta_2"]*S*(I+J)/N
            dE = -dS - paras["sigma_i"]*E - paras["sigma_j"]*E
            dI = paras["sigma_i"]*E - paras["gamma_i"]*I - paras["mu_i"]*I
            dJ = paras["sigma_j"]*E - paras["gamma_j"]*J - paras["mu_j"]*J
            dR = paras["gamma_i"]*I + paras["gamma_j"]*J
            dD = paras["mu_i"]*I + paras["mu_j"]*J
        else:
            dS=dE=dI=dJ=dR=dD=0.0

        # Euler 迭代
        for _ in range(times):
            self.state += np.array([dS,dE,dI,dJ,dR,dD])*delta_time
            # 同步流动累积（保持与 54/58 的 from_node 处理一致）
            Nvalue=np.sum(self.from_node,axis=1)  # 每个来源的在场总量
            ratio=(Nvalue/(N+1e-12))[:,None] if N>0 else np.zeros((self.total_nodes,1))
            inc=np.array([dS,dE,dI,dJ,dR,dD])[None,:]*ratio*delta_time
            self.from_node += inc

        self.state=np.clip(self.state,0,MAX_POP)
        self.from_node=np.clip(self.from_node,0,MAX_POP)

class SEIR_Network:
    def __init__(self, num, states, matrix=None, paras=backup_paras):
        self.node_num=num
        self.nodes={}
        for i in range(num):
            self.nodes[i]=SEIR_Node(states[i], num)

        if matrix is not None:
            row_sum=matrix.sum(axis=1,keepdims=True)
            self.A=matrix/(row_sum+1e-12)
        else:
            A_=np.random.rand(num,num)
            A_/=A_.sum(axis=1,keepdims=True)
            self.A=A_
        self.paras=paras
        self.delta_time=1/2400

    def morning_commuting(self):
        for i in range(self.node_num):
            SEIJR=self.nodes[i].state[:5]
            for j in range(self.node_num):
                self.nodes[i].to_node[j]=SEIJR*self.A[i][j]
        # 倾巢而出
        for i in range(self.node_num):
            self.nodes[i].state[:5]=0.0
        # 抵达并记录来源
        for i in range(self.node_num):
            for j in range(self.node_num):
                self.nodes[j].state[:5]+=self.nodes[i].to_node[j]
                self.nodes[j].from_node[i,:5]=self.nodes[i].to_node[j]

    def evening_commuting(self):
        for i in range(self.node_num):
            for j in range(self.node_num):
                if i==j:
                    continue
                self.nodes[j].state-=self.nodes[j].from_node[i]
                self.nodes[i].state+=self.nodes[j].from_node[i]
        for i in range(self.node_num):
            self.nodes[i].from_node=np.zeros((self.node_num,6))
            self.nodes[i].to_node=np.zeros((self.node_num,5))

    def update_network(self):
        for i in range(self.node_num):
            self.nodes[i].update_seir(self.delta_time, 100, self.paras)

# =======================================================
# Env：比赛、通勤、矩阵与 β 动态耦合
# =======================================================
class Env:
    def __init__(self, is_states=DEFAULT_IS_STATES):
        self.is_states=is_states
        self.num=23
        self.network=None
        self.competitions=[]
        self.places={}
        self.matrix=None
        self.origin_matrix=None
        self.current_tick=0
        self.paras=backup_paras.copy()

    def init_env(self):
        data_json=json.load(open(file_path,"r",encoding="utf-8"))
        self.competitions=process_competitions(data_json)
        self.places=process_places(data_json)
        for c in self.competitions:
            vid=c[0]
            self.places[vid].agenda.append(c)
        for pl in self.places.values():
            pl.agenda.sort(key=lambda x:x[1])

        # 初始状态
        if self.is_states and os.path.exists("data/SEIJRD.csv"):
            init_states = Input_Intial("data/SEIJRD.csv")
        else:
            pops = Input_Population("data/tokyo_population.csv")
            init_states = np.zeros((self.num,6))
            for i in range(self.num):
                S_ = pops[i] - self.paras["initial_infect"]*4.5/self.num
                E_ = self.paras["initial_infect"]*2/self.num
                I_ = self.paras["initial_infect"]*1/self.num
                J_ = self.paras["initial_infect"]*1.5/self.num
                init_states[i] = np.array([S_,E_,I_,J_,0,0], dtype=float)

        # 通勤矩阵
        try:
            mat = Input_Matrix("data/tokyo_commuting_flows_with_intra.csv")
            row_sum=mat.sum(axis=1, keepdims=True)
            mat=mat/(row_sum+1e-12)
            self.origin_matrix=mat.copy()
            self.matrix=mat.copy()
        except Exception as e:
            print("Warning: load commuting matrix failed, use identity.", e)
            self.origin_matrix=np.eye(self.num)
            self.matrix=np.eye(self.num)

        self.network = SEIR_Network(self.num, init_states, self.matrix, self.paras)
        self.current_tick=0

    def check_competition_start(self):
        for pid,place in self.places.items():
            if not place.agenda:
                continue
            comp = place.agenda[0]
            if comp[1]==self.current_tick:
                place.audience_from = np.zeros((self.num,6))
                place.agenda.pop(0)
                place.endtime = comp[2]
                capacity = comp[3]
                slot_id  = comp[4]

                # 简单按各区在场人数的比例分配入场观众
                sums = [np.sum(self.network.nodes[i].state[:5]) for i in range(self.num)]
                sums = np.array(sums)
                tot = np.sum(sums)
                if tot<1e-9: tot=1.0
                for i in range(self.num):
                    portion = self.network.nodes[i].state[:5]*(capacity/tot)
                    portion = np.minimum(portion, self.network.nodes[i].state[:5])
                    place.audience_from[i,:5]=portion
                    self.network.nodes[i].state[:5]-=portion

                place.infect(self.paras)

    def check_competition_end(self):
        for pid,place in self.places.items():
            if place.endtime==self.current_tick:
                for i in range(self.num):
                    self.network.nodes[i].state[:5]+=place.audience_from[i,:5]
                place.audience_from=[]
                place.endtime=None

    def modify_matrix(self, matrix):
        """
        行归一 & β 动态耦合（双 β 版本）：
        - 将通勤强度的相对比例 ratio = offdiag_now/offdiag_orig
        - beta_1/beta_2 随 ratio 在 [floor, beta_*] 之间线性缩放
        """
        row_sum = matrix.sum(axis=1, keepdims=True)
        matrix_ = matrix/(row_sum+1e-12)
        self.network.A=matrix_.copy()
        self.matrix=matrix_.copy()

        base_off = np.sum(self.origin_matrix) - np.trace(self.origin_matrix)
        now_off  = np.sum(self.matrix)        - np.trace(self.matrix)
        ratio = now_off/(base_off+1e-12)
        # 楼层保底（可按需调参）
        b1_floor, b2_floor = 0.02, 0.02
        self.network.paras["beta_1"] = max(b1_floor, backup_paras["beta_1"]*ratio)
        self.network.paras["beta_2"] = max(b2_floor, backup_paras["beta_2"]*ratio)

# =======================================================
# 强化学习环境（小时粒度；修复 tick 推进）
# =======================================================
class CommuneMatrixEnv(gym.Env):
    """
    每 step = 1 小时；1 天 = 24 step；episode 长度 = 24*days。
    观赛开始/结束由 Env.current_tick 控制，本文件中已在 step() 内推进。
    """
    metadata = {"render.modes": []}

    def __init__(self, days=20, is_states=False,
                 reward_mode='balanced', use_adaptive=False,
                 use_threshold_policy=False, use_discrete=False):
        super().__init__()
        self.days=days
        self.is_states=is_states
        self.reward_mode=reward_mode
        self.use_adaptive=use_adaptive
        self.use_threshold=use_threshold_policy
        self.use_discrete=use_discrete

        self.seir_env=Env(is_states=self.is_states)
        self.seir_env.init_env()
        self.num=self.seir_env.num

        obs_dim=self.num*6+2
        self.observation_space=spaces.Box(low=-np.inf,high=np.inf,shape=(obs_dim,),dtype=np.float32)
        self.action_space=spaces.Box(low=0,high=1,shape=(self.num*self.num,),dtype=np.float32)

        self.current_hour=0
        self.current_day=0
        self.episode_steps=0
        self.max_steps=24*self.days

        self.original_matrix=self.seir_env.origin_matrix.copy()
        self.np_random=None

        # 载入/准备 adaptive
        self.reward_configs, self.adaptive_func = load_multi_objective_rewards()

        # 离散模板
        self.discrete_templates, self.apply_discrete = discrete_action_map()

    def seed(self, seed=None):
        self.np_random, seed_ = seeding.np_random(seed)
        return [seed_]

    def reset(self):
        self.seir_env=Env(is_states=self.is_states)
        self.seir_env.init_env()
        self.current_hour=0
        self.current_day=0
        self.episode_steps=0
        self.original_matrix=self.seir_env.origin_matrix.copy()
        return self._get_obs()

    def step(self, action):
        # 先推进 tick（修复 54/58 中漏更新的问题）
        self.seir_env.current_tick = int(self.current_day*24 + self.current_hour)

        # 计算当小时使用的矩阵
        if self.use_threshold:
            matrix=threshold_policy(self)  # 使用传统阈值策略
        else:
            matrix=self._process_action(action)
            if self.use_discrete:
                matrix=self.discrete_templates['reduce_50'](matrix)  # 例：半折返家

        # 应用矩阵并耦合 β
        self.seir_env.modify_matrix(matrix)

        # 比赛开始检查（按 tick）
        self.seir_env.check_competition_start()

        # 早上 8 点通勤
        if self.current_hour==8:
            self.seir_env.network.morning_commuting()

        # 演化 1 小时
        prev_states=np.array([nd.state.copy() for nd in self.seir_env.network.nodes.values()])
        self.seir_env.network.update_network()

        # 晚上 18 点返家
        if self.current_hour==18:
            self.seir_env.network.evening_commuting()

        # 比赛结束检查（按 tick）
        self.seir_env.check_competition_end()

        # 计算奖励
        curr_states=np.array([nd.state for nd in self.seir_env.network.nodes.values()])
        if self.use_adaptive or self.reward_mode=='adaptive':
            rew, extra_info=compute_multi_objective_reward(prev_states,curr_states,matrix,self.seir_env,
                                                           reward_mode='adaptive',
                                                           time_step=self.episode_steps,
                                                           adaptive_func=self.adaptive_func)
        else:
            rew, extra_info=compute_multi_objective_reward(prev_states,curr_states,matrix,self.seir_env,
                                                           reward_mode=self.reward_mode,
                                                           time_step=self.episode_steps)

        # 推进时间
        self.current_hour+=1
        if self.current_hour>=24:
            self.current_hour=0
            self.current_day+=1

        self.episode_steps+=1
        done=(self.episode_steps>=self.max_steps)

        # 构造 obs & info
        obs=self._get_obs()
        prev_infect=np.sum(prev_states[:,1:4])
        curr_infect=np.sum(curr_states[:,1:4])
        infs=max(0,curr_infect - prev_infect)
        prev_d=np.sum(prev_states[:,5])
        curr_d=np.sum(curr_states[:,5])
        deths=max(0,curr_d - prev_d)
        info={
            'infections':float(infs),
            'deaths':float(deths)
        }
        info.update(extra_info)
        return obs, float(rew), done, info

    def _get_obs(self):
        arr=[]
        for i in range(self.num):
            arr.append(self.seir_env.network.nodes[i].state)
        arr=np.concatenate(arr,axis=0)
        return np.concatenate([arr,[self.current_day,self.current_hour]]).astype(np.float32)

    def _process_action(self, action):
        mat=action.reshape(self.num, self.num)
        mat=np.maximum(mat,0)
        row_sum=mat.sum(axis=1,keepdims=True)
        mat=mat/(row_sum+1e-12)
        for i in range(self.num):
            if mat[i,i]<MIN_SELF_ACTIVITY:
                deficit=MIN_SELF_ACTIVITY - mat[i,i]
                non_diag=1.0 - mat[i,i]
                if non_diag>0:
                    scale=(non_diag-deficit)/non_diag
                    for j in range(self.num):
                        if j!=i:
                            mat[i,j]*=scale
                mat[i,i]=MIN_SELF_ACTIVITY
        return mat

# =======================================================
# 传统策略、离散模板与多目标奖励
# =======================================================
def threshold_policy(env):
    states = np.array([env.seir_env.network.nodes[i].state for i in range(env.num)])
    denom = np.sum(states[:,:5], axis=1) + 1e-12
    infection_rates = np.sum(states[:,1:4], axis=1)/denom
    matrix = np.copy(env.original_matrix)
    for i in range(env.num):
        if infection_rates[i]>0.05:
            matrix[i,:]*=0.3
        elif infection_rates[i]>0.02:
            matrix[i,:]*=0.6
    row_sum=matrix.sum(axis=1,keepdims=True)
    matrix=matrix/(row_sum+1e-12)
    for i in range(env.num):
        if matrix[i,i]<MIN_SELF_ACTIVITY:
            deficit=MIN_SELF_ACTIVITY - matrix[i,i]
            non_diag=1.0 - matrix[i,i]
            if non_diag>0:
                scale=(non_diag-deficit)/non_diag
                for j in range(env.num):
                    if j!=i:
                        matrix[i,j]*=scale
            matrix[i,i]=MIN_SELF_ACTIVITY
    return matrix

def discrete_action_map():
    discrete_templates={
        'normal': lambda mat: mat,
        'reduce_50': lambda mat: 0.5*mat + 0.5*np.eye(mat.shape[0]),
        'reduce_80': lambda mat: 0.2*mat + 0.8*np.eye(mat.shape[0]),
        'lockdown': lambda mat: np.eye(mat.shape[0]),
        'neighbor_only': lambda mat: (mat>0)*0.3 + 0.7*np.eye(mat.shape[0])
    }
    def apply_template_to_regions(original_matrix, template_name, regions):
        matrix=original_matrix.copy()
        template_func=discrete_templates[template_name]
        for region in regions:
            row=matrix[region:region+1,:]
            modified_row=template_func(row)
            matrix[region:region+1,:]=modified_row
            row_sum=matrix[region:region+1,:].sum()
            if row_sum>0:
                matrix[region:region+1,:]/=row_sum
        return matrix
    return discrete_templates, apply_template_to_regions

def load_multi_objective_rewards():
    """
    若 ./configs/reward_functions.json 存在则读取；否则提供三组默认权重，
    并返回一个占位的 adaptive 函数。
    """
    path="./configs/reward_functions.json"
    if not os.path.exists(path):
        reward_configs=[
            {'name':'health_priority',   'infection_weight':-2.0, 'death_weight':-5.0, 'commute_weight':0.3},
            {'name':'balanced',          'infection_weight':-1.0, 'death_weight':-3.0, 'commute_weight':0.5},
            {'name':'economic_priority', 'infection_weight':-0.5, 'death_weight':-1.5,'commute_weight':1.0},
        ]
        def dummy_adaptive_func(env,prev_states,curr_states,matrix,time_step):
            return -1.0, {}
        return reward_configs, dummy_adaptive_func
    else:
        with open(path,'r') as f:
            data=json.load(f)
        def adaptive_reward_func(env, prev_states, curr_states, matrix, time_step):
            return 0.0, {}
        return data, adaptive_reward_func

def compute_multi_objective_reward(prev_states, curr_states, matrix, env,
                                   reward_mode='balanced',
                                   time_step=0,
                                   adaptive_func=None):
    if reward_mode=='adaptive' and adaptive_func is not None:
        rew, info = adaptive_func(env, prev_states, curr_states, matrix, time_step)
        return rew, info

    if not hasattr(compute_multi_objective_reward,'_loaded'):
        _cfgs, _ = load_multi_objective_rewards()
        compute_multi_objective_reward._loaded=True
        compute_multi_objective_reward._cfgs=_cfgs
    configs=compute_multi_objective_reward._cfgs

    rcfg = next((c for c in configs if c['name']==reward_mode), None)
    if not rcfg:
        rcfg={'infection_weight':-1.0,'death_weight':-3.0,'commute_weight':0.5}

    w_inf=rcfg.get('infection_weight',-1.0)
    w_dth=rcfg.get('death_weight',-3.0)
    w_cmt=rcfg.get('commute_weight',0.5)

    prev_infect=np.sum(prev_states[:,1:4])
    curr_infect=np.sum(curr_states[:,1:4])
    new_inf = max(0, curr_infect - prev_infect)

    prev_d=np.sum(prev_states[:,5]); curr_d=np.sum(curr_states[:,5])
    new_d=max(0,curr_d - prev_d)

    ori_c=np.sum(env.origin_matrix)-np.trace(env.origin_matrix)
    now_c=np.sum(matrix)-np.trace(matrix)
    ratio= now_c/(ori_c+1e-12)
    total_pop=np.sum(curr_states[:,:5])

    reward= w_inf*new_inf + w_dth*new_d + w_cmt*total_pop*ratio
    info={'infection_w':w_inf,'death_w':w_dth,'commute_w':w_cmt}
    return reward, info

# =======================================================
# 训练与评估
# =======================================================
class CustomCallback(BaseCallback):
    def __init__(self, exp_name="", log_interval=2000, verbose=1):
        super().__init__(verbose)
        self.exp_name=exp_name
        self.log_interval=log_interval
        self.start_time=None
        self.last_save=None

    def _on_training_start(self):
        self.start_time=time.time()
        self.last_save=self.start_time
        print(f"[{self.exp_name}] start training...")

    def _on_step(self):
        if self.n_calls%self.log_interval==0:
            steps=self.model.num_timesteps
            total_steps=self.model._total_timesteps
            pct=steps/total_steps*100 if total_steps>0 else 0
            elapsed=time.time()-self.start_time
            remain=elapsed/(steps+1e-9)*(total_steps-steps) if steps>0 else 0
            print(f"[{self.exp_name}] step {steps}/{total_steps}({pct:.1f}%), elapsed={elapsed/60:.1f}m, ETA={remain/60:.1f}m")
            if time.time()-self.last_save>600:
                ckpt=f"./models/{self.exp_name}_step{steps}.zip"
                self.model.save(ckpt)
                self.last_save=time.time()
                print(f"[{self.exp_name}] checkpoint => {ckpt}")
        return True

def make_env(n_envs=1, seeds=[0], env_kwargs=None):
    def env_fn(seed):
        def _thunk():
            env=CommuneMatrixEnv(**(env_kwargs or {}))
            env.seed(seed)
            return env
        return _thunk
    if n_envs==1:
        return DummyVecEnv([env_fn(seeds[0])])
    else:
        return SubprocVecEnv([env_fn(s) for s in seeds])

def train_model(exp_cfg):
    algo=exp_cfg['algo']
    exp_id=exp_cfg['exp_id']
    gpu=exp_cfg.get('gpu_id',0)
    days=exp_cfg.get('days',20)
    is_states=exp_cfg.get('is_states',False)
    tsteps=exp_cfg.get('total_timesteps',500_000)
    reward_mode=exp_cfg.get('reward_mode','balanced')
    use_adaptive=exp_cfg.get('use_adaptive',False)
    use_threshold=exp_cfg.get('use_threshold',False)
    use_discrete=exp_cfg.get('use_discrete',False)

    os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu)

    env_kwargs={
        'days':days,
        'is_states':is_states,
        'reward_mode':reward_mode,
        'use_adaptive':use_adaptive,
        'use_threshold_policy':use_threshold,
        'use_discrete':use_discrete
    }
    n_envs=1
    seed_base=10000+exp_id
    vec_env=make_env(n_envs, [seed_base], env_kwargs)
    set_random_seed(seed_base); np.random.seed(seed_base); torch.manual_seed(seed_base)

    exp_name=f"{algo}_exp{exp_id}_{reward_mode}_{'adaptive' if use_adaptive else 'fixed'}"
    cb=CustomCallback(exp_name=exp_name, log_interval=2000)

    if algo=='PPO':
        default_ppo={'learning_rate':3e-4,'batch_size':128,'ent_coef':0.01,'n_steps':2048,'n_epochs':10,'clip_range':0.2}
        user_args=exp_cfg.get('ppo_args',{})
        if 'weight_set' in user_args:
            user_args={k:v for k,v in user_args.items() if k!='weight_set'}
        default_ppo.update(user_args)
        model=PPO("MlpPolicy", vec_env, device="cuda", verbose=0, tensorboard_log=f"./tensorboard/{exp_name}", **default_ppo)
    elif algo=='A2C':
        default_a2c={'learning_rate':7e-4,'ent_coef':0.01,'vf_coef':0.5,'n_steps':5,'max_grad_norm':0.5}
        user_args=exp_cfg.get('a2c_args',{})
        if 'weight_set' in user_args:
            user_args={k:v for k,v in user_args.items() if k!='weight_set'}
        default_a2c.update(user_args)
        model=A2C("MlpPolicy", vec_env, device="cuda", verbose=0, tensorboard_log=f"./tensorboard/{exp_name}", **default_a2c)
    else:
        raise ValueError("Unsupported algo")

    model.learn(total_timesteps=tsteps, callback=cb, progress_bar=True)
    final_path=f"./models/{exp_name}_final.zip"
    model.save(final_path)
    vec_env.close()
    print(f"[{exp_name}] training done => {final_path}")
    return final_path

def evaluate_model(model_path, algo, exp_name, n_episodes=1, days=20,
                   is_states=False, reward_mode='balanced',
                   use_adaptive=False, use_threshold=False,
                   use_discrete=False):
    if algo=='PPO':
        model=PPO.load(model_path)
    elif algo=='A2C':
        model=A2C.load(model_path)
    else:
        raise ValueError("Unsupported algo")

    results={'method':exp_name,'episodes':[]}

    for ep in range(n_episodes):
        env=CommuneMatrixEnv(days=days,is_states=is_states,reward_mode=reward_mode,
                             use_adaptive=use_adaptive,use_threshold_policy=use_threshold,
                             use_discrete=use_discrete)
        obs=env.reset()
        done=False
        ep_data={
            'hourly_infections':[],
            'hourly_deaths':[],
            'hourly_commute_ratio':[],
            'hourly_reward':[],
            'district_infections':[],
            'policy_matrices':[]
        }
        dist_infs=[[] for _ in range(env.num)]
        step_count=0
        ep_reward=0.0

        while not done:
            if not use_threshold:
                action, _=model.predict(obs, deterministic=True)
            else:
                action=np.zeros(env.action_space.shape)
            obs, reward, done, info=env.step(action)

            ep_reward+=float(reward)
            ep_data['hourly_reward'].append(float(reward))
            ep_data['hourly_infections'].append(float(info.get('infections',0.0)))
            ep_data['hourly_deaths'].append(float(info.get('deaths',0.0)))

            mat=env.seir_env.matrix
            ori_c=np.sum(env.original_matrix)-np.trace(env.original_matrix)
            now_c=np.sum(mat)-np.trace(mat)
            c_ratio= now_c/(ori_c+1e-12)
            ep_data['hourly_commute_ratio'].append(float(c_ratio))

            for d in range(env.num):
                eij = env.seir_env.network.nodes[d].state[1:4].sum()
                dist_infs[d].append(float(eij))

            if step_count%24==0:
                ep_data['policy_matrices'].append(mat.copy().tolist())

            step_count+=1

        ep_data['total_reward']=float(ep_reward)
        ep_data['district_infections']=dist_infs
        results['episodes'].append(ep_data)

    out_dir=f"./results/{exp_name}/"
    os.makedirs(out_dir, exist_ok=True)
    json_path=os.path.join(out_dir,"evaluation.json")
    with open(json_path,"w") as f:
        json.dump(results,f,indent=2)

    # 简单时序图（最后一条 episode）
    last_ep = results['episodes'][-1]
    length = len(last_ep['hourly_infections'])
    x_idx = np.arange(length)

    plt.figure(figsize=(12, 8))
    plt.subplot(4, 1, 1)
    plt.plot(x_idx, last_ep['hourly_infections'], label='hourly_infections'); plt.legend(); plt.grid(alpha=0.3)
    plt.subplot(4, 1, 2)
    plt.plot(x_idx, last_ep['hourly_deaths'], label='hourly_deaths'); plt.legend(); plt.grid(alpha=0.3)
    plt.subplot(4, 1, 3)
    plt.plot(x_idx, last_ep['hourly_commute_ratio'], label='hourly_commute_ratio'); plt.legend(); plt.grid(alpha=0.3)
    plt.subplot(4, 1, 4)
    plt.plot(x_idx, last_ep['hourly_reward'], label='hourly_reward'); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "evaluation_timeseries.png"))
    plt.close()

    print(f"[{exp_name}] Evaluate done => {json_path}")
    return results

# -------- baseline 策略（可选运行，以输出 baseline_* 目录，便于对比） --------
BASELINE_STRATEGIES = {}

def baseline_original(env):
    return env.origin_matrix.copy()

def baseline_diagonal(env):
    mat = np.zeros_like(env.origin_matrix); np.fill_diagonal(mat, 1.0); return mat

def baseline_neighbor(env):
    mat = (env.origin_matrix > 1e-12).astype(float)
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

def evaluate_baseline(baseline_name, days=20, n_episodes=1, is_states=False,
                      reward_mode='balanced', use_adaptive=False, use_discrete=False):
    out_dir = f"./results/baseline_{baseline_name}/"
    os.makedirs(out_dir, exist_ok=True)
    results = {'method': f"baseline_{baseline_name}", 'episodes': []}

    for ep in range(n_episodes):
        env = CommuneMatrixEnv(days=days, is_states=is_states,
                               reward_mode=reward_mode, use_adaptive=use_adaptive,
                               use_threshold_policy=True if baseline_name=='diagonal' else False,
                               use_discrete=use_discrete)
        obs = env.reset()
        done = False
        ep_data = {'hourly_infections': [], 'hourly_deaths': [], 'hourly_commute_ratio': [], 'hourly_reward': []}
        ep_reward = 0.0
        fix_mat = BASELINE_STRATEGIES[baseline_name](env.seir_env)

        while not done:
            action = fix_mat.flatten()
            obs, rew, done, info = env.step(action)
            ep_reward += float(rew)

            mat = env.seir_env.matrix
            orig_c = np.sum(env.original_matrix) - np.trace(env.original_matrix)
            now_c = np.sum(mat) - np.trace(mat)
            commute_ratio = now_c / (orig_c + 1e-12)

            ep_data['hourly_infections'].append(float(info.get('infections',0.0)))
            ep_data['hourly_deaths'].append(float(info.get('deaths',0.0)))
            ep_data['hourly_commute_ratio'].append(float(commute_ratio))
            ep_data['hourly_reward'].append(float(rew))

        ep_data['total_reward'] = float(ep_reward)
        results['episodes'].append(ep_data)

    json_path = os.path.join(out_dir, "baseline_evaluation.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # 最后一条 episode 的时序图
    last_ep = results['episodes'][-1]
    length = len(last_ep['hourly_infections'])
    x_idx = np.arange(length)
    plt.figure(figsize=(12, 8))
    plt.subplot(4, 1, 1); plt.plot(x_idx, last_ep['hourly_infections'], label='hourly_infections'); plt.legend(); plt.grid(alpha=0.3)
    plt.subplot(4, 1, 2); plt.plot(x_idx, last_ep['hourly_deaths'], label='hourly_deaths'); plt.legend(); plt.grid(alpha=0.3)
    plt.subplot(4, 1, 3); plt.plot(x_idx, last_ep['hourly_commute_ratio'], label='hourly_commute_ratio'); plt.legend(); plt.grid(alpha=0.3)
    plt.subplot(4, 1, 4); plt.plot(x_idx, last_ep['hourly_reward'], label='hourly_reward'); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "baseline_timeseries.png")); plt.close()
    print(f"Baseline {baseline_name} evaluate done => {json_path}")

# =======================================================
# 并行封装 & CLI
# =======================================================
def parallel_train_and_evaluate(exp_configs):
    procs = []
    for cfg in exp_configs:
        p = multiprocessing.Process(target=run_single_experiment, args=(cfg,))
        p.start(); procs.append(p); time.sleep(2)
    for p in procs:
        p.join()
    print("All train+eval done.")

def run_single_experiment(cfg):
    model_path = train_model(cfg)
    if model_path:
        exp_name = f"{cfg['algo']}_exp{cfg['exp_id']}_{cfg['reward_mode']}_{'adaptive' if cfg.get('use_adaptive', False) else 'fixed'}"
        evaluate_model(model_path, cfg['algo'], exp_name, n_episodes=1, days=cfg.get('days', 20),
                       is_states=cfg.get('is_states', False), reward_mode=cfg.get('reward_mode', 'balanced'),
                       use_adaptive=cfg.get('use_adaptive', False), use_threshold=cfg.get('use_threshold', False),
                       use_discrete=cfg.get('use_discrete', False))

def load_diversified_parameters(json_file="./configs/diversified_parameters.json"):
    if not os.path.exists(json_file):
        data={
            "ppo_configs":[
                {"learning_rate":3e-4, "batch_size":128, "ent_coef":0.01, "n_steps":2048, "n_epochs":10, "clip_range":0.2, "weight_set":"balanced"},
                {"learning_rate":1e-4, "batch_size":64,  "ent_coef":0.001,"n_steps":4096, "n_epochs":15, "clip_range":0.1, "weight_set":"health_priority"}
            ],
            "a2c_configs":[
                {"learning_rate":7e-4,  "ent_coef":0.01, "vf_coef":0.5, "n_steps":5,  "max_grad_norm":0.5, "weight_set":"balanced"},
                {"learning_rate":3e-3,  "ent_coef":0.1,  "vf_coef":0.7, "n_steps":10, "max_grad_norm":1.0, "weight_set":"economic_priority"}
            ]
        }
        with open(json_file,'w') as f:
            json.dump(data,f,indent=2)
        return data["ppo_configs"], data["a2c_configs"]
    else:
        with open(json_file,'r') as f:
            loaded=json.load(f)
        return loaded["ppo_configs"], loaded["a2c_configs"]

def prepare_experiment_configs(n_exps=2, days=20,
                               reward_mode='balanced',
                               use_adaptive=False,
                               use_threshold=False,
                               use_discrete=False):
    ppo_cfgs, a2c_cfgs = load_diversified_parameters()
    ppo_cfgs=ppo_cfgs[:n_exps]
    a2c_cfgs=a2c_cfgs[:n_exps]
    exp_configs=[]
    exp_id=1
    gpu=0
    for pcfg in ppo_cfgs:
        conf={'algo':'PPO','exp_id':exp_id,'gpu_id':gpu,'days':days,'is_states':False,
              'total_timesteps':300_000,'reward_mode':reward_mode,'use_adaptive':use_adaptive,
              'use_threshold':use_threshold,'use_discrete':use_discrete,'ppo_args':pcfg}
        exp_configs.append(conf); exp_id+=1; gpu=(gpu+1)%max(1,NUM_GPUS)
    for acfg in a2c_cfgs:
        conf={'algo':'A2C','exp_id':exp_id,'gpu_id':gpu,'days':days,'is_states':False,
              'total_timesteps':300_000,'reward_mode':reward_mode,'use_adaptive':use_adaptive,
              'use_threshold':use_threshold,'use_discrete':use_discrete,'a2c_args':acfg}
        exp_configs.append(conf); exp_id+=1; gpu=(gpu+1)%max(1,NUM_GPUS)
    return exp_configs

def main():
    parser=argparse.ArgumentParser(description="SEIJRD multi-objective RL with 717 dynamics - main_RL_910.py")
    parser.add_argument("--mode",type=str,choices=["train_eval","baseline","visualize","all"],default="train_eval")
    parser.add_argument("--days",type=int,default=20)
    parser.add_argument("--exp-count",type=int,default=2)
    parser.add_argument("--reward-mode",type=str,default="balanced")
    parser.add_argument("--use-adaptive",action="store_true")
    parser.add_argument("--use-threshold",action="store_true")
    parser.add_argument("--use-discrete",action="store_true")
    args=parser.parse_args()

    if args.mode=="train_eval" or args.mode=="all":
        exp_cfgs=prepare_experiment_configs(n_exps=args.exp_count, days=args.days,
                                            reward_mode=args.reward_mode,
                                            use_adaptive=args.use_adaptive,
                                            use_threshold=args.use_threshold,
                                            use_discrete=args.use_discrete)
        parallel_train_and_evaluate(exp_cfgs)

    if args.mode=="baseline" or args.mode=="all":
        for b in ["original","diagonal","neighbor","random50"]:
            evaluate_baseline(baseline_name=b, days=args.days, n_episodes=1,
                              is_states=False, reward_mode=args.reward_mode,
                              use_adaptive=args.use_adaptive, use_discrete=args.use_discrete)

    if args.mode=="visualize" or args.mode=="all":
        try:
            from advanced_visuals import run_all_visualization_improvements
            run_all_visualization_improvements("./results")
        except Exception as e:
            print("advanced_visuals not found or failed. You can still inspect ./results/*.json")

if __name__=="__main__":
    main()
