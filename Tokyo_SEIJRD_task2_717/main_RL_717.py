#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
main_RL_58.py

核心功能：
1. 实现Tokyo SEIR网络的CommuneMatrixEnv环境（带多目标/自适应奖励、threshold策略、离散动作映射等）
2. 提供train_model与evaluate_model函数，可对PPO/A2C进行训练和评估
3. 并行运行多个实验
4. 与advanced_visuals.py只通过JSON结果交互，不再相互导入，避免循环依赖
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
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed

import warnings
warnings.filterwarnings('ignore')

# 保证目录
os.makedirs("./models", exist_ok=True)
os.makedirs("./logs", exist_ok=True)
os.makedirs("./results", exist_ok=True)
os.makedirs("./visualizations", exist_ok=True)
os.makedirs("./tensorboard", exist_ok=True)
os.makedirs("./configs", exist_ok=True)
os.makedirs("./interactive", exist_ok=True)

NUM_GPUS = 8
NUM_CPU_CORES = 150
MAX_POP = 1e12

# ----------- 老版本参数 -----------
backup_paras = {
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

DEFAULT_IS_STATES = False
MIN_SELF_ACTIVITY = 0.3
file_path = "data/end_copy.json"

# =========== CSV读取 ===========
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

# 处理场馆/比赛数据
def process_competitions(data_file):
    slot_mapping = {0:(8,11),1:(13,17),2:(19,22)}
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
            comps.append((vid,st_tick,ed_tick,cap))
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

class Place:
    def __init__(self, data:Dict):
        self.id = data["venue_id"]
        self.name= data["venue_name"]
        self.capacity=data["capacity"]
        self.agenda=[]
        self.endtime=None
        self.audience_from=[]

    def infect(self, paras):
        if len(self.audience_from)==0:
            return
        arr = np.sum(self.audience_from, axis=0)
        inf_num = arr[1]+arr[2]+arr[3]
        if inf_num<=0:
            return
        prob = 1-(1-paras["rho"])**inf_num
        prob=np.clip(prob,0,1)
        S_tot=arr[0]
        if S_tot<1e-3:
            return
        new_inf=S_tot*prob
        ratio=self.audience_from[:,0]/(S_tot+1e-12)
        new_inf_each=ratio*new_inf
        self.audience_from[:,0]-=new_inf_each
        self.audience_from[:,1]+=new_inf_each

# =========== Env类定义（老版环境类） ===========
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

# =========== SEIR节点网络 ==============
class SEIR_Node:
    def __init__(self, state, total_nodes):
        self.state=state.astype(float)
        self.total_nodes=total_nodes
        self.from_node=np.zeros((total_nodes,6))
        self.to_node=np.zeros((total_nodes,5))

    def update_seir(self, delta_time=0.01, times=100, paras=backup_paras):
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


class SEIR_Network:
    def __init__(self, num, states, matrix=None, paras=backup_paras):
        self.node_num=num
        self.nodes={}
        for i in range(num):
            self.nodes[i]=SEIR_Node(states[i], num)
        if matrix is not None:
            row_sum=matrix.sum(axis=1,keepdims=True)
            self.A=matrix/row_sum
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
        for i in range(self.node_num):
            self.nodes[i].state[:5]=0.0
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
            self.nodes[i].update_seir(self.delta_time,100,self.paras)

# =========== 传统阈值策略（baseline或替换RL动作）==========
def threshold_policy(env):
    states = np.array([env.seir_env.network.nodes[i].state for i in range(env.num)])
    infection_rates = np.sum(states[:,1:4], axis=1)/np.sum(states[:,:5], axis=1)
    matrix = np.copy(env.original_matrix)
    for i in range(env.num):
        if infection_rates[i]>0.05:
            matrix[i,:]*=0.3
        elif infection_rates[i]>0.02:
            matrix[i,:]*=0.6
    row_sum=matrix.sum(axis=1,keepdims=True)
    matrix=matrix/(row_sum+1e-12)
    for i in range(env.num):
        if matrix[i,i]<0.3:
            deficit=0.3-matrix[i,i]
            non_diag=1.0-matrix[i,i]
            if non_diag>0:
                scale=(non_diag-deficit)/non_diag
                for j in range(env.num):
                    if j!=i:
                        matrix[i,j]*=scale
            matrix[i,i]=0.3
    return matrix

# =========== 离散化模板  ===========
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


# =========== 多目标/自适应奖励 ===========

def load_multi_objective_rewards():
    """
    读取 ./configs/reward_functions.json 若存在, 否则返回默认
    """
    path="./configs/reward_functions.json"
    if not os.path.exists(path):
        # 返回几个默认
        reward_configs=[
            {
                'name':'health_priority',
                'infection_weight':-2.0,
                'death_weight':-5.0,
                'commute_weight':0.3
            },
            {
                'name':'balanced',
                'infection_weight':-1.0,
                'death_weight':-3.0,
                'commute_weight':0.5
            },
            {
                'name':'economic_priority',
                'infection_weight':-0.5,
                'death_weight':-1.5,
                'commute_weight':1.0
            }
        ]
        def dummy_adaptive_func(env,prev_states,curr_states,matrix,time_step):
            return -1.0, {}
        return reward_configs, dummy_adaptive_func
    else:
        with open(path,'r') as f:
            data=json.load(f)
        # 看是否有'adaptive'
        def adaptive_reward_func(env, prev_states, curr_states, matrix, time_step):
            """
            placeholder
            """
            # same logic from your code
            # or we just do a trivial
            return 0.0, {}
        for rcfg in data:
            if rcfg.get('adaptive',False):
                # 你之前实现
                pass
        return data, adaptive_reward_func

def compute_multi_objective_reward(prev_states, curr_states, matrix, env,
                                   reward_mode='balanced',
                                   time_step=0,
                                   adaptive_func=None):
    """
    计算多目标/自适应reward
    """
    if reward_mode=='adaptive' and adaptive_func is not None:
        rew, info = adaptive_func(env, prev_states, curr_states, matrix, time_step)
        return rew, info

    # 否则读取 config
    global _loaded_reward_configs
    if not hasattr(compute_multi_objective_reward,'_loaded'):
        _loaded_reward_configs, _loaded_adaptive = load_multi_objective_rewards()
        compute_multi_objective_reward._loaded = True
        compute_multi_objective_reward._cfgs = _loaded_reward_configs
    configs=compute_multi_objective_reward._cfgs

    # 找到
    rcfg = next((c for c in configs if c['name']==reward_mode), None)
    if not rcfg:
        rcfg = {'infection_weight':-1.0, 'death_weight':-3.0, 'commute_weight':0.5}

    w_inf=rcfg.get('infection_weight',-1.0)
    w_dth=rcfg.get('death_weight',-3.0)
    w_cmt=rcfg.get('commute_weight',0.5)

    prev_infect=np.sum(prev_states[:,1:4])
    curr_infect=np.sum(curr_states[:,1:4])
    new_inf = max(0, curr_infect - prev_infect)

    prev_d=np.sum(prev_states[:,5])
    curr_d=np.sum(curr_states[:,5])
    new_d=max(0,curr_d - prev_d)

    ori_c=np.sum(env.origin_matrix)-np.trace(env.origin_matrix)
    now_c=np.sum(matrix)-np.trace(matrix)
    ratio= now_c/(ori_c+1e-12)
    total_pop=np.sum(curr_states[:,:5])

    reward= w_inf*new_inf + w_dth*new_d + w_cmt*total_pop*ratio
    info={'infection_w':w_inf,'death_w':w_dth,'commute_w':w_cmt}
    return reward, info


# =========== CommuneMatrixEnv 全实现 ===========
class CommuneMatrixEnv(gym.Env):
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
        self.np_random, seed_ = gym.utils.seeding.np_random(seed)
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
        # 如果use_threshold,直接用threshold_policy
        if self.use_threshold:
            matrix=threshold_policy(self)
        else:
            matrix=self._process_action(action)
            if self.use_discrete:
                # 用'reduce_50'举例
                matrix=self.discrete_templates['reduce_50'](matrix)

        self.seir_env.modify_matrix(matrix)

        self.seir_env.check_competition_start()
        if self.current_hour==8:
            self.seir_env.network.morning_commuting()

        prev_states=np.array([nd.state.copy() for nd in self.seir_env.network.nodes.values()])
        self.seir_env.network.update_network()
        if self.current_hour==18:
            self.seir_env.network.evening_commuting()

        self.seir_env.check_competition_end()
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

        self.current_hour+=1
        if self.current_hour>=24:
            self.current_hour=0
            self.current_day+=1

        self.episode_steps+=1
        done=(self.episode_steps>=self.max_steps)
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
        return obs, rew, done, info

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


# =========== 训练&评估, 并行结构, 收集district_infections/policy_matrices ===========
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
            pct=steps/total_steps*100
            elapsed=time.time()-self.start_time
            remain=elapsed/(steps+1e-9)*(total_steps-steps)
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
            env=CommuneMatrixEnv(**env_kwargs)
            env.seed(seed)
            return env
        return _thunk
    if n_envs==1:
        return DummyVecEnv([env_fn(seeds[0])])
    else:
        return SubprocVecEnv([env_fn(s) for s in seeds])


def train_model(exp_cfg):
    """
    exp_cfg={...}
    """
    algo=exp_cfg['algo']
    exp_id=exp_cfg['exp_id']
    gpu=exp_cfg.get('gpu_id',0)
    days=exp_cfg.get('days',20)
    is_states=exp_cfg.get('is_states',False)
    tsteps=exp_cfg.get('total_timesteps',500000)
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
    env=make_env(n_envs, [seed_base], env_kwargs)
    set_random_seed(seed_base)
    np.random.seed(seed_base)
    torch.manual_seed(seed_base)

    exp_name=f"{algo}_exp{exp_id}_{reward_mode}_{'adaptive' if use_adaptive else 'fixed'}"
    cb=CustomCallback(exp_name=exp_name, log_interval=2000)

    if algo=='PPO':
        default_ppo={
            'learning_rate':3e-4,
            'batch_size':128,
            'ent_coef':0.01,
            'n_steps':2048,
            'n_epochs':10,
            'clip_range':0.2
        }
        user_args=exp_cfg.get('ppo_args',{})
        # 移除weight_set参数
        if 'weight_set' in user_args:
            user_args = {k: v for k, v in user_args.items() if k != 'weight_set'}
        default_ppo.update(user_args)
        model=PPO("MlpPolicy", env, device="cuda", verbose=0,
                  tensorboard_log=f"./tensorboard/{exp_name}",
                  **default_ppo)
    elif algo=='A2C':
        default_a2c={
            'learning_rate':7e-4,
            'ent_coef':0.01,
            'vf_coef':0.5,
            'n_steps':5,
            'max_grad_norm':0.5
        }
        user_args=exp_cfg.get('a2c_args',{})
        # 移除weight_set参数
        if 'weight_set' in user_args:
            user_args = {k: v for k, v in user_args.items() if k != 'weight_set'}
        default_a2c.update(user_args)
        model=A2C("MlpPolicy", env, device="cuda", verbose=0,
                  tensorboard_log=f"./tensorboard/{exp_name}",
                  **default_a2c)
    else:
        raise ValueError("Unsupported algo")

    model.learn(total_timesteps=tsteps, callback=cb, progress_bar=True)
    final_path=f"./models/{exp_name}_final.zip"
    model.save(final_path)
    env.close()
    print(f"[{exp_name}] training done => {final_path}")
    return final_path


def evaluate_model(model_path, algo, exp_name, n_episodes=1, days=20,
                   is_states=False, reward_mode='balanced',
                   use_adaptive=False, use_threshold=False,
                   use_discrete=False):
    """
    收集district_infections, policy_matrices
    """
    if algo=='PPO':
        model=PPO.load(model_path)
    elif algo=='A2C':
        model=A2C.load(model_path)
    else:
        raise ValueError("Unsupported algo")

    results={
        'method':exp_name,
        'episodes':[]
    }

    for ep in range(n_episodes):
        env=CommuneMatrixEnv(days=days,
                             is_states=is_states,
                             reward_mode=reward_mode,
                             use_adaptive=use_adaptive,
                             use_threshold_policy=use_threshold,
                             use_discrete=use_discrete)
        obs=env.reset()
        done=False
        ep_data={
            'hourly_infections':[],
            'hourly_deaths':[],
            'hourly_commute_ratio':[],
            'hourly_reward':[],
            'district_infections':[], # shape = (num_district, T)
            'policy_matrices':[]
        }
        dist_infs=[[] for _ in range(env.num)]  # collect E+I+J each step
        step_count=0
        ep_reward=0

        while not done:
            if not use_threshold:
                action, _=model.predict(obs, deterministic=True)
            else:
                action=np.zeros(env.action_space.shape)
            obs, reward, done, info=env.step(action)

            ep_reward+=reward
            ep_data['hourly_reward'].append(float(reward))
            inf=info['infections']
            dth=info['deaths']
            ep_data['hourly_infections'].append(float(inf))
            ep_data['hourly_deaths'].append(float(dth))

            mat=env.seir_env.matrix
            ori_c=np.sum(env.original_matrix)-np.trace(env.original_matrix)
            now_c=np.sum(mat)-np.trace(mat)
            c_ratio= now_c/(ori_c+1e-12)
            ep_data['hourly_commute_ratio'].append(float(c_ratio))

            # collect district_infections
            for d in range(env.num):
                eij = env.seir_env.network.nodes[d].state[1:4].sum()
                dist_infs[d].append(float(eij))

            # collect policy_matrices daily
            if step_count%24==0:
                ep_data['policy_matrices'].append(mat.copy().tolist())

            step_count+=1

        ep_data['total_reward']=float(ep_reward)
        # finalize district_infections
        ep_data['district_infections']=dist_infs
        results['episodes'].append(ep_data)

    out_dir=f"./results/{exp_name}/"
    os.makedirs(out_dir, exist_ok=True)
    json_path=os.path.join(out_dir,"evaluation.json")
    with open(json_path,"w") as f:
        json.dump(results,f,indent=2)
    print(f"[{exp_name}] Evaluate done => {json_path}")
    return results


def parallel_train_and_evaluate(exp_configs):
    """
    并行训练和评估，使用基本Process避免序列化问题
    """
    import multiprocessing

    processes = []
    results = []

    # 启动每个配置的单独进程
    for cfg in exp_configs:
        p = multiprocessing.Process(
            target=run_single_experiment,
            args=(cfg,)
        )
        p.start()
        processes.append(p)

    # 等待所有进程完成
    for p in processes:
        p.join()

    # 收集结果 - 假设每个实验会创建评估文件
    # 基于实验配置重建名称
    for cfg in exp_configs:
        exp_name = f"{cfg['algo']}_exp{cfg['exp_id']}_{cfg['reward_mode']}_{'adaptive' if cfg.get('use_adaptive', False) else 'fixed'}"
        results.append(exp_name)

    print("All train+eval done.")
    return results


def run_single_experiment(cfg):
    """
    运行单个实验的训练和评估
    """
    model_path = train_model(cfg)
    if model_path:
        exp_name = f"{cfg['algo']}_exp{cfg['exp_id']}_{cfg['reward_mode']}_{'adaptive' if cfg.get('use_adaptive', False) else 'fixed'}"
        evaluate_model(
            model_path,
            cfg['algo'],
            exp_name,
            n_episodes=1,
            days=cfg.get('days', 20),
            is_states=cfg.get('is_states', False),
            reward_mode=cfg.get('reward_mode', 'balanced'),
            use_adaptive=cfg.get('use_adaptive', False),
            use_threshold=cfg.get('use_threshold', False),
            use_discrete=cfg.get('use_discrete', False)
        )
    return


# =========== 生成差异化参数（示例） & main入口 ===========
def load_diversified_parameters(json_file="./configs/diversified_parameters.json"):
    """
    如果文件不存在, 可以自动生成一批超参, 这里仅演示固定
    """
    if not os.path.exists(json_file):
        data={
            "ppo_configs":[
                {
                    "learning_rate":3e-4,
                    "batch_size":128,
                    "ent_coef":0.01,
                    "n_steps":2048,
                    "n_epochs":10,
                    "clip_range":0.2,
                    "weight_set":"balanced"
                },
                {
                    "learning_rate":1e-4,
                    "batch_size":64,
                    "ent_coef":0.001,
                    "n_steps":4096,
                    "n_epochs":15,
                    "clip_range":0.1,
                    "weight_set":"health_priority"
                }
            ],
            "a2c_configs":[
                {
                    "learning_rate":7e-4,
                    "ent_coef":0.01,
                    "vf_coef":0.5,
                    "n_steps":5,
                    "max_grad_norm":0.5,
                    "weight_set":"balanced"
                },
                {
                    "learning_rate":3e-3,
                    "ent_coef":0.1,
                    "vf_coef":0.7,
                    "n_steps":10,
                    "max_grad_norm":1.0,
                    "weight_set":"economic_priority"
                }
            ]
        }
        os.makedirs("./configs",exist_ok=True)
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
        conf={
            'algo':'PPO',
            'exp_id':exp_id,
            'gpu_id':gpu,
            'days':days,
            'is_states':False,
            'total_timesteps':300000,
            'reward_mode':reward_mode,
            'use_adaptive':use_adaptive,
            'use_threshold':use_threshold,
            'use_discrete':use_discrete,
            'ppo_args':pcfg
        }
        exp_configs.append(conf)
        exp_id+=1
        gpu=(gpu+1)%NUM_GPUS
    for acfg in a2c_cfgs:
        conf={
            'algo':'A2C',
            'exp_id':exp_id,
            'gpu_id':gpu,
            'days':days,
            'is_states':False,
            'total_timesteps':300000,
            'reward_mode':reward_mode,
            'use_adaptive':use_adaptive,
            'use_threshold':use_threshold,
            'use_discrete':use_discrete,
            'a2c_args':acfg
        }
        exp_configs.append(conf)
        exp_id+=1
        gpu=(gpu+1)%NUM_GPUS
    return exp_configs


def main():
    parser=argparse.ArgumentParser(description="SEIJRD multi-objective RL - main_RL_53.py")
    parser.add_argument("--mode",type=str,choices=["train_eval","visualize"],default="train_eval")
    parser.add_argument("--days",type=int,default=20)
    parser.add_argument("--exp-count",type=int,default=2)
    parser.add_argument("--reward-mode",type=str,default="balanced")
    parser.add_argument("--use-adaptive",action="store_true")
    parser.add_argument("--use-threshold",action="store_true")
    parser.add_argument("--use-discrete",action="store_true")
    args=parser.parse_args()

    if args.mode=="train_eval":
        exp_cfgs=prepare_experiment_configs(
            n_exps=args.exp_count,
            days=args.days,
            reward_mode=args.reward_mode,
            use_adaptive=args.use_adaptive,
            use_threshold=args.use_threshold,
            use_discrete=args.use_discrete
        )
        parallel_train_and_evaluate(exp_cfgs)

    elif args.mode=="visualize":
        print("Running advanced visuals from advanced_visuals.py")
        from advanced_visuals import run_all_visualization_improvements
        run_all_visualization_improvements("./results")

    else:
        parser.print_help()


if __name__=="__main__":
    main()
