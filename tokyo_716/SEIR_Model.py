from enum import Enum
import random
from typing import List, Dict
import matplotlib.pyplot as plt
import numpy as np
import time
import json
import datetime

file_path = "data/end_copy.json"
NUM = 50
paras = {
    "beta": 0.28,
    "gamma": 0.1,
    "sigma": 0.1,
    "mu": 0.01,
    "rho": 1e-4 # 接触传染概率
}

def process_competitions(data_file):
    # 定义槽位与时间段的映射：
    # slot 0: 早上 8:00 到 11:00（3 小时）
    # slot 1: 下午 13:00 到 17:00（4 小时）
    # slot 2: 晚上 19:00 到 22:00（3 小时）
    slot_mapping = {
        0: (8, 11),
        1: (13, 17),
        2: (19, 22)
    }

    # 假定所有时间都在同一年（比如 2023 年）
    year = 2021

    # 遍历所有记录，找到最早的日期
    earliest_date = None
    for venue in data_file:
        for t in venue["time"]:
            event_date = datetime.date(year, t["month"], t["day"])
            if earliest_date is None or event_date < earliest_date:
                earliest_date = event_date

    # 以 earliest_date 的 0 点作为 tick0
    # 对于每个事件，计算距离 tick0 的小时数
    competitions = []
    for venue in data_file:
        venue_id = venue["venue_id"]
        # 将容量字符串中的逗号去掉后转换为数字
        capacity = int(venue["capacity"].replace(',', ''))
        for t in venue["time"]:
            # 计算事件日期
            event_date = datetime.date(year, t["month"], t["day"])
            days_diff = (event_date - earliest_date).days
            # 根据槽位获取对应的开始和结束小时
            slot = t["slot"]
            start_hour, end_hour = slot_mapping[slot]
            # tick 值：从 tick0 开始，每小时为一 tick
            start_tick = days_diff * 24 + start_hour
            end_tick = days_diff * 24 + end_hour
            competitions.append((venue_id, start_tick, end_tick, capacity))

    return competitions

def process_places(data_file):
    places = {}
    for data_point in data_file:
        venue = {
            "venue_id": data_point["venue_id"],
            "venue_name": data_point["venue_name"],
            "capacity": int(data_point["capacity"].replace(',', ''))
        }
        places[data_point["venue_id"]] = Place(venue)
    
    return places

class Place:
    def __init__(self, data: Dict):
        self.id = data["venue_id"]
        self.name = data["venue_name"]
        self.capacity = data["capacity"]
        self.agenda = []

        self.endtime = -1

        self.audience_from = []
    
    def infect(self):
        SEIR = np.sum(self.audience_from, axis = 0)
        infectious_num = sum(SEIR[1:3])
        probability = 1 - (1 - paras["rho"]) ** infectious_num

        new_infected = SEIR[0] * probability
        
        susceptible = self.audience_from[:, 0]
        distribution = susceptible / np.sum(susceptible)

        new_infected_by_row = distribution * new_infected
        self.audience_from[:, 0] -= new_infected_by_row
        self.audience_from[:, 1] += new_infected_by_row
        
class Env:
    def __init__(self):
        self.places = {}
        self.competitions = []
        self.current_tick = 0
        self.network = None
        self.capacity_strategy = None

    def init_env(self):
        competitions_data = json.load(open(file_path, "r", encoding = 'utf-8'))
        self.competitions = process_competitions(competitions_data)
        self.places = process_places(competitions_data)

        for competition in self.competitions:
            place_id = competition[0]
            self.places[place_id].agenda.append(competition)
        
        for place in self.places.values():
            place.agenda.sort(key = lambda x: x[1])

        self.network = SEIR_Network(NUM, [np.array([92200, 800, 0, 0, 0])] * NUM)
    
    def check_competition_start(self):
        for id, place in self.places.items():
            if len(place.agenda) == 0:
                continue
            competition = place.agenda[0]
            if competition[1] == self.current_tick:
                place.audience_from = np.zeros((self.network.node_num, 4))
                place.agenda.pop(0)
                place.endtime = competition[2]
                
                # 从 SEIR 网络中随机选择一部分人群到场
                N = []
                for i in range(self.network.node_num):
                    N.append(sum(self.network.nodes[i].state[:4]))
                N = np.array(N)
                
                for i in range(self.network.node_num):
                    try:
                        place.audience_from[i] = np.array(self.network.nodes[i].state[:4]) * competition[3] / sum(N) * self.capacity_strategy[id - 1]
                    except:
                        place.audience_from[i] = np.array(self.network.nodes[i].state[:4]) * competition[3] / sum(N)
                        print(id)
                    self.network.nodes[i].state[:4] -= place.audience_from[i]
                
                place.infect()


    def check_competition_end(self):
        for place in self.places.values():
            if place.endtime == self.current_tick:
                for i in range(self.network.node_num):
                    self.network.nodes[i].state[:4] += place.audience_from[i]
                place.audience_from = []
                place.endtime = -1

class SEIR_Node:
    def __init__(self, state, total_nodes):
        self.total_nodes = total_nodes
        self.state = state.astype(np.float64)
        self.from_node = np.zeros((total_nodes, 5))
    
    def update_seir(self, delta_time = 0.01, times = 100):
        N = sum(self.state)
        SEIRD = self.state.copy()

        N += np.sum(self.from_node)
        SEIRD += np.sum(self.from_node, axis = 0)
        # print(SEIRD)

        if N > 0:
            dS = - paras["beta"] * SEIRD[0] * SEIRD[1] / N
            dE = paras["beta"] * SEIRD[0] * SEIRD[1] / N - paras["sigma"] * SEIRD[1]
            dI = paras["sigma"] * SEIRD[1] - paras["gamma"] * SEIRD[2] - paras["mu"] * SEIRD[2]
            dR = paras["gamma"] * SEIRD[2]
            dD = paras["mu"] * SEIRD[2]
        else:
            dS = 0.0
            dE = 0.0
            dI = 0.0
            dR = 0.0
            dD = 0.0
        
        Nself = np.sum(self.state[:4])
        self.state += np.array([dS, dE, dI, dR, dD]) * (Nself / N) * delta_time * times

        Nvalue = np.sum(self.from_node, axis = 1)
        self.from_node += (np.array([dS, dE, dI, dR, dD])[None, :]) * ((Nvalue / N)[:, None]) * delta_time * times
        
    def plot(self):
        plt.plot(self.state)
        plt.show()

class SEIR_Network:
    def __init__(self, num, states):
        self.node_num = num
        self.nodes = {}
        for i in range(num):
            self.nodes[i] = SEIR_Node(states[i], num)

        self.A = np.random.random((num, num))
        self.A = self.A / np.sum(self.A, axis = 1, keepdims = True)
        self.delta_time = 1 / 2400

    def morning_commuting(self):
        for i in range(self.node_num):
            SEIRD = self.nodes[i].state.copy()
            for j in range(self.node_num):
                if i == j: 
                    continue
                a_ij = self.A[i][j]
                self.nodes[j].from_node[i] = SEIRD * a_ij
                self.nodes[i].state -= self.nodes[j].from_node[i]

    def evening_commuting(self):
        for i in range(self.node_num):
            for j in range(self.node_num):
                if i == j:
                    continue
                self.nodes[j].state += self.nodes[i].from_node[j]
            
            self.nodes[i].from_node = np.zeros((self.node_num, 5))

    def update_network(self):
        for i in range(self.node_num):
            node = self.nodes[i]
            node.update_seir(self.delta_time, 100)

def RL_strategy(Env, data):
    return [1.0 for place in Env.places]

if __name__ == "__main__":

    np.random.seed(0)
    env = Env()
    env.init_env()
    
    states_history = []

    #### 强化学习策略 ####
    env.capacity_strategy = RL_strategy(env, states_history)
    #####################

    for day in range(19):
        for hour in range(24):
            tick = day * 24 + hour
            start_time = time.time()
            env.current_tick = tick
            env.check_competition_start()
            if tick % 24 == 8:
                env.network.morning_commuting()
            
            env.network.update_network()

            if tick % 24 == 18:
                env.network.evening_commuting()
            env.check_competition_end()
            end_time = time.time()

            if tick % 12 == 0:
                SEIRD = np.zeros((NUM, 5))
                for i in range(NUM):
                    SEIRD[i] = env.network.nodes[i].state.copy()
                    for j in range(NUM):
                        if i == j:
                            continue
                        SEIRD[i] += env.network.nodes[j].from_node[i]
                    if i == 0:
                        print("tick: ", tick, "time: ", end_time - start_time, np.array2string(SEIRD[i], formatter = {'float_kind': lambda x: f"{x:.2f}"}))
                
                states_history.append(SEIRD.copy())

        #### 强化学习策略 ####
        env.capacity_strategy = RL_strategy(env, states_history)
        #####################

    states_history = np.array(states_history)
    node0_history = np.vstack([state[0] for state in states_history])
    for i in range(5):
        plt.plot(node0_history[:,i])

    plt.legend(["S", "E", "I", "R", "D"])
    plt.savefig(f"plot/{paras['rho']}_all.png")
    
        
