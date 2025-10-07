from enum import Enum
import random
from typing import List, Dict
import matplotlib.pyplot as plt
import numpy as np
import time
import json
import datetime
import pandas as pd

file_path = "data/end_copy.json"
backup_paras = {
    "beta": 0.155,
    "gamma_i": 0.079,
    "gamma_j": 0.031,
    "sigma_i": 0.0299,
    "sigma_j": 0.0156,
    "mu_i": 1.95e-5,
    "mu_j": 0.00025,
    "rho": 8.48e-5, # 接触传染概率
    "initial_infect": 500, # 初始感染人数
}

ward_en_mapping = {
        13101: "Chiyoda",
        13102: "Chuo",
        13103: "Minato",
        13104: "Shinjuku",
        13105: "Bunkyo",
        13106: "Taito",
        13107: "Sumida",
        13108: "Koto",
        13109: "Shinagawa",
        13110: "Meguro",
        13111: "Ota",
        13112: "Setagaya",
        13113: "Shibuya",
        13114: "Nakano",
        13115: "Suginami",
        13116: "Toshima",
        13117: "Kita",
        13118: "Arakawa",
        13119: "Itabashi",
        13120: "Nerima",
        13121: "Adachi",
        13122: "Katsushika",
        13123: "Edogawa"
    }

def filter_data(json_data, start_str, end_str):
    """
    从 JSON 数据中提取指定日期范围内的 count
    参数:
        json_data (dict) : 原始 JSON 数据
        start_str (str)  : 起始日期 (格式: "YYYY-MM-DD")
        end_str (str)    : 结束日期 (格式: "YYYY-MM-DD")
    返回:
        list : 包含对应 count 值的列表
    """
    start_date = datetime.datetime.strptime(start_str, "%Y-%m-%d")
    end_date = datetime.datetime.strptime(end_str, "%Y-%m-%d")
    
    filtered = []
    for entry in json_data["data"]:
        entry_date_str = entry["date"].split("T")[0]
        entry_date = datetime.datetime.strptime(entry_date_str, "%Y-%m-%d")
        
        if start_date <= entry_date <= end_date:
            filtered.append(entry["count"])
    
    return filtered

def Input_Population(file_path = "data/tokyo_population.csv"):
    # 假设 CSV 文件有 5 列：行政区代码、日文区名、英文区名、dummy、人口
    pop_df = pd.read_csv(file_path, header = None, 
                         names=["code", "ward_jp", "ward_en", "dummy", "population"])
    pop_df["population"] = pop_df["population"].astype(int)
    return pop_df["population"].tolist()

def Input_Matrix(file_path = "data/tokyo_commuting_flows_with_intra.csv"):
    # 读取 CSV 文件，其中第一列作为行索引，第一行作为列名
    df = pd.read_csv(file_path, index_col = 0)
    # 可选：将数据转为数值类型（若文件中可能存在非数值字符）
    df = df.apply(pd.to_numeric, errors='coerce')
    # 将 DataFrame 中的中间数值部分转为 NumPy 数组
    matrix = df.values
    # print(matrix.shape)
    return matrix

def Input_Intial(filepath = "data/SEIJRD.csv"):
    df = pd.read_csv(filepath, index_col = None)
    df = df.astype(float)
    # print(df.shape)
    # 将 DataFrame 中的中间数值部分转为 NumPy 数组
    initial_states = df.values
    # print(initial_states)
    return initial_states

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

        self.endtime = None

        self.audience_from = []
    
    def infect(self, paras):
        SEIR = np.sum(self.audience_from, axis = 0)
        infectious_num = sum(SEIR[1:4])
        probability = 1 - (1 - paras["rho"]) ** infectious_num

        new_infected = SEIR[0] * probability
        
        susceptible = self.audience_from[:, 0]
        if susceptible.sum() < 1e-3:
            return
        distribution = susceptible / np.sum(susceptible)

        new_infected_by_row = distribution * new_infected
        self.audience_from[:, 0] -= new_infected_by_row
        self.audience_from[:, 1] += new_infected_by_row
        
class Env:
    def __init__(self, populations: List[int], matrix = None, paras = None):
        self.populations = populations
        self.matrix = matrix.copy()
        self.origin_matrix = matrix.copy()

        self.num = len(populations)
        self.paras = paras
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
        
        states = np.zeros((self.num, 6))

        if self.paras["is_states"]:
            initial_states = Input_Intial()
            for i in range(self.num):
                states[i] = initial_states[i]
        else:
            for i in range(self.num):
                states[i] = np.array([self.populations[i] - self.paras["initial_infect"] * 4.5 / self.num, self.paras["initial_infect"] * 2 / self.num, self.paras["initial_infect"] / self.num, self.paras["initial_infect"] * 1.5 / self.num, 0, 0])

        if self.matrix is not None:
            self.network = SEIR_Network(self.num, states, self.matrix, self.paras)
        else:
            self.network = SEIR_Network(self.num, [np.array([250200, 4000, 2500, 0, 0, 0])] * self.num, paras = self.paras)
    
    def check_competition_start(self):
        for id, place in self.places.items():
            if len(place.agenda) == 0:
                continue
            competition = place.agenda[0]
            if competition[1] == self.current_tick:
                place.audience_from = np.zeros((self.network.node_num, 5))
                place.agenda.pop(0)
                place.endtime = competition[2]
                
                # 从 SEIR 网络中随机选择一部分人群到场
                N = []
                for i in range(self.network.node_num):
                    N.append(sum(self.network.nodes[i].state[:4]))
                N = np.array(N)
                
                for i in range(self.network.node_num):
                    try:
                        place.audience_from[i] = np.array(self.network.nodes[i].state[:5]) * competition[3] / sum(N) * self.capacity_strategy[id - 1]
                    except:
                        place.audience_from[i] = np.array(self.network.nodes[i].state[:5]) * competition[3] / sum(N)
                        print(id)
                    self.network.nodes[i].state[:5] -= place.audience_from[i]
                
                place.infect(self.paras)


    def check_competition_end(self):
        for place in self.places.values():
            if place.endtime == self.current_tick:
                for i in range(self.network.node_num):
                    self.network.nodes[i].state[:5] += place.audience_from[i]
                place.audience_from = []
                place.endtime = None
    
    def modify_matrix(self, matrix):
        matrix_ = matrix / np.sum(matrix, axis = 1, keepdims = True)
        self.network.A = matrix_.copy()
        self.matrix = matrix_.copy()

class SEIR_Node:
    def __init__(self, state, total_nodes, name = None):
        self.total_nodes = total_nodes
        self.name = name
        self.state = state.astype(np.float64)
        self.from_node = np.zeros((total_nodes, 6))
        self.to_node = np.zeros((total_nodes, 5))
    
    def update_seir(self, delta_time = 0.01, times = 100, paras = backup_paras):
        N = sum(self.state)
        SEIJRD = self.state.copy()

        if N > 0:
            dS = - paras["beta"] * SEIJRD[0] * SEIJRD[1] / N
            dE = paras["beta"] * SEIJRD[0] * SEIJRD[1] / N - paras["sigma_i"] * SEIJRD[1] - paras["sigma_j"] * SEIJRD[1]
            dI = paras["sigma_i"] * SEIJRD[1] - paras["gamma_i"] * SEIJRD[2] - paras["mu_i"] * SEIJRD[2]
            dJ = paras["sigma_j"] * SEIJRD[1] - paras["gamma_j"] * SEIJRD[3] - paras["mu_j"] * SEIJRD[3]
            dR = paras["gamma_i"] * SEIJRD[2] + paras["gamma_j"] * SEIJRD[3]
            dD = paras["mu_i"] * SEIJRD[2] + paras["mu_j"] * SEIJRD[3]
        else:
            dS = 0.0
            dE = 0.0
            dI = 0.0
            dJ = 0.0
            dR = 0.0
            dD = 0.0
        
        self.state += np.array([dS, dE, dI, dJ, dR, dD]) * delta_time * times

        Nvalue = np.sum(self.from_node, axis = 1)
        # print(self.from_node.shape)
        self.from_node += (np.array([dS, dE, dI, dJ, dR, dD])[None, :]) * ((Nvalue / N)[:, None]) * delta_time * times
        
    def plot(self):
        plt.plot(self.state)
        plt.show()

class SEIR_Network:
    def __init__(self, num, states, matrix = None, paras = backup_paras):
        self.node_num = num
        self.nodes = {}
        self.paras = paras

        for i in range(num):
            self.nodes[i] = SEIR_Node(states[i], num)

        if matrix is not None:
            self.A = matrix / np.sum(matrix, axis = 1, keepdims = True)
        else:
            self.A = np.random.random((num, num))
            self.A = self.A / np.sum(self.A, axis = 1, keepdims = True)
            # print(self.A)

        self.delta_time = 1 / 2400

    def morning_commuting(self):
        for i in range(self.node_num):
            SEIJR = self.nodes[i].state[:5]
            self.nodes[i].to_node = np.outer(self.A[i], SEIJR)
        
        for i in range(self.node_num):
            self.nodes[i].state[:5] = [0.0] * 5   # 倾巢而出
        
        for i in range(self.node_num):
            for j in range(self.node_num):
                self.nodes[i].state[:5] += self.nodes[j].to_node[i] # 到达目的地，直接加上
                self.nodes[j].from_node[i][:5] = self.nodes[i].to_node[j] # 记录到达的节点
           
        # print("after morning commuting")
        # populations = [sum(self.nodes[i].state) for i in range(self.node_num)]
        # print(f"total: {sum(populations)}, {populations}")

    def evening_commuting(self):
        for i in range(self.node_num):
            for j in range(self.node_num):
                if i == j:
                    continue
                self.nodes[j].state -= self.nodes[j].from_node[i] # 回家，直接减去
                self.nodes[i].state += self.nodes[j].from_node[i] # 到家了，加上 
            
        for i in range(self.node_num):
            self.nodes[i].from_node = np.zeros((self.node_num, 6))
            self.nodes[i].to_node = np.zeros((self.node_num, 5))
        
        # print("after evening commuting")
        # populations = [sum(self.nodes[i].state) for i in range(self.node_num)]
        # print(f"total: {sum(populations)}, {populations}")

    def update_network(self):
        for i in range(self.node_num):
            self.nodes[i].update_seir(self.delta_time, 100, self.paras)

def RL_strategy_1(Env, data):
    ans = [0.0 for place in Env.places]
    return ans

def RL_strategy_2(Env, data):
    return np.eye(Env.num)

def main_update(start_date = "2021-07-21", end_date = "2021-08-09", paras = backup_paras):
    origin_paras = paras.copy()
    populations = Input_Population()
    matrix = Input_Matrix()
    np.random.seed(0)
    env = Env(populations, matrix, paras)
    env.init_env()
    
    states_history = []

    #### 强化学习策略 ####
    env.capacity_strategy = RL_strategy_1(env, states_history)
    #####################

    ref_date = datetime.datetime.strptime("2021-07-21", "%Y-%m-%d").date()
    # 将传入的起始日期字符串转换为日期对象
    start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
    end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d").date()

    offset_l = (ref_date - start_date).days
    offset_r = (end_date - ref_date).days

    for day in range(-offset_l, offset_r + 1):
        for hour in range(24):
            tick = day * 24 + hour
            start_time = time.time()
            env.current_tick = tick
            #### 新增调整矩阵 ####
            env.modify_matrix(RL_strategy_2(env, states_history))
            paras['beta'] = 0.05 + (origin_paras['beta'] - 0.05) * (np.sum(env.matrix) - np.trace(env.matrix)) / (np.sum(env.origin_matrix) - np.trace(env.origin_matrix))
            #####################
            env.check_competition_start()
            if tick % 24 == 8:
                env.network.morning_commuting()
            
            env.network.update_network()

            if tick % 24 == 18:
                env.network.evening_commuting()
            env.check_competition_end()
            end_time = time.time()

            # populations = [sum(env.network.nodes[i].state) for i in range(env.network.node_num)]
            # print(f"tick: {tick}, total: {sum(populations)}, {populations}")

            if tick % 24 == 0:
                SEIRD = np.zeros((env.num, 6))
                for i in range(env.num):
                    SEIRD[i] = env.network.nodes[i].state.copy()
                    # for j in range(NUM):
                    #     if i == j:
                    #         continue
                    #     SEIRD[i] += env.network.nodes[j].from_node[i]
                    # if i == 0:
                    #     print("tick: ", tick, "time: ", end_time - start_time, np.array2string(SEIRD[i], formatter = {'float_kind': lambda x: f"{x:.2f}"}))
                
                states_history.append(SEIRD.copy())

        #### 强化学习策略 ####
        env.capacity_strategy = RL_strategy_1(env, states_history)
        #####################
    
    # 将状态保存
    # SEIJRD = np.zeros((env.num, 6))
    # for i in range(env.num):
    #     SEIJRD[i] = env.network.nodes[i].state.copy()
    
    # np.savetxt("data/SEIJRD.csv", SEIJRD, delimiter=",", header="S,E,I,J,R,D", comments="")

    return states_history

if __name__ == "__main__":
    start_str = "2021-07-21"
    end_str   = "2021-08-09"

    # recovery_time = 7 # 恢复时间
    # with open("./data/infect_window.json", "r", encoding="utf-8") as f:
    #     data_infect = json.load(f)
    # with open("./data/death_window.json", "r", encoding="utf-8") as f:
    #     data_death = json.load(f)

    # result_infect = filter_data(
    #     json_data = data_infect,
    #     start_str = start_str,
    #     end_str   = end_str
    # )

    # result_death = filter_data(
    #     json_data = data_death,
    #     start_str = start_str,
    #     end_str   = end_str
    # )

    # initial_infect = np.sum(np.array(result_infect[:recovery_time]) - np.array(result_death[:recovery_time])) # 假设7天才能康复，选择前7天的新增感染减去前两周的死亡作为一个预估
    
    paras = {
        "beta": 0.155,
        "gamma_i": 0.079,
        "gamma_j": 0.031,
        "sigma_i": 0.0299,
        "sigma_j": 0.0156,
        "mu_i": 1.95e-5,
        "mu_j": 0.00025,
        "rho": 8.48e-5, # 接触传染概率
        # "initial_infect": initial_infect, # 初始感染人数
        "is_states" : True
    }

    states_history = main_update(start_date = start_str, end_date = end_str, paras = paras)
    states_history = np.array(states_history)
    node0_history = np.vstack([np.sum(state, axis = 0) for state in states_history])
    
    for i in range(6):
        plt.plot(node0_history[:,i])

    plt.legend(["S", "E", "I", "J", "R", "D"])
    plt.savefig(f"plot/{paras['rho']}_all.png")
    
        
