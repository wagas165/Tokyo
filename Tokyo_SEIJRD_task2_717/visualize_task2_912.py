#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Task2 visualization - visualize_task2_910.py

- 读取 ./results/**/evaluation.json 与 ./results/baseline_*/baseline_evaluation.json
- 输出：
  1) ./visualizations/algorithm_radar_chart.png  (全部实验曲线)
  2) ./visualizations/grouped_results.png        (PPO/A2C/Baseline 分组 4 指标)
  3) ./visualizations/differential_analysis.png  (相对 baseline_original 的相对差)
  4) ./interactive/interactive_visualization.html + ./interactive/results_data.json
"""

import os, glob, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from typing import Dict, Tuple

# ---------------- helpers ----------------
def _load_last_episode(json_path: str) -> Dict:
    with open(json_path, "r") as f:
        data = json.load(f)
    if data.get("episodes"):
        return data["episodes"][-1]
    return None

def _derived(ep: Dict) -> Dict:
    d={}
    for k in ["hourly_infections","hourly_deaths","hourly_commute_ratio","hourly_reward"]:
        d[k] = list(ep.get(k, []))
    d["hourly_cumulative_infections"] = np.cumsum(d["hourly_infections"]).tolist() if d["hourly_infections"] else []
    d["hourly_cumulative_deaths"]     = np.cumsum(d["hourly_deaths"]).tolist() if d["hourly_deaths"] else []
    s=0.0; avg=[]
    for i,v in enumerate(d["hourly_commute_ratio"]):
        s+=v; avg.append(s/(i+1))
    d["hourly_average_commute"] = avg
    s=0.0; cr=[]
    for v in d["hourly_reward"]:
        s+=v; cr.append(s)
    d["hourly_cumulative_reward"] = cr
    return d

def _collect_all(results_root="./results"):
    """返回 all_results: {key: dict}，key形如 PPO_exp1 / baseline_original"""
    import re
    all_results = {}

    # RL
    rl_paths = glob.glob(os.path.join(results_root, "*_exp*/evaluation.json"))
    for p in rl_paths:
        last = _load_last_episode(p)
        if last is None:
            continue
        m = re.search(r"(PPO|A2C)_exp(\d+)", p)
        if not m:
            continue
        key = f"{m.group(1)}_exp{m.group(2)}"
        all_results[key] = _derived(last)

    # Baseline
    bl_paths = glob.glob(os.path.join(results_root, "baseline_*/baseline_evaluation.json"))
    for p in bl_paths:
        last = _load_last_episode(p)
        if last is None:
            continue
        m = re.search(r"baseline_(\w+)", p)
        if not m:
            continue
        key = f"baseline_{m.group(1)}"
        all_results[key] = _derived(last)

    return all_results

# ---------------- plots ----------------
def algorithm_radar_chart(all_results: Dict[str,Dict]):
    """全量雷达图：感染控制、死亡控制、通勤保持、计算效率(经验赋值)、策略稳定性"""
    keys = sorted(all_results.keys())
    if not keys:
        return
    # 归一化因子
    inf_sums = [np.sum(all_results[k].get("hourly_infections",[])) for k in keys]
    dth_sums = [np.sum(all_results[k].get("hourly_deaths",[])) for k in keys]
    max_inf = max(1.0, max(inf_sums))
    max_dth = max(1.0, max(dth_sums))

    metrics = ["Infection Control","Death Prevention","Commute Preservation","Computational Efficiency","Policy Stability"]
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    plt.figure(figsize=(14,10))
    ax = plt.subplot(111, polar=True)

    for k in keys:
        v = all_results[k]
        inf = np.sum(v.get("hourly_infections",[]))
        dth = np.sum(v.get("hourly_deaths",[]))
        cmt = float(np.mean(v.get("hourly_commute_ratio",[0])))
        eff = 0.6 if k.startswith("PPO") else (0.8 if k.startswith("A2C") else 1.0)
        stab = max(0.0, 1.0 - float(np.std(v.get("hourly_commute_ratio",[0]))))
        data = [1 - inf/max_inf, 1 - dth/max_dth, cmt, eff, stab]
        data += data[:1]
        ax.plot(angles, data, linewidth=1.8, label=k)
        ax.fill(angles, data, alpha=0.07)

    ax.set_xticks(angles[:-1]); ax.set_xticklabels(metrics)
    ax.set_yticklabels([])
    ax.set_title("Algorithm Radar Chart (All Experiments)", pad=20)
    ax.grid(alpha=.3)
    # 放到右侧，避免曲线遮挡
    ax.legend(loc='upper left', bbox_to_anchor=(1.03, 1.02), fontsize=9, ncol=1)
    plt.tight_layout()
    plt.savefig("./visualizations/algorithm_radar_chart.png")
    plt.close()

def grouped_results(all_results: Dict[str,Dict]):
    groups = {
        "PPO": {k:v for k,v in all_results.items() if k.startswith("PPO_")},
        "A2C": {k:v for k,v in all_results.items() if k.startswith("A2C_")},
        "Baseline": {k:v for k,v in all_results.items() if k.startswith("baseline_")}
    }
    fig = plt.figure(figsize=(22,14))
    gs = fig.add_gridspec(3,4)
    titles=["Infections","Deaths","Commute","Reward"]
    metrics=["hourly_infections","hourly_deaths","hourly_commute_ratio","hourly_reward"]
    for r,(gname,gdict) in enumerate(groups.items()):
        for c,(m,tt) in enumerate(zip(metrics,titles)):
            ax = fig.add_subplot(gs[r,c])
            for k,v in sorted(gdict.items()):
                if m in v and len(v[m])>0:
                    ax.plot(v[m], label=k, alpha=0.9, linewidth=1.2)
            ax.set_title(f"{gname} - {tt}")
            ax.grid(alpha=.3)
            if r==0 and c==0:
                ax.legend(fontsize=8, ncol=1)
    plt.tight_layout()
    plt.savefig("./visualizations/grouped_results.png")
    plt.close()

def differential_analysis(all_results: Dict[str,Dict], base="baseline_original"):
    if base not in all_results:
        print(f"No baseline {base} found => skip differential_visualization.")
        return
    base_v = all_results[base]
    plt.figure(figsize=(16,18))
    pairs=[
        ("hourly_infections", "Infections (% diff)"),
        ("hourly_deaths", "Deaths (% diff)"),
        ("hourly_commute_ratio", "Commute Ratio (% diff)"),
        ("hourly_reward", "Reward (% diff)")
    ]
    for i,(m,tt) in enumerate(pairs):
        ax = plt.subplot(4,1,i+1)
        B = np.array(base_v.get(m,[]))
        for k,v in sorted(all_results.items()):
            if k==base:
                continue
            if m not in v:
                continue
            A = np.array(v[m]); L = min(len(A), len(B))
            if L==0:
                continue
            d = (A[:L]-B[:L])/(np.abs(B[:L])+1e-12)
            d = np.clip(d, -1, 1)
            ax.plot(d, label=k, linewidth=1.2)
        ax.axhline(0,color='k',ls='--',alpha=.3)
        ax.set_title(tt); ax.grid(alpha=.3)
        if i==0: ax.legend(bbox_to_anchor=(1.02,1.0), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig("./visualizations/differential_analysis.png")
    plt.close()

# ---------------- interactive ----------------
def make_interactive(all_results: Dict[str,Dict]):
    os.makedirs("./interactive", exist_ok=True)
    # 数据：确保都是 list
    safe_dict={}
    for k,v in all_results.items():
        safe_v={}
        for kk,vv in v.items():
            if isinstance(vv, np.ndarray):
                safe_v[kk] = vv.tolist()
            elif isinstance(vv, list):
                safe_v[kk] = [float(x) if isinstance(x,(np.floating,np.float32,np.float64)) else x for x in vv]
            else:
                safe_v[kk] = vv
        safe_dict[k]=safe_v
    with open("./interactive/results_data.json","w") as f:
        json.dump(safe_dict, f)  # 不再触发 ndarray 可序列化问题

    # HTML
    html = r"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<title>Task2 Interactive Visualization</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
<style>
body {font-family:Arial, sans-serif; margin:20px;}
.container{display:flex; flex-wrap:wrap;}
.chart{width:48%; margin:1%; height:360px;}
.ctrl{margin-bottom:12px;}
button{margin-left:8px;}
</style>
</head>
<body>
<h2>Task2 Interactive Visualization</h2>
<div class="ctrl">
<label>Filter:</label>
<select id="algof">
<option value="all">All</option>
<option value="PPO">PPO</option>
<option value="A2C">A2C</option>
<option value="baseline">Baselines</option>
</select>
<button id="toggle-diff">Diff vs baseline_original</button>
</div>
<div class="container">
  <canvas id="c1" class="chart"></canvas>
  <canvas id="c2" class="chart"></canvas>
  <canvas id="c3" class="chart"></canvas>
  <canvas id="c4" class="chart"></canvas>
</div>
<script>
let ALL={}, DIFF=false;
async function load(){
  const res = await fetch('results_data.json'); ALL = await res.json();
  draw();
}
function fset(){return document.getElementById('algof').value;}
function filt(){
  const v=fset(); const out={};
  for(const k in ALL){
    if(v==='all' || (v==='PPO'&&k.startsWith('PPO')) || (v==='A2C'&&k.startsWith('A2C')) || (v==='baseline'&&k.startsWith('baseline'))){
      out[k]=ALL[k];
    }
  }
  return out;
}
function diff(dict){
  const base=ALL['baseline_original']; if(!base) return dict;
  const out={};
  for(const k in dict){
    if(k==='baseline_original') continue;
    out[k]={};
    for(const m of ['hourly_infections','hourly_deaths','hourly_commute_ratio','hourly_reward']){
      if(!dict[k][m] || !base[m]) continue;
      const L=Math.min(dict[k][m].length, base[m].length);
      const arr=[]; for(let i=0;i<L;i++){
        const b=base[m][i], r=dict[k][m][i];
        const d=(Math.abs(b)>1e-12)?(r-b)/Math.abs(b):r;
        arr.push(Math.max(-1,Math.min(1,d)));
      }
      out[k][m]=arr;
    }
  }
  return out;
}
let g1,g2,g3,g4;
function draw(){
  const fd=filt(); const data = DIFF?diff(fd):fd;
  const colors=d3.schemeCategory10;
  const m=['hourly_infections','hourly_deaths','hourly_commute_ratio','hourly_reward'];
  const ids=['c1','c2','c3','c4'];
  const titles= DIFF? ['Infections (rel. diff)','Deaths (rel. diff)','Commute (rel. diff)','Reward (rel. diff)'] :
                       ['Hourly Infections','Hourly Deaths','Hourly Commute','Hourly Reward'];
  const charts=[g1,g2,g3,g4];
  for(let k=0;k<4;k++){
    const ctx=document.getElementById(ids[k]).getContext('2d');
    if(charts[k]) charts[k].destroy();
    const dsets=[]; let i=0;
    for(const name in data){
      if(!data[name][m[k]]) continue;
      dsets.push({label:name, data:data[name][m[k]], borderColor:colors[i%10], backgroundColor:colors[i%10]+'33', fill:false, pointRadius:0, borderWidth:1.5});
      i++;
    }
    charts[k]=new Chart(ctx, {type:'line', data:{labels:[...Array(dsets.length?dsets[0].data.length:0).keys()], datasets:dsets}, options:{responsive:true, plugins:{legend:{display:true}}}});
    [g1,g2,g3,g4]=charts;
  }
}
document.getElementById('algof').addEventListener('change',draw);
document.getElementById('toggle-diff').addEventListener('click', ()=>{DIFF=!DIFF; draw();});
load();
</script>
</body>
</html>
"""
    with open("./interactive/interactive_visualization.html","w", encoding="utf-8") as f:
        f.write(html)
    print("已生成 ./interactive/interactive_visualization.html 与 ./interactive/results_data.json")

# ---------------- orchestration ----------------
def run_all_visualizations(results_root="./results"):
    all_results = _collect_all(results_root)
    if not all_results:
        print("No results found; nothing to visualize.")
        return
    algorithm_radar_chart(all_results)
    grouped_results(all_results)
    differential_analysis(all_results, base="baseline_original")
    make_interactive(all_results)
    print("All Task2 visualizations are generated into ./visualizations and ./interactive")

if __name__ == "__main__":
    run_all_visualizations("./results")
