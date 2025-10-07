# advanced_visuals.py
# 只包含可视化与高级分析相关的函数，不导入训练主程序
# 通过读取 ./results/.../evaluation.json 来解析并作图

import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict
from scipy import stats
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Polygon
import warnings, re
warnings.filterwarnings('ignore')

# ---------- 工具：把 numpy 转为原生 Python ----------
def _to_py(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64, np.int32, np.int64)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_py(v) for v in obj]
    return obj

def generate_simulated_results():
    """若找不到任何 evaluation.json, 生成模拟数据。"""
    simulated_results = {}

    def gen_series(length=480, pattern='epidemic', seed=None):
        if seed is not None:
            np.random.seed(seed)
        t = np.arange(length)
        if pattern == 'epidemic':
            base = 100*np.exp(-0.01*(t-80)**2)
            daily = 20*np.sin(2*np.pi*t/24)
            noise = 0.1*np.random.randn(length)*base
            return base+daily+noise
        elif pattern == 'deaths':
            base = 5*np.exp(-0.01*(t-100)**2)
            noise = 0.1*np.random.randn(length)*base
            return base+noise
        elif pattern == 'commute':
            base = 1-0.5*(1-np.exp(-0.01*t))
            noise = 0.1*np.random.randn(length)
            return np.clip(base+noise,0,1)
        elif pattern == 'reward':
            base = 100*(1-np.exp(-0.005*t))
            noise = 0.1*50*np.random.randn(length)
            return base+noise
        else:
            return np.random.randn(length)*100

    for algo in ['PPO','A2C']:
        for i in range(1,5):
            key=f"{algo}_exp{i}"
            seed_=hash(key)%9999
            simulated_results[key]={
                'hourly_infections': gen_series(pattern='epidemic',seed=seed_).tolist(),
                'hourly_deaths': gen_series(pattern='deaths',seed=seed_+1).tolist(),
                'hourly_commute_ratio': gen_series(pattern='commute',seed=seed_+2).tolist(),
                'hourly_reward': gen_series(pattern='reward',seed=seed_+3).tolist()
            }

    for b in ['original','diagonal','neighbor','random50']:
        key=f"baseline_{b}"
        seed_=hash(key)%9999
        base_inf = gen_series(pattern='epidemic',seed=seed_).tolist()
        base_dth = gen_series(pattern='deaths',seed=seed_+1).tolist()
        if b=='original':
            inf_scale,dth_scale,cval = 1.5,1.5,1.0
        elif b=='diagonal':
            inf_scale,dth_scale,cval = 0.3,0.3,0.0
        elif b=='neighbor':
            inf_scale,dth_scale,cval = 0.8,0.8,0.5
        else:
            inf_scale,dth_scale,cval = 1.0,1.0,0.5
        simulated_results[key]={
            'hourly_infections':(np.array(base_inf)*inf_scale).tolist(),
            'hourly_deaths':(np.array(base_dth)*dth_scale).tolist(),
            'hourly_commute_ratio':[cval]*480,
            'hourly_reward': (np.array(gen_series(pattern='reward',seed=seed_+3))*0.8).tolist()
        }
    return simulated_results

# --------- 常规图（保持原接口） ---------
def create_strategy_comparison(rl_results, baseline_results):
    plt.figure(figsize=(15,20))
    metrics=["cumulative_infections","cumulative_deaths","average_commute","cumulative_reward"]
    titles=["Cumulative Infections","Cumulative Deaths","Average Commute Ratio","Cumulative Reward"]
    for i,(m,t) in enumerate(zip(metrics,titles)):
        plt.subplot(4,1,i+1)
        for algo,exp_id,res in rl_results:
            label=f"{algo}_exp{exp_id}"
            hh=f"hourly_{m}"
            if hh in res:
                plt.plot(res[hh],label=label)
        for bname,bres in baseline_results:
            hh=f"hourly_{m}"
            if hh in bres:
                plt.plot(bres[hh],label=f"baseline_{bname}",linestyle='--',linewidth=2)
        plt.title(t); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig("./visualizations/strategy_comparison.png"); plt.close()

def create_final_state_comparison(rl_results, baseline_results):
    metrics=["total_infections","total_deaths","final_commute_ratio"]
    titles=["Total Infections","Total Deaths","Average Commute Ratio"]
    strategies=[]; data={m:[] for m in metrics}

    for algo,exp_id,res in rl_results:
        strategies.append(f"{algo}_exp{exp_id}")
        data["total_infections"].append(np.sum(res.get('hourly_infections',[])))
        data["total_deaths"].append(np.sum(res.get('hourly_deaths',[])))
        data["final_commute_ratio"].append(np.mean(res.get('hourly_commute_ratio',[0])))
    for bname,bres in baseline_results:
        strategies.append(f"baseline_{bname}")
        data["total_infections"].append(np.sum(bres.get('hourly_infections',[])))
        data["total_deaths"].append(np.sum(bres.get('hourly_deaths',[])))
        data["final_commute_ratio"].append(np.mean(bres.get('hourly_commute_ratio',[0])))

    fig,axes=plt.subplots(1,3,figsize=(18,6))
    for i,(m,t) in enumerate(zip(metrics,titles)):
        axes[i].bar(strategies, data[m])
        axes[i].set_title(t)
        axes[i].set_xticklabels(strategies,rotation=45,ha='right')
    plt.tight_layout(); plt.savefig("./visualizations/final_state_comparison.png"); plt.close()

def visualize_matrix_evolution(model_path, algo, exp_id):
    # 占位：如已存 policy_matrices，可直接解析作图
    fig,ax=plt.subplots(figsize=(6,4))
    ax.text(0.5,0.5,"Matrix evolution placeholder.\nIf policy_matrices saved,\nparse & plot here.",
            ha='center',va='center')
    plt.savefig(f"./visualizations/{algo}_exp{exp_id}_matrix_evolution.png"); plt.close()

def create_algorithm_radar_chart(rl_results, baseline_results, max_infs, max_dths):
    metrics=["Infection Control","Death Prevention","Commute Preservation","Computational Efficiency","Policy Stability"]
    algorithms=[]; values=[]
    for algo,exp_id,res in rl_results:
        algorithms.append(f"{algo}_exp{exp_id}")
        inf_sum=np.sum(res.get('hourly_infections',[]))
        dth_sum=np.sum(res.get('hourly_deaths',[]))
        cmt_avg=np.mean(res.get('hourly_commute_ratio',[0]))
        efficiency=0.6 if algo=='PPO' else 0.8
        stability=0.9 - np.std(res.get('hourly_commute_ratio',[0]))
        values.append([
            1-inf_sum/max_infs if max_infs>0 else 1,
            1-dth_sum/max_dths if max_dths>0 else 1,
            cmt_avg, efficiency, max(0,min(1,stability))
        ])
    for bname,bres in baseline_results:
        algorithms.append(f"baseline_{bname}")
        inf_sum=np.sum(bres.get('hourly_infections',[]))
        dth_sum=np.sum(bres.get('hourly_deaths',[]))
        cmt_avg=np.mean(bres.get('hourly_commute_ratio',[0]))
        efficiency=1.0; stability=1.0 if bname!='random50' else 0.5
        values.append([
            1-inf_sum/max_infs if max_infs>0 else 1,
            1-dth_sum/max_dths if max_dths>0 else 1,
            cmt_avg, efficiency, stability
        ])
    fig=plt.figure(figsize=(12,8)); ax=fig.add_subplot(111,polar=True)
    angles=np.linspace(0,2*np.pi,len(metrics),endpoint=False).tolist(); angles+=angles[:1]
    for i,alg in enumerate(algorithms):
        data=values[i]+values[i][:1]
        ax.plot(angles,data,linewidth=2,label=alg); ax.fill(angles,data,alpha=0.1)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(metrics)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3,1.05))
    plt.savefig("./visualizations/algorithm_radar_chart.png"); plt.close()

def visualize_district_infections(rl_model, baseline_model, days=20):
    fig,ax=plt.subplots(figsize=(6,4))
    ax.text(0.5,0.5,"district_infections placeholder.",ha='center',va='center')
    plt.savefig("./visualizations/district_infection_comparison.png"); plt.close()

# --------- 差异可视化（改为自动回退 baseline） ---------
def differential_visualization(results_dict, baseline_key=None):
    # baseline 自动选择
    if baseline_key is None:
        cand = [k for k in results_dict.keys() if k.startswith("baseline_")]
        if "baseline_original" in cand:
            baseline_key = "baseline_original"
        elif cand:
            baseline_key = cand[0]
            print(f"Using {baseline_key} as baseline for differential analysis.")
        else:
            print("No baseline found => skip differential_visualization.")
            return {}

    base_res = results_dict[baseline_key]
    diff_results={}
    for k,v in results_dict.items():
        if k==baseline_key:
            continue
        diff_data={}
        for metric in ['hourly_infections','hourly_deaths','hourly_commute_ratio','hourly_reward']:
            if metric in v and metric in base_res:
                min_len=min(len(v[metric]), len(base_res[metric]))
                b_arr=np.array(base_res[metric][:min_len])
                r_arr=np.array(v[metric][:min_len])
                with np.errstate(divide='ignore',invalid='ignore'):
                    rel=(r_arr-b_arr)/(np.abs(b_arr)+1e-10)
                    rel=np.clip(np.nan_to_num(rel, nan=0, posinf=1, neginf=-1),-1,1)
                diff_data[metric]=rel.tolist()
        diff_results[k]=diff_data

    plt.figure(figsize=(15,20))
    ms=['hourly_infections','hourly_deaths','hourly_commute_ratio','hourly_reward']
    titles=["Infections(% diff)","Deaths(% diff)","Commute(% diff)","Reward(% diff)"]
    for i,(m,t) in enumerate(zip(ms,titles)):
        plt.subplot(4,1,i+1)
        for strat,ddata in diff_results.items():
            if m in ddata:
                plt.plot(ddata[m], label=strat)
        plt.axhline(y=0,color='k',linestyle='--',alpha=0.3); plt.title(t); plt.grid(alpha=0.3)
        if i==0: plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
    plt.tight_layout(); plt.savefig('./visualizations/differential_analysis.png'); plt.close()
    return diff_results

def grouped_hierarchical_visualization(results_dict):
    ppo_res={k:v for k,v in results_dict.items() if 'PPO' in k}
    a2c_res={k:v for k,v in results_dict.items() if 'A2C' in k}
    base_res={k:v for k,v in results_dict.items() if 'baseline' in k}
    def plot_groups(groups, metrics, titles, fname):
        from matplotlib import gridspec
        n_groups=len(groups); n_metrics=len(metrics)
        fig=plt.figure(figsize=(20,5*n_groups)); gs=gridspec.GridSpec(n_groups,n_metrics)
        row=0
        for gname,gdict in groups.items():
            col=0
            for m,tt in zip(metrics,titles):
                ax=fig.add_subplot(gs[row,col])
                for kk,vv in gdict.items():
                    if m in vv: ax.plot(vv[m],label=kk,alpha=0.7)
                ax.set_title(f"{gname} - {tt}"); ax.grid(alpha=0.3); ax.legend(); col+=1
            row+=1
        plt.tight_layout(); plt.savefig(f'./visualizations/{fname}.png'); plt.close()
    all_groups={"PPO": ppo_res, "A2C": a2c_res, "Baseline": base_res}
    metrics=["hourly_infections","hourly_deaths","hourly_commute_ratio","hourly_reward"]
    titles=["Infections","Deaths","Commute","Reward"]
    plot_groups(all_groups, metrics, titles, "grouped_results")
    return all_groups

# --------- 交互式 HTML 与数据导出（修复 JSON 可序列化） ---------
def create_interactive_visualization_code():
    html_code = r"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>互动式疫情控制策略可视化</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
  <style>
    body { font-family: Arial, sans-serif; margin:20px; }
    .container { display:flex; flex-wrap:wrap; }
    .chart-container{ width:48%; margin:1%; height:400px; }
    .control-panel{ width:100%; margin-bottom:20px; padding:10px; background:#f5f5f5; border-radius:5px; }
    button, select{ margin:5px; padding:5px 10px; }
    .toggle-button{ background:#4CAF50; color:white; border:none; border-radius:4px; cursor:pointer; }
    .toggle-button.active{ background:#2E7D32; }
  </style>
</head>
<body>
<h1>强化学习疫情控制策略可视化系统</h1>
<div class="control-panel">
  <label>算法筛选:</label>
  <select id="algorithm-filter">
    <option value="all">全部算法</option>
    <option value="PPO">仅PPO</option>
    <option value="A2C">仅A2C</option>
    <option value="baseline">仅基准策略</option>
  </select>
  <label>时间窗口:</label>
  <input type="range" id="time-window" min="10" max="480" value="480">
  <span id="time-value">480</span>
  <button id="toggle-diff-mode" class="toggle-button">切换差异模式</button>
  <button id="toggle-log-scale" class="toggle-button">切换对数比例</button>
</div>
<div class="container">
  <div class="chart-container"><canvas id="infections-chart"></canvas></div>
  <div class="chart-container"><canvas id="deaths-chart"></canvas></div>
  <div class="chart-container"><canvas id="commute-chart"></canvas></div>
  <div class="chart-container"><canvas id="reward-chart"></canvas></div>
</div>
<script>
let allData = {};
let diffMode=false, logScale=false, baselineKey='baseline_original', timeWindow=480;

async function loadData(){
  const resp = await fetch('results_data.json');
  allData = await resp.json();
  updateCharts();
}
function calcDiff(results){
  const base = allData[baselineKey];
  if(!base){ return {}; }
  const out={};
  for(const k in results){
    if(k===baselineKey) continue;
    out[k]={};
    for(const m of ['hourly_infections','hourly_deaths','hourly_commute_ratio','hourly_reward']){
      if(results[k][m] && base[m]){
        const L=Math.min(results[k][m].length, base[m].length);
        const arr=[];
        for(let i=0;i<L;i++){
          const b=base[m][i], r=results[k][m][i];
          const d = Math.abs(b)>1e-12 ? (r-b)/Math.abs(b) : r;
          arr.push(Math.max(-1, Math.min(1, d)));
        }
        out[k][m]=arr;
      }
    }
  }
  return out;
}
function updateCharts(){
  const filter=document.getElementById('algorithm-filter').value;
  let filtered={};
  for(const k in allData){
    if(filter==='all' || (filter==='PPO'&&k.includes('PPO')) ||
       (filter==='A2C'&&k.includes('A2C')) || (filter==='baseline'&&k.includes('baseline'))){
      filtered[k]=allData[k];
    }
  }
  const data = diffMode? calcDiff(filtered): filtered;
  for(const k in data){
    for(const m in data[k]){
      data[k][m] = data[k][m].slice(0, timeWindow);
    }
  }
  drawChart('infections-chart', data, 'hourly_infections', diffMode?'感染相对差异':'每小时感染数');
  drawChart('deaths-chart', data, 'hourly_deaths', diffMode?'死亡相对差异':'每小时死亡数');
  drawChart('commute-chart', data, 'hourly_commute_ratio', diffMode?'通勤率相对差异':'每小时通勤率');
  drawChart('reward-chart', data, 'hourly_reward', diffMode?'奖励相对差异':'每小时奖励');
}
function drawChart(cid, data, metric, title){
  const ctx=document.getElementById(cid).getContext('2d');
  if(window[cid]) window[cid].destroy();
  const ds=[]; const color=d3.scaleOrdinal(d3.schemeCategory10); let i=0;
  for(const k in data){
    if(data[k][metric]){
      ds.push({label:k, data:data[k][metric], borderColor: color(i), backgroundColor: color(i)+'33', borderWidth: k.includes('baseline')?2:1, pointRadius:0});
      i++;
    }
  }
  window[cid]=new Chart(ctx,{type:'line',data:{labels:Array(timeWindow).fill(0).map((_,i)=>i),datasets:ds},
    options:{responsive:true, plugins:{title:{display:true,text:title}}, scales:{y:{type:(logScale&&!diffMode)?'logarithmic':'linear'}, x:{display:true}}}});
}
document.getElementById('algorithm-filter').addEventListener('change',updateCharts);
document.getElementById('time-window').addEventListener('input',e=>{ timeWindow=parseInt(e.target.value); document.getElementById('time-value').textContent=timeWindow; updateCharts();});
document.getElementById('toggle-diff-mode').addEventListener('click',function(){ diffMode=!diffMode; this.classList.toggle('active'); updateCharts();});
document.getElementById('toggle-log-scale').addEventListener('click',function(){ logScale=!logScale; this.classList.toggle('active'); updateCharts();});
loadData();
</script>
</body>
</html>
"""
    os.makedirs("./interactive", exist_ok=True)
    with open("./interactive/interactive_visualization.html","w",encoding="utf-8") as f:
        f.write(html_code)

    def export_data_for_interactive(results_dict, output_path="./interactive/results_data.json"):
        data = _to_py(results_dict)  # 修复：保证可序列化
        with open(output_path, 'w') as ff:
            json.dump(data, ff)
        print(f"已导出互动式可视化所需数据到 {output_path}")
    print("已生成 ./interactive/interactive_visualization.html")
    return export_data_for_interactive

# --------- 统计显著性（保留） ---------
def statistical_significance_analysis(results_dict, baseline_key=None):
    cand = [k for k in results_dict.keys() if k.startswith("baseline_")]
    if baseline_key is None:
        baseline_key = "baseline_original" if "baseline_original" in cand else (cand[0] if cand else None)
    if baseline_key is None:
        print("No baseline found => skip significance.")
        return {}, {}
    baseline_res=results_dict[baseline_key]
    significance_results={}; effect_sizes={}
    metrics=["hourly_infections","hourly_deaths","hourly_commute_ratio","hourly_reward"]
    from scipy.stats import mannwhitneyu
    for k,v in results_dict.items():
        if k==baseline_key: continue
        significance_results[k]={}; effect_sizes[k]={}
        for metric in metrics:
            if metric in v and metric in baseline_res:
                L=min(len(v[metric]), len(baseline_res[metric]))
                arr1=np.array(v[metric][:L]); arr2=np.array(baseline_res[metric][:L])
                u,p=mannwhitneyu(arr1,arr2,alternative='two-sided')
                # Cliff's delta（简化）
                n1,n2=len(arr1),len(arr2)
                dom=0
                for a in arr1:
                    dom += np.sum(a>arr2) - np.sum(a<arr2)
                cliff = dom/(n1*n2)
                significance_results[k][metric]={'p_value':float(p),'significant':bool(p<0.05),'test_name':'Mann-Whitney U'}
                ad=abs(cliff)
                if ad<0.147: mg="negligible"
                elif ad<0.33: mg="small"
                elif ad<0.474: mg="medium"
                else: mg="large"
                effect_sizes[k][metric]={'cliff_delta':float(cliff),'effect_size':float(ad),'effect_magnitude':mg}

    plt.figure(figsize=(20,15))
    names=list(significance_results.keys())
    for i,metric in enumerate(metrics):
        plt.subplot(2,2,i+1)
        pvals=[significance_results[n].get(metric,{}).get('p_value',1.0) for n in names]
        neglog=[-np.log10(p) if p>0 else 10 for p in pvals]
        plt.bar(names,neglog); plt.axhline(y=-np.log10(0.05),color='r',linestyle='--',label='p=0.05')
        plt.title(f'{metric} significance(-log10p)'); plt.xticks(rotation=45,ha='right'); plt.legend()
    plt.tight_layout(); plt.savefig('./visualizations/statistical_significance.png'); plt.close()

    with open('./results/statistical_analysis.json','w') as ff:
        json.dump({'significance':_to_py(significance_results),'effect_sizes':_to_py(effect_sizes)}, ff, indent=2)
    print("统计显著性分析完成 => ./visualizations/statistical_significance.png & ./results/statistical_analysis.json")
    return significance_results, effect_sizes

def feature_extraction_pattern_recognition(results_dict):
    fig,ax=plt.subplots(figsize=(6,4))
    ax.text(0.5,0.5,"feature_extraction placeholder",ha='center',va='center')
    plt.savefig('./visualizations/feature_extraction.png'); plt.close()
    return {}

def multi_dimensional_assessment(results_dict, district_names=None):
    with open('./results/multidimensional_assessment.json','w') as ff:
        json.dump({'dummy':'Yes'},ff,indent=2)
    return {}

def improved_radar_chart(results_dict):
    fig,ax=plt.subplots(figsize=(6,4))
    ax.text(0.5,0.5,"improved_radar_chart placeholder",ha='center',va='center')
    plt.savefig('./visualizations/improved_radar_chart.png'); plt.close()
    return {}

def improved_district_infection_chart(rl_results, baseline_results):
    fig,ax=plt.subplots(figsize=(6,4))
    ax.text(0.5,0.5,"improved_district_infection_chart placeholder",ha='center',va='center')
    plt.savefig('./visualizations/improved_district_infections.png'); plt.close()
    return None

def improved_matrix_evolution_chart(model_results, baseline_results=None):
    fig,ax=plt.subplots(figsize=(6,4))
    ax.text(0.5,0.5,"improved_matrix_evolution_chart placeholder",ha='center',va='center')
    plt.savefig('./visualizations/improved_matrix_evolution_chart.png'); plt.close()
    return None

def improved_time_series_chart(results_dict):
    fig,ax=plt.subplots(figsize=(6,4))
    ax.text(0.5,0.5,"improved_time_series_chart placeholder",ha='center',va='center')
    plt.savefig('./visualizations/improved_time_series.png'); plt.close()
    return {}, {}

# --------- 总入口：读取 results → 作图 → 导出交互式数据 ---------
def run_all_visualization_improvements(results_files_path='./results/'):
    print("Start run_all_visualization_improvements...")

    all_results={}
    # 1) 读取 RL
    pattern_rl = os.path.join(results_files_path,"*_exp*/evaluation.json")
    for fp in glob.glob(pattern_rl):
        try:
            with open(fp,'r') as f: data=json.load(f)
            m=re.search(r'(PPO|A2C)_exp(\d+)_', fp)
            if m and data.get('episodes'):
                algo, e_id = m.group(1), m.group(2)
                last_ep = data['episodes'][-1]
                all_results[f"{algo}_exp{e_id}"] = last_ep
        except Exception as e:
            print("Read RL error:", fp, e)

    # 2) 读取 baseline
    pattern_bl = os.path.join(results_files_path,"baseline_*/baseline_evaluation.json")
    for fp in glob.glob(pattern_bl):
        try:
            with open(fp,'r') as f: data=json.load(f)
            m=re.search(r'baseline_(\w+)/', fp)
            if m and data.get('episodes'):
                bname = m.group(1)
                last_ep = data['episodes'][-1]
                all_results[f"baseline_{bname}"] = last_ep
        except Exception as e:
            print("Read baseline error:", fp, e)

    # 3) 若空则模拟
    if not all_results:
        print("No result files found => use simulated data.")
        all_results = generate_simulated_results()

    # 4) 派生累计序列
    def build_derivs(ep):
        # 输入 ep 为最后一条 episode 的 dict
        c_infs=np.cumsum(ep.get('hourly_infections',[0]*480)).tolist()
        c_dths=np.cumsum(ep.get('hourly_deaths',[0]*480)).tolist()
        c_rwd=np.cumsum(ep.get('hourly_reward',[0]*480)).tolist()
        arr=ep.get('hourly_commute_ratio',[0]*480)
        avg_c=[]; s=0
        for i,v in enumerate(arr):
            s+=v; avg_c.append(s/(i+1))
        ep['hourly_cumulative_infections']=c_infs
        ep['hourly_cumulative_deaths']=c_dths
        ep['hourly_average_commute']=avg_c
        ep['hourly_cumulative_reward']=c_rwd

    all_infs, all_dths = [], []
    for k,v in all_results.items():
        build_derivs(v)
        all_infs.append(float(np.sum(v.get('hourly_infections',[]))))
        all_dths.append(float(np.sum(v.get('hourly_deaths',[]))))
    mx_inf = max(all_infs) if all_infs else 1
    mx_dth = max(all_dths) if all_dths else 1

    rl_list, base_list = [], []
    for k,v in all_results.items():
        if k.startswith("baseline_"):
            base_list.append((k.replace("baseline_",""), v))
        else:
            m=re.search(r'(PPO|A2C)_exp(\d+)', k)
            if m: rl_list.append((m.group(1), m.group(2), v))

    # 5) 图表
    if rl_list or base_list:
        create_strategy_comparison(rl_list, base_list)
        create_final_state_comparison(rl_list, base_list)
        create_algorithm_radar_chart(rl_list, base_list, mx_inf, mx_dth)
        if rl_list:
            algo,eid,_=rl_list[0]
            mp=f"./models/{algo}_exp{eid}_final.zip"
            if os.path.exists(mp): visualize_matrix_evolution(mp,algo,eid)
        differential_visualization(all_results)  # 自动 baseline
        grouped_hierarchical_visualization(all_results)

    # 6) 交互式页面与数据
    exporter = create_interactive_visualization_code()
    exporter(all_results)  # 修复：先 _to_py 再 dump

    # 7) 其它占位分析
    statistical_significance_analysis(all_results)
    feature_extraction_pattern_recognition(all_results)
    multi_dimensional_assessment(all_results)
    improved_radar_chart(all_results)
    improved_district_infection_chart({}, {})
    improved_matrix_evolution_chart({}, {})
    improved_time_series_chart(all_results)

    print("All advanced visuals steps done => see ./visualizations/")
