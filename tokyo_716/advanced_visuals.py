# advanced_visuals.py
# 读取 ./results/**/evaluation.json（或 baseline_evaluation.json），生成全局对比图与交互式可视化
# 与任何 main 解耦；Task‑2 / Task‑3 通用
import os, json, glob, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# ------------------- 工具：安全序列化 -------------------
def _to_py(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {k:_to_py(v) for k,v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_py(v) for v in obj]
    return obj

# ------------------- 读取所有结果 -------------------
def _load_all_results(results_root="./results"):
    all_results = {}     # key -> last_episode
    rl_list = []         # (algo, exp_id, last_ep)
    base_list = []       # (baseline_name, last_ep)

    # RL
    for fp in glob.glob(os.path.join(results_root, "*_exp*/evaluation.json")):
        try:
            with open(fp,"r") as f:
                data = json.load(f)
            if not data.get("episodes"): continue
            last_ep = data["episodes"][-1]
            # 解析 algo/exp_id
            basename = os.path.basename(os.path.dirname(fp))  # e.g., PPO_exp3
            algo, exp_id = basename.split("_exp")
            key = f"{algo}_exp{exp_id}"
            all_results[key] = _patch_derivatives(last_ep)
            rl_list.append((algo, exp_id, all_results[key]))
        except Exception as e:
            print(f"[WARN] fail to parse {fp}: {e}")

    # baselines
    for fp in glob.glob(os.path.join(results_root, "baseline_*/baseline_evaluation.json")):
        try:
            with open(fp,"r") as f:
                data = json.load(f)
            if not data.get("episodes"): continue
            last_ep = data["episodes"][-1]
            bname = os.path.basename(os.path.dirname(fp)).replace("baseline_","")
            key = f"baseline_{bname}"
            all_results[key] = _patch_derivatives(last_ep)
            base_list.append((bname, all_results[key]))
        except Exception as e:
            print(f"[WARN] fail to parse {fp}: {e}")

    return all_results, rl_list, base_list

def _patch_derivatives(ep):
    # 构造累计/均值序列，方便画图
    arr_inf = np.array(ep.get("hourly_infections", []), dtype=float)
    arr_dth = np.array(ep.get("hourly_deaths", []), dtype=float)
    arr_cmt = np.array(ep.get("hourly_commute_ratio", []), dtype=float)
    arr_rwd = np.array(ep.get("hourly_reward", []), dtype=float)

    c_inf = np.cumsum(arr_inf)
    c_dth = np.cumsum(arr_dth)
    c_rwd = np.cumsum(arr_rwd)
    avg_c = pd.Series(arr_cmt).expanding().mean().values if len(arr_cmt)>0 else np.array([])

    ep["hourly_cumulative_infections"] = c_inf
    ep["hourly_cumulative_deaths"] = c_dth
    ep["hourly_cumulative_reward"] = c_rwd
    ep["hourly_average_commute"] = avg_c
    return ep

# ------------------- 画图：单图四联（全局比较） -------------------
def create_strategy_comparison(rl_results, base_results, save_path="./visualizations/strategy_comparison.png"):
    plt.figure(figsize=(16,20))
    metrics = [
        ("hourly_cumulative_infections","Cumulative Infections"),
        ("hourly_cumulative_deaths","Cumulative Deaths"),
        ("hourly_average_commute","Average Commute Ratio"),
        ("hourly_cumulative_reward","Cumulative Reward")
    ]
    for i,(m,t) in enumerate(metrics):
        plt.subplot(4,1,i+1)
        for algo,eid,res in rl_results:
            if m in res: plt.plot(res[m], label=f"{algo}_exp{eid}")
        for bname,bres in base_results:
            if m in bres: plt.plot(bres[m], label=f"baseline_{bname}", linestyle="--", linewidth=2)
        plt.title(t); plt.grid(alpha=.3); plt.legend()
    plt.tight_layout(); plt.savefig(save_path); plt.close()

# ------------------- 雷达图（覆盖所有实验与全部基线） -------------------
def create_algorithm_radar_chart(rl_results, base_results, save_path="./visualizations/algorithm_radar_chart.png"):
    metrics = ["Infection Control","Death Prevention","Commute Preservation","Computational Efficiency","Policy Stability"]
    algos=[]; values=[]

    # 归一化基准
    sums_inf=[]; sums_dth=[]
    for a,e,r in rl_results:
        sums_inf.append(np.sum(r.get("hourly_infections",[])))
        sums_dth.append(np.sum(r.get("hourly_deaths",[])))
    for b,r in base_results:
        sums_inf.append(np.sum(r.get("hourly_infections",[])))
        sums_dth.append(np.sum(r.get("hourly_deaths",[])))
    mx_inf = max(sums_inf) if sums_inf else 1.0
    mx_dth = max(sums_dth) if sums_dth else 1.0

    def _pack(name, res, eff=0.8, stab=0.9):
        inf = np.sum(res.get("hourly_infections",[]))
        dth = np.sum(res.get("hourly_deaths",[]))
        cmt = np.mean(res.get("hourly_commute_ratio",[0]))
        return [
            1.0 - (inf/mx_inf if mx_inf>0 else 0),
            1.0 - (dth/mx_dth if mx_dth>0 else 0),
            float(cmt),
            eff, stab
        ]

    for a,e,r in rl_results:
        algos.append(f"{a}_exp{e}")
        eff = 0.6 if a=="PPO" else 0.8
        stab= max(0,min(1, 0.9 - np.std(r.get("hourly_commute_ratio",[0]))))
        values.append(_pack(algos[-1], r, eff, stab))

    for b,r in base_results:
        algos.append(f"baseline_{b}")
        eff = 1.0
        stab= 1.0 if "random" not in b else 0.5
        values.append(_pack(algos[-1], r, eff, stab))

    theta = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)
    theta = np.concatenate([theta, theta[:1]])
    plt.figure(figsize=(14,10))
    ax = plt.subplot(111, polar=True)
    for i, name in enumerate(algos):
        data = np.array(values[i]); data = np.concatenate([data, data[:1]])
        ax.plot(theta, data, linewidth=2, label=name)
        ax.fill(theta, data, alpha=0.08)
    ax.set_xticks(theta[:-1]); ax.set_xticklabels(metrics)
    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.05))
    plt.savefig(save_path); plt.close()

# ------------------- 差分图（自动选择 baseline） -------------------
def differential_visualization(results_dict, save_path="./visualizations/differential_analysis.png"):
    # 自动选择基线：优先 baseline_original；否则平均通勤率最大的 baseline
    base_key = None
    candidates = [k for k in results_dict.keys() if k.startswith("baseline_")]
    if "baseline_original" in candidates:
        base_key = "baseline_original"
    elif candidates:
        # 选择通勤均值最高的 baseline
        best_key = None; best_val = -1
        for k in candidates:
            arr = results_dict[k].get("hourly_commute_ratio", [])
            val = float(np.mean(arr)) if len(arr)>0 else 0.0
            if val > best_val: best_val, best_key = val, k
        base_key = best_key
    else:
        print("No baseline found => skip differential_visualization.")
        return

    base = results_dict[base_key]
    plt.figure(figsize=(16,20))
    items = [
        ("hourly_infections","Infections (% diff)"),
        ("hourly_deaths","Deaths (% diff)"),
        ("hourly_commute_ratio","Commute (% diff)"),
        ("hourly_reward","Reward (% diff)")
    ]
    for i,(m,t) in enumerate(items):
        plt.subplot(4,1,i+1)
        for k,v in results_dict.items():
            if k==base_key: continue
            if m in v and m in base:
                n = min(len(v[m]), len(base[m]));
                b = np.array(base[m][:n], dtype=float)
                r = np.array(v[m][:n], dtype=float)
                with np.errstate(divide='ignore',invalid='ignore'):
                    diff = (r-b)/(np.abs(b)+1e-12)
                diff = np.clip(np.nan_to_num(diff, nan=0, posinf=1, neginf=-1), -1, 1)
                plt.plot(diff, label=k)
        plt.axhline(y=0,color='k',linestyle='--',alpha=.3); plt.title(t); plt.grid(alpha=.3)
        if i==0: plt.legend(bbox_to_anchor=(1.05,1), loc="upper left")
    plt.tight_layout(); plt.savefig(save_path); plt.close()

# ------------------- 分组多图 -------------------
def grouped_hierarchical_visualization(results_dict, save_path="./visualizations/grouped_results.png"):
    ppo = {k:v for k,v in results_dict.items() if k.startswith("PPO")}
    a2c = {k:v for k,v in results_dict.items() if k.startswith("A2C")}
    bas = {k:v for k,v in results_dict.items() if k.startswith("baseline")}
    groups = {"PPO": ppo, "A2C": a2c, "Baseline": bas}

    import math
    from matplotlib import gridspec
    metrics=["hourly_infections","hourly_deaths","hourly_commute_ratio","hourly_reward"]
    titles=["Infections","Deaths","Commute","Reward"]
    fig=plt.figure(figsize=(20,5*len(groups)))
    gs=gridspec.GridSpec(len(groups), len(metrics))
    row=0
    for gname, gdict in groups.items():
        for col,(m,t) in enumerate(zip(metrics,titles)):
            ax = fig.add_subplot(gs[row,col])
            for k,v in gdict.items():
                if m in v: ax.plot(v[m],label=k,alpha=0.8)
            ax.set_title(f"{gname} - {t}"); ax.grid(alpha=.3); ax.legend(fontsize=8)
        row+=1
    plt.tight_layout(); plt.savefig(save_path); plt.close()

# ------------------- 交互式可视化 -------------------
def create_interactive_visualization(results_dict, out_dir="./interactive"):
    os.makedirs(out_dir, exist_ok=True)
    # 写 HTML
    html = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Interactive Visualization</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
<style>
body{font-family:Arial;margin:20px;} .container{display:flex;flex-wrap:wrap;}
.chart{width:48%;margin:1%;height:380px;} .panel{margin-bottom:10px}
button,select{margin:5px;padding:6px 10px}
</style></head><body>
<h2>Interactive RL Epidemic Control (Task-2/3)</h2>
<div class="panel">
<select id="filter"><option value="all">All</option><option value="PPO">PPO</option><option value="A2C">A2C</option><option value="baseline">Baseline</option></select>
<input type="range" id="tw" min="0" max="480" value="480"/><span id="twv">480</span>
<button id="diff">Toggle Diff</button><button id="logy">Log Y</button>
</div>
<div class="container">
<canvas class="chart" id="c1"></canvas><canvas class="chart" id="c2"></canvas>
<canvas class="chart" id="c3"></canvas><canvas class="chart" id="c4"></canvas>
</div>
<script>
let allData={}; let diff=false, logy=false, tw=480, baseKey=null;
async function init(){
 const r=await fetch('results_data.json'); allData=await r.json();
 // 自动选择 baseline (average commute 最大)
 let bk=null, bv=-1;
 Object.keys(allData).forEach(k=>{
   if(k.startsWith('baseline_')){
     const arr=allData[k]['hourly_commute_ratio']||[];
     const mean = arr.length?arr.reduce((a,b)=>a+b,0)/arr.length:0;
     if(mean>bv){bv=mean; bk=k;}
   }
 });
 baseKey=bk;
 draw();
}
function deriveDiff(d){
 if(!baseKey || !d[baseKey]) return d;
 const base=d[baseKey]; const out={};
 for(const k in d){
  if(k===baseKey){ out[k]=d[k]; continue;}
  out[k]={}; ["hourly_infections","hourly_deaths","hourly_commute_ratio","hourly_reward"].forEach(m=>{
    const a=d[k][m]||[], b=base[m]||[]; const n=Math.min(a.length,b.length);
    const arr=[]; for(let i=0;i<n;i++){ const bb=Math.abs(b[i])>1e-12?b[i]:0; const val=bb? (a[i]-b[i])/Math.abs(bb):a[i]; arr.push(Math.max(-1,Math.min(1,val))); }
    out[k][m]=arr;
  });
 }
 return out;
}
function draw(){
 const f=document.getElementById('filter').value;
 let data={}; Object.keys(allData).forEach(k=>{
   if(f==='all' || (f==='baseline'&&k.startsWith('baseline')) || (f==='PPO'&&k.startsWith('PPO')) || (f==='A2C'&&k.startsWith('A2C'))) data[k]=allData[k];
 });
 if(diff) data=deriveDiff(data);
 render('c1', data,'hourly_infections', diff?'Infections (% diff)':'Hourly infections');
 render('c2', data,'hourly_deaths',     diff?'Deaths (% diff)':'Hourly deaths');
 render('c3', data,'hourly_commute_ratio', diff?'Commute (% diff)':'Hourly commute ratio');
 render('c4', data,'hourly_reward',     diff?'Reward (% diff)':'Hourly reward');
}
function render(id,data,metric,title){
 const ctx=document.getElementById(id).getContext('2d'); if(window[id]) window[id].destroy();
 const ds=[]; const colors=d3.schemeCategory10; let i=0;
 Object.keys(data).forEach(k=>{ const arr=(data[k][metric]||[]).slice(0,tw);
   ds.push({label:k, data:arr, borderColor:colors[i%10], backgroundColor:colors[i%10]+'33', borderWidth:k.startsWith('baseline')?2:1, pointRadius:0}); i++;});
 window[id]=new Chart(ctx,{type:'line', data:{labels:Array(tw).fill(0).map((_,i)=>i), datasets:ds},
  options:{responsive:true, plugins:{title:{display:true,text:title}}, scales:{y:{type:(logy&&!diff)?'logarithmic':'linear'}}}});
}
document.getElementById('filter').addEventListener('change',draw);
document.getElementById('tw').addEventListener('input',e=>{tw=parseInt(e.target.value); document.getElementById('twv').textContent=tw; draw();});
document.getElementById('diff').addEventListener('click',()=>{diff=!diff; draw();});
document.getElementById('logy').addEventListener('click',()=>{logy=!logy; draw();});
init();
</script></body></html>"""
    with open(os.path.join(out_dir,"interactive_visualization.html"),"w") as f:
        f.write(html)
    # 写 JSON（修复 ndarray 序列化）
    with open(os.path.join(out_dir,"results_data.json"),"w") as f:
        json.dump(_to_py(results_dict), f)
    print("已生成 ./interactive/interactive_visualization.html & results_data.json")

# ------------------- 主控：跑所有改进 -------------------
def run_all_visualization_improvements(results_files_path="./results/"):
    os.makedirs("./visualizations", exist_ok=True)

    all_results, rl_list, base_list = _load_all_results(results_files_path)
    if not all_results:
        print("No results found under ./results, nothing to visualize.")
        return

    # 全局四联图（含 cumulative reward）
    create_strategy_comparison(rl_list, base_list, "./visualizations/strategy_comparison.png")
    # 雷达图（覆盖所有）
    create_algorithm_radar_chart(rl_list, base_list, "./visualizations/algorithm_radar_chart.png")
    # 差分图（自动挑 baseline）
    differential_visualization(all_results, "./visualizations/differential_analysis.png")
    # 分组多图
    grouped_hierarchical_visualization(all_results, "./visualizations/grouped_results.png")
    # 交互式
    create_interactive_visualization(all_results)
    print("All advanced visuals done -> ./visualizations/ ; ./interactive/")
