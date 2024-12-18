import wandb
import numpy as np
import matplotlib.pyplot as plt

# 初始化API接口
api = wandb.Api()

# 指定你的实体（用户或团队名称）和项目名称
entity = "overtheriver861-bupt"
project = "cleanRL-mujuco"

# 获取项目中的所有run
runs = api.runs(f"{entity}/{project}")

groups = {}
for run in runs:
    if run.group is not None:  # 只考虑有分组的run
        if run.group not in groups:
            groups[run.group] = []
        groups[run.group].append(run)

# 存储每个group的平均值
group_averages = {}

for group_name, group_runs in groups.items():
    metric_data = []  # 存储该组中所有run的metric数据
    for run in group_runs:
        history = run.scan_history()  # 获取run的历史记录
        metric_values = [record['episodic_return'] for record in history if 'episodic_return' in record]
        metric_data.append(metric_values)
    
    # 确保所有的run有相同数量的时间步长
    min_length = min(len(values) for values in metric_data)
    trimmed_metric_data = [values[:min_length] for values in metric_data]

    # 计算平均值
    average_curve = np.mean(trimmed_metric_data, axis=0)
    group_averages[group_name] = average_curve

# 绘制结果
for group_name, avg_curve in group_averages.items():
    plt.plot(avg_curve, label=f"Group {group_name}")

plt.legend()
plt.show()