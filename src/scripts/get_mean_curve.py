import wandb
import numpy as np

# 初始化 API 客户端
api = wandb.Api()
# 设置你要查询的项目和 group
project_name = "cleanRL-mujuco"
group_name = "Humanoid-v4__baseline"
entity = "overtheriver861-bupt"

# 获取所有属于同一 group 的 runs
runs = api.runs(path=f"{entity}/{project_name}")
env_id = {
    ""
}
# 假设我们感兴趣的是 'loss' 曲线
returns = []

# 遍历每个 run，获取 loss 曲线
for run in runs:
    # 假设每个 run 都有一个 'loss' 记录
    returns.append(run.summary.get('charts/episodic_return'))

# 计算所有 loss 的平均值
average_loss = np.mean(returns)

print(f"Average loss for group {group_name}: {average_loss}")
