#!/bin/bash

echo "Mujoco Run Scripts"
exp="mujoco"
algo="ppo-penalty"
seeds=(1 2 3)
update_epochs=(10)
envs=(
    # "HumanoidStandup-v4"
    # "Humanoid-v4"
    # "HalfCheetah-v4"
    # "Ant-v4"

    "Hopper-v4"
    "Reacher-v4"
    "Walker2d-v4"
    "InvertedDoublePendulum-v4"
    "InvertedPendulum-v4"
)

# set root dir, it will return "src/" abs path
PROJECT_ROOT=$(dirname "$(dirname "$(realpath "$0")")")
CONFIG_PATH="$PROJECT_ROOT/conf/con_ppo1_run.yaml"

# check config file
if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Config file not found at $CONFIG_PATH"
    exit 1
fi


# 循环遍历每个 seed 值，启动 main.py
for env_id in "${envs[@]}"
do
    for seed in "${seeds[@]}"
    do
        for e in "${update_epochs[@]}"
        do
            echo "ppo-penalty continous action space"
            CUDA_VISIBLE_DEVICES=0 python -m src.main --seed $seed  --yaml $CONFIG_PATH --env_id $env_id --env_type $exp --algo $algo

            # echo "ppo-clip continous action space"
            # ppoclip_yaml="/home/wangchenxu/ppobased-mujuco/src/conf/con_ppoclip_run.yaml"
            # /home/wangchenxu/anaconda3/envs/mujoco_v4/bin/python ./src/algos/mujoco/run.py --seed $seed  --yaml $appo_yaml --env_id $env_id --env_type $exp         
        done
        echo "Experiment with seed=$seed finished."
    done
done

