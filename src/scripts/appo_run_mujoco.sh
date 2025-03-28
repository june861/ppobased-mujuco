#!/bin/bash

echo "Mujoco Run Scripts"
exp="mujoco"
seeds=(1 2 3)
update_epochs=(10)
envs=(
    "HumanoidStandup-v4"
    # "Humanoid-v4"
    # "HalfCheetah-v4"
    # "Ant-v4"

    # "Hopper-v4"
    # "Reacher-v4"
    # "Walker2d-v4"
    # "InvertedDoublePendulum-v4"
    # "InvertedPendulum-v4"
)

# 循环遍历每个 seed 值，启动 main.py
for seed in "${seeds[@]}"
do
    for e in "${update_epochs[@]}"
    do
        for env_id in "${envs[@]}"
        do
            echo "appo continous action space"
            appo_yaml="src/conf/con_appo_run.yaml"
            /home/wangchenxu/anaconda3/envs/mujoco_v4/bin/python ./src/algos/mujoco/run.py --seed $seed  --yaml $appo_yaml --env_id $env_id --env_type $exp

            # echo "ppo-clip continous action space"
            # ppoclip_yaml="/home/wangchenxu/ppobased-mujuco/src/conf/con_ppoclip_run.yaml"
            # /home/wangchenxu/anaconda3/envs/mujoco_v4/bin/python ./src/algos/mujoco/run.py --seed $seed  --yaml $appo_yaml --env_id $env_id --env_type $exp         
        done
        echo "Experiment with seed=$seed finished."
    done
done

